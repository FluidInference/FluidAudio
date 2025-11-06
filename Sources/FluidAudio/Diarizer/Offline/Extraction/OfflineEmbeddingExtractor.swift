import Accelerate
import CoreML
import Foundation
import OSLog

private struct OfflineEmbeddingPending: Sendable {
    let chunkIndex: Int
    let speakerIndex: Int
    let startFrame: Int
    let endFrame: Int
    let frameWeights: [Float]
    let startTime: Double
    let endTime: Double
    let embedding256: [Float]

    init(
        chunkIndex: Int,
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        frameWeights: [Float],
        startTime: Double,
        endTime: Double,
        embedding256: [Float]
    ) {
        self.chunkIndex = chunkIndex
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.frameWeights = frameWeights
        self.startTime = startTime
        self.endTime = endTime
        self.embedding256 = embedding256
    }
}

private struct OfflineChunkBatchInfo: Sendable {
    let chunkIndex: Int
    let chunkOffsetSeconds: Double
    let frameDuration: Double
    let speakerWeights: [[Float]]

    init(
        chunkIndex: Int,
        chunkOffsetSeconds: Double,
        frameDuration: Double,
        speakerWeights: [[Float]]
    ) {
        self.chunkIndex = chunkIndex
        self.chunkOffsetSeconds = chunkOffsetSeconds
        self.frameDuration = frameDuration
        self.speakerWeights = speakerWeights
    }
}

@available(macOS 14.0, iOS 17.0, *)
struct OfflineEmbeddingExtractor {
    private let fbankModel: MLModel
    private let embeddingModel: MLModel
    private let pldaTransform: PLDATransform
    private let config: OfflineDiarizerConfig
    private let logger = AppLogger(category: "OfflineEmbedding")
    private let memoryOptimizer = ANEMemoryOptimizer()
    private let fbankInputName: String
    private let fbankOutputName: String
    private let fbankFeatureName: String
    private let weightInputName: String
    private let fbankInputShape: [NSNumber]
    private let weightInputShape: [NSNumber]
    private let audioSampleCount: Int
    private let weightFrameCount: Int
    private let modelBatchLimit: Int
    private let embeddingOutputName: String

    init(
        fbankModel: MLModel,
        embeddingModel: MLModel,
        pldaTransform: PLDATransform,
        config: OfflineDiarizerConfig
    ) {
        self.fbankModel = fbankModel
        self.embeddingModel = embeddingModel
        self.pldaTransform = pldaTransform
        self.config = config

        // Resolve FBANK input metadata
        let fbankDescription = fbankModel.modelDescription
        guard
            let audioInput = fbankDescription.inputDescriptionsByName["audio"],
            let audioConstraint = audioInput.multiArrayConstraint
        else {
            logger.error("FBANK model is missing `audio` multiarray input; required for offline pipeline")
            preconditionFailure("FBANK model must expose an `audio` MLMultiArray input")
        }
        self.fbankInputName = "audio"
        let resolvedAudioSamples = OfflineEmbeddingExtractor.resolveElementCount(
            from: audioConstraint,
            fallback: config.samplesPerWindow
        )
        self.audioSampleCount = max(1, min(config.samplesPerWindow, resolvedAudioSamples))
        let audioFallbackShape = OfflineEmbeddingExtractor.defaultShape(
            dimensionHint: OfflineEmbeddingExtractor.dimensionHint(for: audioConstraint),
            minimumCount: 3,
            lastDimension: self.audioSampleCount
        )
        self.fbankInputShape = OfflineEmbeddingExtractor.sanitizedShape(
            from: audioConstraint,
            fallback: audioFallbackShape
        )

        // Resolve FBANK output metadata
        guard
            let fbankOutput = fbankDescription.outputDescriptionsByName["fbank_features"],
            fbankOutput.type == .multiArray
        else {
            logger.error("FBANK model missing `fbank_features` multiarray output")
            preconditionFailure("FBANK model must expose `fbank_features` multiarray output")
        }
        self.fbankOutputName = "fbank_features"

        // Resolve embedding model inputs
        let embeddingDescription = embeddingModel.modelDescription
        let embeddingInputs = embeddingDescription.inputDescriptionsByName
        guard
            let fbankFeatureInput = embeddingInputs["fbank_features"],
            fbankFeatureInput.type == .multiArray
        else {
            logger.error("Embedding model missing `fbank_features` multiarray input")
            preconditionFailure("Embedding model must expose `fbank_features` multiarray input")
        }
        self.fbankFeatureName = "fbank_features"

        guard
            let weightInput = embeddingInputs["weights"],
            let weightConstraint = weightInput.multiArrayConstraint
        else {
            logger.error("Embedding model missing `weights` multiarray input")
            preconditionFailure("Embedding model must expose `weights` MLMultiArray input")
        }
        self.weightInputName = "weights"
        let resolvedWeightFrames = OfflineEmbeddingExtractor.resolveElementCount(
            from: weightConstraint,
            fallback: 589
        )
        self.weightFrameCount = max(1, resolvedWeightFrames)
        let weightFallbackShape = OfflineEmbeddingExtractor.defaultShape(
            dimensionHint: OfflineEmbeddingExtractor.dimensionHint(for: weightConstraint),
            minimumCount: 2,
            lastDimension: self.weightFrameCount
        )
        self.weightInputShape = OfflineEmbeddingExtractor.sanitizedShape(
            from: weightConstraint,
            fallback: weightFallbackShape
        )

        self.modelBatchLimit = max(1, min(config.embeddingBatchSize, 32))

        guard embeddingDescription.outputDescriptionsByName["embedding"] != nil else {
            logger.error("Embedding model missing `embedding` multiarray output")
            preconditionFailure("Embedding model must expose `embedding` multiarray output")
        }
        self.embeddingOutputName = "embedding"

        let audioShapeString = fbankInputShape.map { "\($0.intValue)" }.joined(separator: "×")
        let weightShapeString = weightInputShape.map { "\($0.intValue)" }.joined(separator: "×")
        logger.debug(
            "Offline embedding configured with FBANK input \(fbankInputName)[\(audioShapeString)] → \(fbankOutputName); embedding consumes \(fbankFeatureName) + \(weightInputName)[\(weightShapeString)] (frames=\(weightFrameCount)), maxBatch=\(modelBatchLimit), output=\(embeddingOutputName)"
        )
    }

    func extractEmbeddings(
        audio: [Float],
        segmentation: SegmentationOutput
    ) async throws -> [TimedEmbedding] {
        try await extractEmbeddings(
            audioSource: ArrayAudioSampleSource(samples: audio),
            segmentation: segmentation
        )
    }

    func extractEmbeddings(
        audioSource: StreamingAudioSampleSource,
        segmentation: SegmentationOutput
    ) async throws -> [TimedEmbedding] {
        let stream = AsyncThrowingStream<SegmentationChunk, Error> { continuation in
            for chunkIndex in 0..<segmentation.numChunks {
                guard segmentation.speakerWeights.indices.contains(chunkIndex) else { continue }
                let chunkSpeakerWeights = segmentation.speakerWeights[chunkIndex]
                guard !chunkSpeakerWeights.isEmpty else { continue }

                let chunkOffsetSeconds: Double
                if segmentation.chunkOffsets.indices.contains(chunkIndex) {
                    chunkOffsetSeconds = segmentation.chunkOffsets[chunkIndex]
                } else {
                    chunkOffsetSeconds = Double(chunkIndex) * config.windowDuration
                }

                let chunkLogProbs: [[Float]]
                if segmentation.logProbs.indices.contains(chunkIndex) {
                    chunkLogProbs = segmentation.logProbs[chunkIndex]
                } else {
                    chunkLogProbs = []
                }

                let chunk = SegmentationChunk(
                    chunkIndex: chunkIndex,
                    chunkOffsetSeconds: chunkOffsetSeconds,
                    frameDuration: segmentation.frameDuration,
                    logProbs: chunkLogProbs,
                    speakerWeights: chunkSpeakerWeights
                )
                continuation.yield(chunk)
            }
            continuation.finish()
        }

        return try await extractEmbeddings(
            audioSource: audioSource,
            segmentationStream: stream
        )
    }

    func extractEmbeddings<S: AsyncSequence>(
        audioSource: StreamingAudioSampleSource,
        segmentationStream: S
    ) async throws -> [TimedEmbedding] where S.Element == SegmentationChunk {
        var embeddings: [TimedEmbedding] = []
        embeddings.reserveCapacity(config.embeddingBatchSize * 8)

        let overlapThreshold: Float = 1e-3

        let maxPLDABatch = max(1, min(config.embeddingBatchSize, modelBatchLimit))
        let fbankBatchLimit = min(modelBatchLimit, 32)
        var pendingEmbeddings: [[Float]] = []
        var pendingMetadata: [OfflineEmbeddingPending] = []
        pendingEmbeddings.reserveCapacity(maxPLDABatch)
        pendingMetadata.reserveCapacity(maxPLDABatch)

        var processedMasks = 0
        var fallbackMaskCount = 0
        var emptyMaskCount = 0
        var accumulatedMaskFrames: Double = 0
        let chunkSize = config.samplesPerWindow
        var chunkBuffer = [Float](repeating: 0, count: chunkSize)
        let totalSamples = audioSource.sampleCount
        var batchAudioInputs: [MLMultiArray] = []
        var batchInfos: [OfflineChunkBatchInfo] = []
        batchAudioInputs.reserveCapacity(fbankBatchLimit)
        batchInfos.reserveCapacity(fbankBatchLimit)
        let clock = ContinuousClock()
        var fbankDuration: Duration = .zero
        var maskPreparationDuration: Duration = .zero
        var resampleDuration: Duration = .zero
        var embeddingDuration: Duration = .zero
        var pldaDuration: Duration = .zero
        var evaluatedMaskCount = 0
        var fbankWindowCount = 0
        var fbankBatchCallCount = 0
        var pldaOutputCount = 0
        var pldaBatchCallCount = 0

        func performEmbeddingWarmup() throws {
            let warmupAudioArray = try memoryOptimizer.createAlignedArray(
                shape: fbankInputShape,
                dataType: .float32
            )
            let warmupAudioPointer = warmupAudioArray.dataPointer.assumingMemoryBound(to: Float.self)
            vDSP_vclr(warmupAudioPointer, 1, vDSP_Length(warmupAudioArray.count))

            let warmupFbankFeatures = try runFbankModel(audioArray: warmupAudioArray)
            let zeroWeights = [Float](repeating: 0, count: weightFrameCount)
            let warmupWeightsArray = try prepareWeightsInput(weights: zeroWeights)
            _ = try runEmbeddingModel(
                fbankFeatures: warmupFbankFeatures,
                weightsArray: warmupWeightsArray
            )
        }

        func resolveFrameDuration(_ chunk: SegmentationChunk) -> Double {
            if chunk.frameDuration > 0 {
                return chunk.frameDuration
            }
            let frameCount = max(1, chunk.speakerWeights.count)
            guard frameCount > 0 else {
                return max(1e-3, config.windowDuration)
            }
            return config.windowDuration / Double(frameCount)
        }

        func requiredMinFrames(for frameDuration: Double) -> Int {
            guard frameDuration > 0 else {
                return 1
            }
            let count = Int(ceil(config.minSegmentDuration / frameDuration))
            return max(1, count)
        }

        func flushPending() async throws {
            guard !pendingEmbeddings.isEmpty else { return }
            let pldaStart = clock.now
            let rhoBatch = try await pldaTransform.transform(pendingEmbeddings)
            pldaDuration += pldaStart.duration(to: clock.now)
            pldaOutputCount += rhoBatch.count
            pldaBatchCallCount += 1
            guard rhoBatch.count == pendingMetadata.count else {
                throw OfflineDiarizationError.processingFailed(
                    "PldaRho batch size mismatch (expected \(pendingMetadata.count), got \(rhoBatch.count))"
                )
            }

            for (info, rho) in zip(pendingMetadata, rhoBatch) {
                let timedEmbedding = TimedEmbedding(
                    chunkIndex: info.chunkIndex,
                    speakerIndex: info.speakerIndex,
                    startFrame: info.startFrame,
                    endFrame: info.endFrame,
                    frameWeights: info.frameWeights,
                    startTime: info.startTime,
                    endTime: info.endTime,
                    embedding256: info.embedding256,
                    rho128: rho
                )
                embeddings.append(timedEmbedding)
            }

            pendingEmbeddings.removeAll(keepingCapacity: true)
            pendingMetadata.removeAll(keepingCapacity: true)
        }

        func processChunk(info: OfflineChunkBatchInfo, fbankFeatures: MLMultiArray) async throws {
            let chunkSpeakerWeights = info.speakerWeights
            guard !chunkSpeakerWeights.isEmpty else { return }
            let frameCount = chunkSpeakerWeights.count
            guard let speakerCount = chunkSpeakerWeights.first?.count, speakerCount > 0 else { return }

            let minFramesForEmbedding = requiredMinFrames(for: info.frameDuration)

            var baseMask = [Float](repeating: 0, count: frameCount)
            var cleanMask = [Float](repeating: 0, count: frameCount)
            let overlapFrames: [Bool]
            if config.embeddingExcludeOverlap {
                var frames = [Bool](repeating: false, count: frameCount)
                for (frame, weights) in chunkSpeakerWeights.enumerated() {
                    var active = 0
                    for value in weights where value > overlapThreshold {
                        active += 1
                        if active > 1 {
                            frames[frame] = true
                            break
                        }
                    }
                }
                overlapFrames = frames
            } else {
                overlapFrames = []
            }

            let totalWeightCount = frameCount * speakerCount
            guard totalWeightCount > 0 else { return }

            let rowMajorShape: [NSNumber] = [
                NSNumber(value: frameCount),
                NSNumber(value: speakerCount),
            ]
            let rowMajorKey = "offline_embedding_row_major_\(frameCount)_\(speakerCount)"
            let rowMajorBuffer = try memoryOptimizer.getPooledBuffer(
                key: rowMajorKey,
                shape: rowMajorShape,
                dataType: .float32
            )
            let rowMajorPointer = rowMajorBuffer.dataPointer.assumingMemoryBound(to: Float.self)
            vDSP_vclr(rowMajorPointer, 1, vDSP_Length(totalWeightCount))

            chunkSpeakerWeights.enumerated().forEach { frameIndex, weights in
                let destination = rowMajorPointer.advanced(by: frameIndex * speakerCount)
                weights.withUnsafeBufferPointer { rowPointer in
                    guard let rowPtrBase = rowPointer.baseAddress else { return }
                    destination.update(from: rowPtrBase, count: speakerCount)
                }
            }

            let transposedShape: [NSNumber] = [
                NSNumber(value: speakerCount),
                NSNumber(value: frameCount),
            ]
            let transposedKey = "offline_embedding_transposed_\(speakerCount)_\(frameCount)"
            let transposedBuffer = try memoryOptimizer.getPooledBuffer(
                key: transposedKey,
                shape: transposedShape,
                dataType: .float32
            )
            let transposedPointer = transposedBuffer.dataPointer.assumingMemoryBound(to: Float.self)
            vDSP_mtrans(
                rowMajorPointer,
                1,
                transposedPointer,
                1,
                vDSP_Length(frameCount),
                vDSP_Length(speakerCount)
            )

            for speakerIndex in 0..<speakerCount {
                evaluatedMaskCount += 1
                let maskStart = clock.now

                let columnOffset = speakerIndex * frameCount
                baseMask.withUnsafeMutableBufferPointer { pointer in
                    guard let destBase = pointer.baseAddress else { return }
                    let columnPointer = transposedPointer.advanced(by: columnOffset)
                    destBase.update(from: columnPointer, count: frameCount)
                }

                let baseSum = VDSPOperations.sum(baseMask)
                if baseSum <= 0 {
                    maskPreparationDuration += maskStart.duration(to: clock.now)
                    emptyMaskCount += 1
                    continue
                }

                cleanMask = baseMask
                if config.embeddingExcludeOverlap {
                    for frame in 0..<frameCount where overlapFrames[frame] {
                        cleanMask[frame] = 0
                    }
                }

                let cleanSum = VDSPOperations.sum(cleanMask)
                let maskToUse: [Float]
                let maskSum: Float
                if cleanSum >= Float(minFramesForEmbedding) {
                    maskToUse = cleanMask
                    maskSum = cleanSum
                } else {
                    maskToUse = baseMask
                    maskSum = baseSum
                    fallbackMaskCount += 1
                }

                let maskPrepEnd = clock.now
                maskPreparationDuration += maskStart.duration(to: maskPrepEnd)

                if maskSum <= 0 {
                    emptyMaskCount += 1
                    continue
                }

                let resampleStart = maskPrepEnd
                let resampledMask = WeightInterpolation.resample(maskToUse, to: weightFrameCount)
                let maskEnergy = VDSPOperations.sum(resampledMask)
                let resampleEnd = clock.now
                resampleDuration += resampleStart.duration(to: resampleEnd)
                if maskEnergy <= 0 {
                    emptyMaskCount += 1
                    continue
                }

                let embeddingStart = resampleEnd
                let weightsArray = try prepareWeightsInput(weights: resampledMask)
                let embedding256 = try runEmbeddingModel(
                    fbankFeatures: fbankFeatures,
                    weightsArray: weightsArray
                )
                let embeddingEnd = clock.now
                embeddingDuration += embeddingStart.duration(to: embeddingEnd)

                let firstActive = maskToUse.firstIndex(where: { $0 > overlapThreshold }) ?? 0
                let lastActive = maskToUse.lastIndex(where: { $0 > overlapThreshold }) ?? firstActive
                let startTime = info.chunkOffsetSeconds + Double(firstActive) * info.frameDuration
                let endTime = info.chunkOffsetSeconds + Double(lastActive + 1) * info.frameDuration

                processedMasks += 1
                accumulatedMaskFrames += Double(maskSum)

                pendingEmbeddings.append(embedding256)
                pendingMetadata.append(
                    OfflineEmbeddingPending(
                        chunkIndex: info.chunkIndex,
                        speakerIndex: speakerIndex,
                        startFrame: firstActive,
                        endFrame: lastActive,
                        frameWeights: maskToUse,
                        startTime: startTime,
                        endTime: endTime,
                        embedding256: embedding256
                    )
                )

                if pendingEmbeddings.count == maxPLDABatch {
                    try await flushPending()
                }
            }
        }

        func flushFbankBatch() async throws {
            guard !batchAudioInputs.isEmpty else { return }
            let fbankStart = clock.now
            let fbankOutputs = try runFbankBatch(audioArrays: batchAudioInputs)
            fbankDuration += fbankStart.duration(to: clock.now)
            fbankBatchCallCount += 1
            guard fbankOutputs.count == batchInfos.count else {
                throw OfflineDiarizationError.processingFailed(
                    "FBANK batch produced mismatched output count (\(fbankOutputs.count) vs \(batchInfos.count))"
                )
            }

            for index in 0..<batchInfos.count {
                try await processChunk(info: batchInfos[index], fbankFeatures: fbankOutputs[index])
            }

            batchAudioInputs.removeAll(keepingCapacity: true)
            batchInfos.removeAll(keepingCapacity: true)
        }

        do {
            try performEmbeddingWarmup()
        } catch {
            logger.debug("Embedding warmup skipped due to error: \(error.localizedDescription)")
        }

        for try await chunk in segmentationStream {
            try Task.checkCancellation()

            let chunkSpeakerWeights = chunk.speakerWeights
            guard !chunkSpeakerWeights.isEmpty else { continue }

            let frameDuration = resolveFrameDuration(chunk)
            let chunkOffsetSeconds =
                chunk.chunkOffsetSeconds.isFinite
                ? chunk.chunkOffsetSeconds
                : Double(chunk.chunkIndex) * config.windowDuration

            let estimatedStartSample = Int((chunkOffsetSeconds * Double(config.sampleRate)).rounded())
            let clampedStartSample = max(0, min(estimatedStartSample, totalSamples))
            let endSample = min(clampedStartSample + chunkSize, totalSamples)
            guard clampedStartSample < endSample else {
                continue
            }

            chunkBuffer.withUnsafeMutableBufferPointer { pointer in
                vDSP_vclr(pointer.baseAddress!, 1, vDSP_Length(pointer.count))
            }
            try chunkBuffer.withUnsafeMutableBufferPointer { pointer in
                guard let baseAddress = pointer.baseAddress else { return }
                try audioSource.copySamples(
                    into: baseAddress,
                    offset: clampedStartSample,
                    count: chunkSize
                )
            }

            let chunkLength = endSample - clampedStartSample
            let fbankInput = try chunkBuffer.withUnsafeBufferPointer { pointer -> MLMultiArray in
                guard let baseAddress = pointer.baseAddress else {
                    throw OfflineDiarizationError.processingFailed("Failed to access chunk buffer")
                }
                return try prepareFbankInput(
                    chunkPointer: baseAddress,
                    length: chunkLength
                )
            }

            fbankWindowCount += 1
            batchAudioInputs.append(fbankInput)
            batchInfos.append(
                OfflineChunkBatchInfo(
                    chunkIndex: chunk.chunkIndex,
                    chunkOffsetSeconds: chunkOffsetSeconds,
                    frameDuration: frameDuration,
                    speakerWeights: chunkSpeakerWeights
                )
            )

            if batchAudioInputs.count == fbankBatchLimit {
                try await flushFbankBatch()
            }
        }

        try await flushFbankBatch()
        try await flushPending()

        if processedMasks > 0 {
            let meanMaskFrames = accumulatedMaskFrames / Double(processedMasks)
            let meanString = String(format: "%.2f", meanMaskFrames)
            logger.debug(
                "Embedding masks generated: \(embeddings.count) (meanActiveFrames=\(meanString), fallbackMasks=\(fallbackMaskCount), emptyMasks=\(emptyMaskCount))"
            )
        } else {
            logger.debug("Embedding extractor produced no valid speaker masks")
        }

        if fbankWindowCount > 0 || evaluatedMaskCount > 0 || pldaOutputCount > 0 {
            let fbankMs = Self.milliseconds(from: fbankDuration)
            let maskPrepMs = Self.milliseconds(from: maskPreparationDuration)
            let resampleMs = Self.milliseconds(from: resampleDuration)
            let embeddingMs = Self.milliseconds(from: embeddingDuration)
            let pldaMs = Self.milliseconds(from: pldaDuration)

            let fbankPerWindow = fbankWindowCount > 0 ? fbankMs / Double(fbankWindowCount) : 0
            let maskPrepPerEval = evaluatedMaskCount > 0 ? maskPrepMs / Double(evaluatedMaskCount) : 0
            let resamplePerValid = processedMasks > 0 ? resampleMs / Double(processedMasks) : 0
            let embeddingPerValid = processedMasks > 0 ? embeddingMs / Double(processedMasks) : 0
            let pldaPerValid = processedMasks > 0 ? pldaMs / Double(processedMasks) : 0
            let message =
                """
                Embedding timings: fbankTotal=\(String(format: "%.2f", fbankMs))ms (perWindow=\(String(format: "%.3f", fbankPerWindow))ms), \
                maskPrepTotal=\(String(format: "%.2f", maskPrepMs))ms (perEval=\(String(format: "%.3f", maskPrepPerEval))ms), \
                resampleTotal=\(String(format: "%.2f", resampleMs))ms (perValid=\(String(format: "%.3f", resamplePerValid))ms), \
                embeddingTotal=\(String(format: "%.2f", embeddingMs))ms (perValid=\(String(format: "%.3f", embeddingPerValid))ms), \
                pldaTotal=\(String(format: "%.2f", pldaMs))ms (perValid=\(String(format: "%.3f", pldaPerValid))ms), \
                batches(fbank=\(fbankBatchCallCount), plda=\(pldaBatchCallCount))
                """
            logger.debug(message)
            Self.emitProfileLog(message)
        }

        return embeddings
    }

    private func runFbankModel(
        audioArray: MLMultiArray
    ) throws -> MLMultiArray {
        guard let result = try runFbankBatch(audioArrays: [audioArray]).first else {
            throw OfflineDiarizationError.processingFailed("FBANK model produced no output")
        }
        return result
    }

    private func runFbankBatch(
        audioArrays: [MLMultiArray]
    ) throws -> [MLMultiArray] {
        guard !audioArrays.isEmpty else { return [] }
        var providers: [MLFeatureProvider] = []
        providers.reserveCapacity(audioArrays.count)
        for array in audioArrays {
            providers.append(
                ZeroCopyDiarizerFeatureProvider(
                    features: [
                        fbankInputName: MLFeatureValue(multiArray: array)
                    ]
                )
            )
        }

        let options = MLPredictionOptions()
        if #available(macOS 14.0, iOS 17.0, *) {
            for array in audioArrays {
                array.prefetchToNeuralEngine()
            }
        }

        let batchProvider = MLArrayBatchProvider(array: providers)
        let outputBatch = try fbankModel.predictions(from: batchProvider, options: options)
        guard outputBatch.count == audioArrays.count else {
            throw OfflineDiarizationError.processingFailed(
                "FBANK batch produced \(outputBatch.count) outputs for \(audioArrays.count) inputs"
            )
        }

        var results: [MLMultiArray] = []
        results.reserveCapacity(outputBatch.count)
        for index in 0..<outputBatch.count {
            guard
                let featureArray = outputBatch.features(at: index)
                    .featureValue(for: fbankOutputName)?.multiArrayValue
            else {
                throw OfflineDiarizationError.processingFailed(
                    "FBANK model missing \(fbankOutputName) output at batch index \(index)"
                )
            }
            results.append(featureArray)
        }

        return results
    }

    private func prepareFbankInput(
        chunkPointer: UnsafePointer<Float>,
        length: Int
    ) throws -> MLMultiArray {
        let array = try memoryOptimizer.createAlignedArray(
            shape: fbankInputShape,
            dataType: .float32
        )

        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        vDSP_vclr(pointer, 1, vDSP_Length(array.count))

        let copyCount = min(length, audioSampleCount, array.count)
        if copyCount > 0 {
            vDSP_mmov(
                chunkPointer,
                pointer,
                vDSP_Length(copyCount),
                1,
                vDSP_Length(copyCount),
                1
            )
        }

        return array
    }

    private func prepareWeightsInput(
        weights: [Float]
    ) throws -> MLMultiArray {
        let array = try memoryOptimizer.createAlignedArray(
            shape: weightInputShape,
            dataType: .float32
        )

        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        vDSP_vclr(pointer, 1, vDSP_Length(array.count))

        let copyCount = min(weights.count, weightFrameCount, array.count)
        if copyCount > 0 {
            weights.withUnsafeBufferPointer { buffer in
                vDSP_mmov(
                    buffer.baseAddress!,
                    pointer,
                    vDSP_Length(copyCount),
                    1,
                    vDSP_Length(copyCount),
                    1
                )
            }
        }

        return array
    }

    private func runEmbeddingModel(
        fbankFeatures: MLMultiArray,
        weightsArray: MLMultiArray
    ) throws -> [Float] {
        let provider = ZeroCopyDiarizerFeatureProvider(
            features: [
                fbankFeatureName: MLFeatureValue(multiArray: fbankFeatures),
                weightInputName: MLFeatureValue(multiArray: weightsArray),
            ]
        )
        let options = MLPredictionOptions()
        if #available(macOS 14.0, iOS 17.0, *) {
            fbankFeatures.prefetchToNeuralEngine()
            weightsArray.prefetchToNeuralEngine()
        }

        let output = try embeddingModel.prediction(from: provider, options: options)
        guard let embeddingArray = output.featureValue(for: embeddingOutputName)?.multiArrayValue else {
            throw OfflineDiarizationError.processingFailed("Embedding model missing \(embeddingOutputName) output")
        }

        let pointer = embeddingArray.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: embeddingArray.count))
    }

    private static func milliseconds(from duration: Duration) -> Double {
        let components = duration.components
        let secondsMs = Double(components.seconds) * 1_000
        let attosecondsMs = Double(components.attoseconds) / 1_000_000_000_000_000.0
        return secondsMs + attosecondsMs
    }

    private static func emitProfileLog(_ message: String) {
        let line = "[Profiling] \(message)\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }

    private static func resolveElementCount(
        from constraint: MLMultiArrayConstraint?,
        fallback: Int
    ) -> Int {
        guard let shape = constraint?.shape, !shape.isEmpty else {
            return fallback
        }

        if let last = shape.last {
            let value = last.intValue
            if value > 0 {
                return value
            }
        }

        if let secondLast = shape.dropLast().last {
            let value = secondLast.intValue
            if value > 0 {
                return value
            }
        }

        return fallback
    }

    private static func sanitizedShape(
        from constraint: MLMultiArrayConstraint,
        fallback: [Int]
    ) -> [NSNumber] {
        if let enumerated = constraint.shapeConstraint.enumeratedShapes.first, !enumerated.isEmpty {
            return sanitizedShape(enumerated, fallback: fallback)
        }

        let explicitShape = constraint.shape
        if !explicitShape.isEmpty {
            return sanitizedShape(explicitShape, fallback: fallback)
        }

        let ranges = constraint.shapeConstraint.sizeRangeForDimension
        if !ranges.isEmpty {
            var sanitized: [NSNumber] = []
            sanitized.reserveCapacity(ranges.count)
            for (index, rangeValue) in ranges.enumerated() {
                let range = rangeValue.rangeValue
                let fallbackValue = fallbackValue(fallback, index: index)
                let candidate = range.location > 0 ? range.location : fallbackValue
                sanitized.append(NSNumber(value: max(1, candidate)))
            }
            return sanitized
        }

        return fallback.map { NSNumber(value: max(1, $0)) }
    }

    private static func sanitizedShape(
        _ shape: [NSNumber],
        fallback: [Int]
    ) -> [NSNumber] {
        guard !shape.isEmpty else {
            return fallback.map { NSNumber(value: max(1, $0)) }
        }

        var sanitized: [NSNumber] = []
        sanitized.reserveCapacity(shape.count)

        for (index, dimension) in shape.enumerated() {
            let value = dimension.intValue
            if value > 0 {
                sanitized.append(dimension)
            } else {
                let fallbackValue = fallbackValue(fallback, index: index)
                sanitized.append(NSNumber(value: max(1, fallbackValue)))
            }
        }

        return sanitized
    }

    private static func fallbackValue(
        _ fallback: [Int],
        index: Int
    ) -> Int {
        guard !fallback.isEmpty else {
            return 1
        }
        if index < fallback.count {
            return fallback[index]
        }
        return fallback.last ?? 1
    }

    private static func defaultShape(
        dimensionHint: Int,
        minimumCount: Int,
        lastDimension: Int
    ) -> [Int] {
        let count = max(max(dimensionHint, minimumCount), 1)
        var shape = Array(repeating: 1, count: count)
        shape[count - 1] = max(1, lastDimension)
        return shape
    }

    private static func dimensionHint(for constraint: MLMultiArrayConstraint) -> Int {
        if let enumerated = constraint.shapeConstraint.enumeratedShapes.first, !enumerated.isEmpty {
            return enumerated.count
        }
        let explicitCount = constraint.shape.count
        if explicitCount > 0 {
            return explicitCount
        }
        let rangeCount = constraint.shapeConstraint.sizeRangeForDimension.count
        if rangeCount > 0 {
            return rangeCount
        }
        return 0
    }
}
