import Accelerate
import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
struct OfflineEmbeddingExtractor {
    private let embeddingModel: MLModel
    private let pldaTransform: PLDATransform
    private let config: OfflineDiarizerConfig
    private let logger = AppLogger(category: "OfflineEmbedding")
    private let memoryOptimizer = ANEMemoryOptimizer()

    private let audioSampleCount: Int
    private let weightFrameCount: Int
    private let embeddingOutputName: String

    init(
        embeddingModel: MLModel,
        pldaTransform: PLDATransform,
        config: OfflineDiarizerConfig
    ) {
        self.embeddingModel = embeddingModel
        self.pldaTransform = pldaTransform
        self.config = config

        let audioConstraint = embeddingModel.modelDescription.inputDescriptionsByName["audio"]?.multiArrayConstraint
        let weightsConstraint = embeddingModel.modelDescription.inputDescriptionsByName["weights"]?.multiArrayConstraint

        self.audioSampleCount = OfflineEmbeddingExtractor.resolveElementCount(
            from: audioConstraint,
            fallback: config.samplesPerWindow
        )
        self.weightFrameCount = OfflineEmbeddingExtractor.resolveElementCount(
            from: weightsConstraint,
            fallback: 589
        )

        if let name = embeddingModel.modelDescription.outputDescriptionsByName.keys.first(where: { $0 == "embedding" })
        {
            self.embeddingOutputName = name
        } else if let name = embeddingModel.modelDescription.outputDescriptionsByName.keys.first(where: {
            $0.contains("embedding")
        }) {
            self.embeddingOutputName = name
        } else {
            self.embeddingOutputName =
                embeddingModel.modelDescription.outputDescriptionsByName.keys.first ?? "embedding"
        }

        logger.debug(
            "Offline embedding model resolved input shapes audio=\(audioSampleCount), weights=\(weightFrameCount), output=\(embeddingOutputName)"
        )
    }

    func extractEmbeddings(
        audio: [Float],
        segmentation: SegmentationOutput
    ) throws -> [TimedEmbedding] {
        var embeddings: [TimedEmbedding] = []
        embeddings.reserveCapacity(segmentation.numChunks * segmentation.numSpeakers)

        let chunkSize = min(config.samplesPerWindow, audioSampleCount)
        let frameDuration = segmentation.frameDuration
        let minFramesForEmbedding: Int
        if frameDuration > 0 {
            minFramesForEmbedding = max(
                1,
                Int(ceil(config.minSegmentDuration / frameDuration))
            )
        } else {
            minFramesForEmbedding = 1
        }
        let overlapThreshold: Float = 1e-3

        struct PendingEmbedding {
            let chunkIndex: Int
            let speakerIndex: Int
            let startFrame: Int
            let endFrame: Int
            let frameWeights: [Float]
            let startTime: Double
            let endTime: Double
            let embedding256: [Float]
        }

        let maxPLDABatch = max(1, min(config.embeddingBatchSize, 32))
        var pendingEmbeddings: [[Float]] = []
        var pendingMetadata: [PendingEmbedding] = []
        pendingEmbeddings.reserveCapacity(maxPLDABatch)
        pendingMetadata.reserveCapacity(maxPLDABatch)

        func flushPending() throws {
            guard !pendingEmbeddings.isEmpty else { return }
            let rhoBatch = try pldaTransform.transform(pendingEmbeddings)
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

        var processedMasks = 0
        var fallbackMaskCount = 0
        var emptyMaskCount = 0
        var accumulatedMaskFrames: Double = 0

        for chunkIndex in 0..<segmentation.numChunks {
            let chunkOffsetSeconds: Double
            if segmentation.chunkOffsets.indices.contains(chunkIndex) {
                chunkOffsetSeconds = segmentation.chunkOffsets[chunkIndex]
            } else {
                chunkOffsetSeconds = Double(chunkIndex) * config.windowDuration
            }

            let estimatedStartSample = Int((chunkOffsetSeconds * Double(config.sampleRate)).rounded())
            let clampedStartSample = max(0, min(estimatedStartSample, audio.count))
            let endSample = min(clampedStartSample + config.samplesPerWindow, audio.count)
            guard clampedStartSample < endSample else {
                continue
            }

            let chunkAudio = Array(audio[clampedStartSample..<endSample])
            let audioArray = try prepareAudioArray(chunk: chunkAudio, chunkSize: chunkSize)

            guard segmentation.speakerWeights.indices.contains(chunkIndex) else {
                continue
            }
            let chunkSpeakerWeights = segmentation.speakerWeights[chunkIndex]
            guard !chunkSpeakerWeights.isEmpty else {
                continue
            }

            let frameCount = chunkSpeakerWeights.count
            guard let speakerCount = chunkSpeakerWeights.first?.count else {
                continue
            }

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

            for speakerIndex in 0..<speakerCount {
                baseMask.withUnsafeMutableBufferPointer { pointer in
                    vDSP_vclr(pointer.baseAddress!, 1, vDSP_Length(frameCount))
                }
                for (frame, weights) in chunkSpeakerWeights.enumerated() where speakerIndex < weights.count {
                    baseMask[frame] = weights[speakerIndex]
                }

                let baseSum = VDSPOperations.sum(baseMask)
                if baseSum <= 0 {
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

                if maskSum <= 0 {
                    emptyMaskCount += 1
                    continue
                }

                let resampledMask = WeightInterpolation.resample(maskToUse, to: weightFrameCount)
                let maskEnergy = VDSPOperations.sum(resampledMask)
                if maskEnergy <= 0 {
                    emptyMaskCount += 1
                    continue
                }

                let weightsArray = try prepareWeightsArray(mask: resampledMask)
                let embedding256 = try runEmbeddingModel(
                    audioArray: audioArray,
                    weightsArray: weightsArray
                )

                let firstActive = maskToUse.firstIndex(where: { $0 > overlapThreshold }) ?? 0
                let lastActive = maskToUse.lastIndex(where: { $0 > overlapThreshold }) ?? firstActive
                let startTime = chunkOffsetSeconds + Double(firstActive) * frameDuration
                let endTime = chunkOffsetSeconds + Double(lastActive + 1) * frameDuration

                processedMasks += 1
                accumulatedMaskFrames += Double(maskSum)

                pendingEmbeddings.append(embedding256)
                pendingMetadata.append(
                    PendingEmbedding(
                        chunkIndex: chunkIndex,
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
                    try flushPending()
                }
            }
        }

        try flushPending()

        if processedMasks > 0 {
            let meanMaskFrames = accumulatedMaskFrames / Double(processedMasks)
            let meanString = String(format: "%.2f", meanMaskFrames)
            logger.debug(
                "Embedding masks generated: \(embeddings.count) (meanActiveFrames=\(meanString), fallbackMasks=\(fallbackMaskCount), emptyMasks=\(emptyMaskCount))"
            )
        } else {
            logger.debug("Embedding extractor produced no valid speaker masks")
        }

        return embeddings
    }

    private func prepareAudioArray(chunk: [Float], chunkSize: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: audioSampleCount)]
        let array = try memoryOptimizer.createAlignedArray(
            shape: shape,
            dataType: .float32
        )
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)

        var zero: Float = 0
        vDSP_vfill(&zero, pointer, 1, vDSP_Length(audioSampleCount))

        chunk.withUnsafeBufferPointer { buffer in
            vDSP_mmov(
                buffer.baseAddress!,
                pointer,
                vDSP_Length(min(buffer.count, chunkSize)),
                1,
                vDSP_Length(min(buffer.count, chunkSize)),
                1
            )
        }

        return array
    }

    private func prepareWeightsArray(mask: [Float]) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: weightFrameCount)]
        let array = try memoryOptimizer.createAlignedArray(
            shape: shape,
            dataType: .float32
        )
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)

        mask.withUnsafeBufferPointer { buffer in
            vDSP_mmov(
                buffer.baseAddress!,
                pointer,
                vDSP_Length(weightFrameCount),
                1,
                vDSP_Length(weightFrameCount),
                1
            )
        }

        return array
    }

    private func runEmbeddingModel(
        audioArray: MLMultiArray,
        weightsArray: MLMultiArray
    ) throws -> [Float] {
        let provider = ZeroCopyDiarizerFeatureProvider(
            features: [
                "audio": MLFeatureValue(multiArray: audioArray),
                "weights": MLFeatureValue(multiArray: weightsArray),
            ]
        )
        let options = MLPredictionOptions()
        if #available(macOS 14.0, iOS 17.0, *) {
            audioArray.prefetchToNeuralEngine()
            weightsArray.prefetchToNeuralEngine()
        }

        let output = try embeddingModel.prediction(from: provider, options: options)
        guard let embeddingArray = output.featureValue(for: embeddingOutputName)?.multiArrayValue else {
            throw OfflineDiarizationError.processingFailed("Embedding model missing \(embeddingOutputName) output")
        }

        let pointer = embeddingArray.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: embeddingArray.count))
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
}
