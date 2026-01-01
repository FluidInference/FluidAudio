import Accelerate
@preconcurrency import CoreML
import Foundation
import OSLog
import os.signpost

struct OfflineSegmentationProcessor {
    private let logger = AppLogger(category: "OfflineSegmentation")
    private let signposter = OSSignposter(
        subsystem: "com.fluidaudio.diarization",
        category: .pointsOfInterest
    )
    private let memoryOptimizer = ANEMemoryOptimizer()

    private let powerset: [[Int]] = [
        [],
        [0],
        [1],
        [2],
        [0, 1],
        [0, 2],
        [1, 2],
        [0, 1, 2],
    ]

    func process(
        audioSamples: [Float],
        segmentationModel: MLModel,
        config: OfflineDiarizerConfig,
        chunkHandler: SegmentationChunkHandler? = nil
    ) async throws -> SegmentationOutput {
        guard !audioSamples.isEmpty else {
            throw OfflineDiarizationError.noSpeechDetected
        }

        return try await process(
            audioSource: ArrayAudioSampleSource(samples: audioSamples),
            segmentationModel: segmentationModel,
            config: config,
            chunkHandler: chunkHandler
        )
    }

    func process(
        audioSource: StreamingAudioSampleSource,
        segmentationModel: MLModel,
        config: OfflineDiarizerConfig,
        chunkHandler: SegmentationChunkHandler? = nil
    ) async throws -> SegmentationOutput {
        let totalSamples = audioSource.sampleCount
        guard totalSamples > 0 else {
            throw OfflineDiarizationError.noSpeechDetected
        }

        let chunkSize = config.samplesPerWindow
        let stepSize = config.samplesPerStep

        var logProbChunks: [[[Float]]] = []
        var weightChunks: [[[Float]]] = []
        var chunkOffsets: [Double] = []
        var frameDuration: Double = 0
        var numFrames = 0
        let speakerCount = 3

        // Pre-compute flat mapping matrix for vectorized speaker activation
        // Matrix[speaker][class] = 1.0 if speaker in powerset[class], else 0.0
        let speakerToClassMapping: [[Float]] = (0..<speakerCount).map { speaker in
            powerset.map { combination in
                combination.contains(speaker) ? Float(1.0) : Float(0.0)
            }
        }
        var classHistogram = Array(repeating: 0, count: powerset.count)
        var classProbabilitySums = Array(repeating: Float.zero, count: powerset.count)
        let chunkCallback = chunkHandler
        var chunkEmissionEnabled = chunkCallback != nil

        logger.debug(
            "Offline segmentation: chunkSize=\(chunkSize), stepSize=\(stepSize), totalSamples=\(totalSamples)"
        )

        var speechFrameCount = 0
        var winningProbabilitySum: Double = 0
        var winningProbabilityCount = 0
        var winningProbabilityMin: Float = 1
        var winningProbabilityMax: Float = 0
        var emptyClassProbabilitySum: Double = 0
        var emptyClassProbabilityCount = 0
        let probabilityThresholds: [Float] = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
        var probabilityThresholdCounts = Array(repeating: 0, count: probabilityThresholds.count)
        let emptyClassIndex = 0
        let onsetThreshold = config.speechOnsetThreshold

        let batchCapacity = 32
        var globalChunkIndex = 0

        let clock = ContinuousClock()
        var prepareDuration: Duration = .zero
        var predictionDuration: Duration = .zero
        var preparedWindowCount = 0
        var slidingWindow = [Float](repeating: 0, count: chunkSize)
        var previousOffset: Int?
        let reuseEnabled = stepSize < chunkSize

        func performWarmup() async throws {
            let warmupShape: [NSNumber] = [1, 1, NSNumber(value: chunkSize)]
            let warmupKey = "offline_segmentation_warmup_\(chunkSize)"
            let warmupArray = try memoryOptimizer.getPooledBuffer(
                key: warmupKey,
                shape: warmupShape,
                dataType: .float32
            )
            let warmupPointer = warmupArray.dataPointer.assumingMemoryBound(to: Float.self)
            vDSP_vclr(warmupPointer, 1, vDSP_Length(chunkSize))

            let warmupProvider = ZeroCopyDiarizerFeatureProvider(
                features: ["audio": MLFeatureValue(multiArray: warmupArray)]
            )
            let warmupOptions = MLPredictionOptions()
            if #available(macOS 14.0, iOS 17.0, *) {
                warmupArray.prefetchToNeuralEngine()
            }
            _ = try await segmentationModel.prediction(from: warmupProvider, options: warmupOptions)
        }

        func populateWindow(
            destination: UnsafeMutablePointer<Float>,
            offset: Int
        ) throws {
            let availableForWindow = max(0, min(chunkSize, totalSamples - offset))

            if reuseEnabled,
                let lastOffset = previousOffset,
                offset == lastOffset + stepSize
            {
                try slidingWindow.withUnsafeMutableBufferPointer { pointer in
                    guard let base = pointer.baseAddress else { return }
                    let reuseCount = max(0, chunkSize - stepSize)
                    if reuseCount > 0 {
                        memmove(
                            base,
                            base.advanced(by: stepSize),
                            reuseCount * MemoryLayout<Float>.stride
                        )
                    }

                    let samplesNeeded = chunkSize - reuseCount
                    if samplesNeeded > 0 {
                        let tailOffset = offset + reuseCount
                        let available = max(
                            0,
                            min(samplesNeeded, totalSamples - tailOffset)
                        )
                        if available > 0 {
                            try audioSource.copySamples(
                                into: base.advanced(by: reuseCount),
                                offset: tailOffset,
                                count: available
                            )
                        }
                        if available < samplesNeeded {
                            vDSP_vclr(
                                base.advanced(by: reuseCount + available),
                                1,
                                vDSP_Length(samplesNeeded - available)
                            )
                        }
                    }
                }
            } else {
                try slidingWindow.withUnsafeMutableBufferPointer { pointer in
                    guard let base = pointer.baseAddress else { return }
                    if availableForWindow > 0 {
                        try audioSource.copySamples(
                            into: base,
                            offset: offset,
                            count: availableForWindow
                        )
                    }
                    if availableForWindow < chunkSize {
                        vDSP_vclr(
                            base.advanced(by: availableForWindow),
                            1,
                            vDSP_Length(chunkSize - availableForWindow)
                        )
                    }
                }
            }

            slidingWindow.withUnsafeBufferPointer { pointer in
                guard let base = pointer.baseAddress else { return }
                destination.update(from: base, count: chunkSize)
            }
            previousOffset = offset
        }

        var processedAnyBatch = false
        var offsetIterator = stride(from: 0, to: totalSamples, by: stepSize).makeIterator()
        var batchOffsets: [Int] = []
        batchOffsets.reserveCapacity(batchCapacity)

        do {
            try await performWarmup()
        } catch {
            logger.debug("Segmentation warmup skipped due to error: \(error.localizedDescription)")
        }

        while true {
            try Task.checkCancellation()

            batchOffsets.removeAll(keepingCapacity: true)
            for _ in 0..<batchCapacity {
                guard let offset = offsetIterator.next() else { break }
                batchOffsets.append(offset)
            }

            if batchOffsets.isEmpty {
                break
            }

            processedAnyBatch = true
            let batchCount = batchOffsets.count
            let shape: [NSNumber] = [
                NSNumber(value: batchCount),
                1,
                NSNumber(value: chunkSize),
            ]
            let bufferKey = "offline_segmentation_audio_\(batchCount)_\(chunkSize)"
            let audioArray = try memoryOptimizer.getPooledBuffer(
                key: bufferKey,
                shape: shape,
                dataType: .float32
            )

            let ptr = audioArray.dataPointer.assumingMemoryBound(to: Float.self)

            let prepareStart = clock.now
            for (localIndex, offset) in batchOffsets.enumerated() {
                let destination = ptr.advanced(by: localIndex * chunkSize)
                try populateWindow(destination: destination, offset: offset)
            }
            prepareDuration += prepareStart.duration(to: clock.now)
            preparedWindowCount += batchOffsets.count

            let provider = ZeroCopyDiarizerFeatureProvider(
                features: ["audio": MLFeatureValue(multiArray: audioArray)]
            )

            let options = MLPredictionOptions()
            if #available(macOS 14.0, iOS 17.0, *) {
                audioArray.prefetchToNeuralEngine()
            }

            let predictionState = signposter.beginInterval("Segmentation Model Prediction")
            try Task.checkCancellation()
            let predictionStart = clock.now
            let output = try await segmentationModel.prediction(from: provider, options: options)
            predictionDuration += predictionStart.duration(to: clock.now)
            signposter.endInterval("Segmentation Model Prediction", predictionState)

            let logitsArray: MLMultiArray
            if let segments = output.featureValue(for: "segments")?.multiArrayValue {
                logitsArray = segments
            } else if let logProbs = output.featureValue(for: "log_probs")?.multiArrayValue {
                logitsArray = logProbs
            } else if let fallback = output.featureNames.compactMap({ name -> MLMultiArray? in
                output.featureValue(for: name)?.multiArrayValue
            }).first {
                logitsArray = fallback
            } else {
                let available = Array(output.featureNames)
                throw OfflineDiarizationError.processingFailed(
                    "Segmentation model missing expected multiarray output. Available: \(available)"
                )
            }

            let logitsShape = logitsArray.shape.map { $0.intValue }
            let (batchSize, frames, classes): (Int, Int, Int)
            switch logitsShape.count {
            case 3:
                batchSize = logitsShape[0]
                frames = logitsShape[1]
                classes = logitsShape[2]
            case 2:
                batchSize = 1
                frames = logitsShape[0]
                classes = logitsShape[1]
            default:
                throw OfflineDiarizationError.processingFailed(
                    "Unexpected segmentation output shape \(logitsShape)"
                )
            }

            frameDuration = config.windowDuration / Double(frames)
            numFrames = frames

            if classes > powerset.count {
                logger.error(
                    "Segmentation model returned \(classes) classes but only \(powerset.count) powerset entries available"
                )
            }

            let logitsPointer = logitsArray.dataPointer.assumingMemoryBound(to: Float.self)

            for localIndex in 0..<batchCount {
                if localIndex >= batchSize {
                    break
                }

                let offset = batchOffsets[localIndex]
                chunkOffsets.append(Double(offset) / Double(config.sampleRate))

                var chunkLogProbs = Array(
                    repeating: Array(repeating: Float.zero, count: classes),
                    count: frames
                )

                var chunkSpeakerProbs = Array(
                    repeating: Array(repeating: Float.zero, count: speakerCount),
                    count: frames
                )

                let baseIndex = localIndex * frames * classes

                var frameLogits = [Float](repeating: 0, count: classes)
                var logProbabilityBuffer = [Float](repeating: 0, count: classes)
                var probabilityBuffer = [Float](repeating: 0, count: classes)

                for frameIndex in 0..<frames {
                    let start = baseIndex + frameIndex * classes
                    frameLogits.withUnsafeMutableBufferPointer { destination in
                        destination.baseAddress!.update(from: logitsPointer.advanced(by: start), count: classes)
                    }

                    var bestIndex = 0
                    var bestValue = -Float.greatestFiniteMagnitude
                    for cls in 0..<classes {
                        let value = frameLogits[cls]
                        if value > bestValue {
                            bestValue = value
                            bestIndex = cls
                        }
                    }

                    let logSumExp = VDSPOperations.logSumExp(frameLogits)
                    var shift = -logSumExp
                    vDSP_vsadd(
                        frameLogits,
                        1,
                        &shift,
                        &logProbabilityBuffer,
                        1,
                        vDSP_Length(classes)
                    )

                    probabilityBuffer = logProbabilityBuffer
                    probabilityBuffer.withUnsafeMutableBufferPointer { pointer in
                        var count = Int32(classes)
                        vvexpf(pointer.baseAddress!, pointer.baseAddress!, &count)
                    }

                    chunkLogProbs[frameIndex].withUnsafeMutableBufferPointer { destination in
                        logProbabilityBuffer.withUnsafeBufferPointer { source in
                            destination.baseAddress!.update(from: source.baseAddress!, count: classes)
                        }
                    }

                    for cls in 0..<min(classes, classProbabilitySums.count) {
                        classProbabilitySums[cls] += probabilityBuffer[cls]
                    }

                    if bestIndex < classHistogram.count {
                        classHistogram[bestIndex] += 1
                    }

                    let winningClass = min(bestIndex, powerset.count - 1)
                    let winningSpeakers = powerset[winningClass].filter { $0 < speakerCount }
                    let winningProbability = probabilityBuffer[winningClass]
                    let emptyProbability =
                        emptyClassIndex < probabilityBuffer.count ? probabilityBuffer[emptyClassIndex] : 0

                    if !winningSpeakers.isEmpty {
                        winningProbabilitySum += Double(winningProbability)
                        winningProbabilityCount += 1
                        if winningProbability < winningProbabilityMin {
                            winningProbabilityMin = winningProbability
                        }
                        if winningProbability > winningProbabilityMax {
                            winningProbabilityMax = winningProbability
                        }
                        emptyClassProbabilitySum += Double(emptyProbability)
                        emptyClassProbabilityCount += 1

                        for (index, threshold) in probabilityThresholds.enumerated() {
                            if winningProbability >= threshold {
                                probabilityThresholdCounts[index] += 1
                            }
                        }
                    }

                    // Vectorized speaker activation using matrix-vector multiply
                    // speakerActivations[speaker] = sum of probabilityBuffer[class] where speaker in powerset[class]
                    // Handle case where model outputs fewer classes than powerset entries (e.g., 7 vs 8)
                    let paddedProbabilityBuffer: [Float]
                    if probabilityBuffer.count < powerset.count {
                        paddedProbabilityBuffer =
                            probabilityBuffer
                            + [Float](
                                repeating: 0,
                                count: powerset.count - probabilityBuffer.count
                            )
                    } else {
                        paddedProbabilityBuffer = Array(probabilityBuffer.prefix(powerset.count))
                    }

                    let speakerActivations = VDSPOperations.matrixVectorMultiply(
                        matrix: speakerToClassMapping,
                        vector: paddedProbabilityBuffer
                    ).map { min(max($0, 0), 1) }
                    chunkSpeakerProbs[frameIndex] = speakerActivations

                    let speechProbability = max(0, min(1, 1 - emptyProbability))
                    if speechProbability >= onsetThreshold {
                        speechFrameCount += 1
                    }
                }

                var chunkWeights = Array(
                    repeating: Array(repeating: Float.zero, count: speakerCount),
                    count: frames
                )

                // Pyannote community-1 powerset models provide powerset probabilities that we marginalize
                // into per-speaker activity weights for each frame (0...1).
                for frameIndex in 0..<frames {
                    chunkWeights[frameIndex] = chunkSpeakerProbs[frameIndex]
                }

                logProbChunks.append(chunkLogProbs)
                weightChunks.append(chunkWeights)

                if chunkEmissionEnabled, let chunkCallback {
                    let chunkOffsetSeconds = chunkOffsets.last ?? Double(offset) / Double(config.sampleRate)
                    let chunk = SegmentationChunk(
                        chunkIndex: globalChunkIndex,
                        chunkOffsetSeconds: chunkOffsetSeconds,
                        frameDuration: frameDuration,
                        logProbs: chunkLogProbs,
                        speakerWeights: chunkWeights
                    )
                    if chunkCallback(chunk) == .stop {
                        chunkEmissionEnabled = false
                    }
                }

                if globalChunkIndex == 0 {
                    let speakerCoverage = chunkSpeakerProbs.reduce(into: Array(repeating: 0, count: speakerCount)) {
                        counts, frame in
                        for (index, probability) in frame.enumerated() where probability >= onsetThreshold {
                            counts[index] += 1
                        }
                    }
                    logger.debug("Chunk 0 speaker frame counts: \(speakerCoverage)")
                }

                globalChunkIndex += 1
            }
        }

        guard processedAnyBatch else {
            throw OfflineDiarizationError.processingFailed("Segmentation produced no analysis windows")
        }

        let totalFrames = classHistogram.reduce(0, +)
        if totalFrames > 0 {
            let speechFrames = totalFrames - classHistogram[0]
            let speechRatio = Double(speechFrames) / Double(totalFrames)
            let nonSpeechProb =
                classProbabilitySums[0] / Float(totalFrames == 0 ? 1 : totalFrames)
            logger.debug(
                """
                Segmentation histogram: speechFrames=\(speechFrames) totalFrames=\(totalFrames) \
                speechRatio=\(String(format: "%.3f", speechRatio)) avgNonSpeechProb=\(String(format: "%.3f", nonSpeechProb))
                """
            )
        }

        let totalFramesWithSpeech = speechFrameCount
        let totalFramesOverall = numFrames * logProbChunks.count
        if totalFramesOverall > 0 {
            let ratio = Double(totalFramesWithSpeech) / Double(totalFramesOverall)
            let ratioString = String(format: "%.3f", ratio)
            let predictedDuration = Double(totalFramesWithSpeech) * frameDuration
            let durationString = String(format: "%.1f", predictedDuration)
            logger.debug(
                "Segmentation mask speech frames = \(totalFramesWithSpeech) / \(totalFramesOverall) (ratio=\(ratioString), speechSeconds≈\(durationString)s)"
            )
        }

        if winningProbabilityCount > 0 {
            let averageWinning = winningProbabilitySum / Double(winningProbabilityCount)
            logger.debug(
                """
                Winning speaker probability stats: count=\(winningProbabilityCount), \
                avg=\(String(format: "%.3f", averageWinning)), \
                min=\(String(format: "%.3f", winningProbabilityMin)), \
                max=\(String(format: "%.3f", winningProbabilityMax))
                """
            )

            var distribution: [String] = []
            for (index, threshold) in probabilityThresholds.enumerated() {
                let count = probabilityThresholdCounts[index]
                let thresholdString = String(format: "%.3f", threshold)
                distribution.append("≥\(thresholdString):\(count)")
            }
            let distributionString = distribution.joined(separator: ", ")
            logger.debug("Winning probability distribution \(distributionString)")
        }

        if emptyClassProbabilityCount > 0 {
            let averageEmpty = emptyClassProbabilitySum / Double(emptyClassProbabilityCount)
            let averageEmptyString = String(format: "%.3f", averageEmpty)
            logger.debug(
                "Empty-class probability on speech frames: avg=\(averageEmptyString)"
            )
        }

        if preparedWindowCount > 0 {
            let prepareMs = Self.milliseconds(from: prepareDuration)
            let predictionMs = Self.milliseconds(from: predictionDuration)
            let preparePerWindow = prepareMs / Double(preparedWindowCount)
            let predictionPerWindow = predictionMs / Double(preparedWindowCount)
            let prepareTotalString = String(format: "%.2f", prepareMs)
            let prepareWindowString = String(format: "%.4f", preparePerWindow)
            let predictionTotalString = String(format: "%.2f", predictionMs)
            let predictionWindowString = String(format: "%.4f", predictionPerWindow)
            let message =
                """
                Segmentation timings: windows=\(preparedWindowCount) \
                prepareTotal=\(prepareTotalString)ms (perWindow=\(prepareWindowString)ms) \
                predictionTotal=\(predictionTotalString)ms (perWindow=\(predictionWindowString)ms)
                """
            logger.debug(message)
            Self.emitProfileLog(message)
        }

        return SegmentationOutput(
            logProbs: logProbChunks,
            speakerWeights: weightChunks,
            numChunks: logProbChunks.count,
            numFrames: numFrames,
            numSpeakers: speakerCount,
            chunkOffsets: chunkOffsets,
            frameDuration: frameDuration
        )
    }

}

extension OfflineSegmentationProcessor {
    fileprivate static func milliseconds(from duration: Duration) -> Double {
        let components = duration.components
        let secondsMs = Double(components.seconds) * 1_000
        let attosecondsMs = Double(components.attoseconds) / 1_000_000_000_000_000.0
        return secondsMs + attosecondsMs
    }

    fileprivate static func emitProfileLog(_ message: String) {
        let line = "[Profiling] \(message)\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}
