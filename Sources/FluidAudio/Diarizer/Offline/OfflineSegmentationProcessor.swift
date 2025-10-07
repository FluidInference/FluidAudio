import Accelerate
import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
struct OfflineSegmentationProcessor {
    private let logger = AppLogger(category: "OfflineSegmentation")
    private let memoryOptimizer = ANEMemoryOptimizer()

    private let powerset: [[Int]] = [
        [],
        [0],
        [1],
        [2],
        [0, 1],
        [0, 2],
        [1, 2],
    ]

    func process(
        audioSamples: [Float],
        segmentationModel: MLModel,
        config: OfflineDiarizerConfig
    ) throws -> SegmentationOutput {
        guard !audioSamples.isEmpty else {
            throw OfflineDiarizationError.noSpeechDetected
        }

        let chunkSize = config.samplesPerWindow
        let stepSize = config.samplesPerStep
        let totalSamples = audioSamples.count

        var logProbChunks: [[[Float]]] = []
        var weightChunks: [[[Float]]] = []
        var chunkOffsets: [Double] = []
        var frameDuration: Double = 0
        var numFrames = 0
        let speakerCount = 3
        var classHistogram = Array(repeating: 0, count: powerset.count)
        var classProbabilitySums = Array(repeating: Float.zero, count: powerset.count)

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

        let offsets = Array(stride(from: 0, to: totalSamples, by: stepSize))
        guard !offsets.isEmpty else {
            throw OfflineDiarizationError.processingFailed("Segmentation produced no analysis windows")
        }

        let batchCapacity = 32
        var globalChunkIndex = 0

        try audioSamples.withUnsafeBufferPointer { audioPointer in
            for batchStart in stride(from: 0, to: offsets.count, by: batchCapacity) {
                let batchCount = min(batchCapacity, offsets.count - batchStart)
                let shape: [NSNumber] = [
                    NSNumber(value: batchCount),
                    1,
                    NSNumber(value: chunkSize),
                ]
                let audioArray = try memoryOptimizer.createAlignedArray(
                    shape: shape,
                    dataType: .float32
                )

                let ptr = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
                vDSP_vclr(ptr, 1, vDSP_Length(batchCount * chunkSize))

                var batchOffsets: [Int] = []
                batchOffsets.reserveCapacity(batchCount)

                for localIndex in 0..<batchCount {
                    let offset = offsets[batchStart + localIndex]
                    batchOffsets.append(offset)

                    let chunkEnd = min(offset + chunkSize, totalSamples)
                    let chunkLength = chunkEnd - offset
                    guard chunkLength > 0 else {
                        continue
                    }

                    let destination = ptr.advanced(by: localIndex * chunkSize)
                    vDSP_mmov(
                        audioPointer.baseAddress!.advanced(by: offset),
                        destination,
                        vDSP_Length(chunkLength),
                        1,
                        vDSP_Length(chunkLength),
                        1
                    )
                }

                let provider = ZeroCopyDiarizerFeatureProvider(
                    features: ["audio": MLFeatureValue(multiArray: audioArray)]
                )

                let options = MLPredictionOptions()
                if #available(macOS 14.0, iOS 17.0, *) {
                    audioArray.prefetchToNeuralEngine()
                }

                let output = try segmentationModel.prediction(from: provider, options: options)

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

                    var chunkWeights = Array(
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

                        if globalChunkIndex == 0, frameIndex < 30 {
                            let formattedWinning = String(format: "%.3f", winningProbability)
                            let formattedEmpty = String(format: "%.3f", emptyProbability)
                            logger.debug(
                                "Frame \(frameIndex) winningClass=\(winningClass) prob=\(formattedWinning) empty=\(formattedEmpty)"
                            )
                        }

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

                        var frameWeights = [Float](repeating: 0, count: speakerCount)
                        for speaker in winningSpeakers where speaker < speakerCount {
                            frameWeights[speaker] = 1.0
                        }

                        chunkWeights[frameIndex] = frameWeights

                        if frameWeights.contains(where: { $0 > 0 }) {
                            speechFrameCount += 1
                        }
                    }

                    logProbChunks.append(chunkLogProbs)
                    weightChunks.append(chunkWeights)

                    if globalChunkIndex == 0 {
                        let speakerCoverage = chunkWeights.reduce(into: Array(repeating: 0, count: speakerCount)) {
                            counts, frame in
                            for (index, value) in frame.enumerated() where value > 0 {
                                counts[index] += 1
                            }
                        }
                        logger.debug("Chunk 0 speaker frame counts: \(speakerCoverage)")
                    }

                    globalChunkIndex += 1
                }
            }
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
