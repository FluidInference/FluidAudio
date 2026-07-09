import Accelerate
import CoreML
import Foundation

/// TitaNet-10s masked embedding extractor — selectable alternative to the
/// WeSpeaker `OfflineEmbeddingExtractor` (`config.embedding.backend == .titanet10s`).
///
/// Same inputs (audio source + segmentation chunk stream), same output
/// (`[TimedEmbedding]`), different mechanics: the TitaNet front + encoder run
/// ONCE per 10 s window (audio → encoder frames), and the mask-input decoder
/// runs once per active speaker with that speaker's overlap-excluded mask
/// (segmentation weights resampled to the 1000 mel frames of the window).
///
/// The model ships as THREE CoreML stages (front | encoder | maskdec). The
/// front and encoder must not be fused into one graph: a fused fp16 graph
/// collapses (chain parity 0.099 vs the reference); the package boundary
/// forces fp32 I/O between them and every stage is healthy at fp16.
///
/// `embedding256` carries a 192-d vector (field name is historical);
/// `rho128` is empty — PLDA/VBx are undefined for this space, which is why
/// the config validator requires `clustering.algorithm == .nmesc`.
@available(macOS 14.0, iOS 17.0, *)
struct TitaNetMaskedExtractor {
    private let frontModel: MLModel
    private let encoderModel: MLModel
    private let maskdecModel: MLModel
    private let config: OfflineDiarizerConfig
    private let logger = AppLogger(category: "TitaNetExtractor")

    static let embeddingDimension = 192
    private static let melFrames = 1000
    private static let windowSamples = 160_000
    private static let overlapThreshold: Float = 1e-3
    /// Mirrors OfflineEmbeddingExtractor's minActiveRatio guard.
    private static let minActiveRatio: Float = 0.2

    private static let frontName = "TitaNet10s_front_fp16"
    private static let encoderName = "TitaNet10s_encoder_fp16"
    private static let maskdecName = "TitaNet10s_maskdec_fp16"

    init(
        directory: URL,
        config: OfflineDiarizerConfig,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws {
        let mlConfiguration = MLModelConfiguration()
        mlConfiguration.computeUnits = computeUnits
        frontModel = try await Self.loadModel(
            named: Self.frontName, in: directory, configuration: mlConfiguration)
        encoderModel = try await Self.loadModel(
            named: Self.encoderName, in: directory, configuration: mlConfiguration)
        maskdecModel = try await Self.loadModel(
            named: Self.maskdecName, in: directory, configuration: mlConfiguration)
        self.config = config
        logger.info("TitaNet 10s masked extractor loaded from \(directory.path)")
    }

    /// Loads `<name>.mlmodelc` if present, otherwise compiles `<name>.mlpackage`.
    private static func loadModel(
        named name: String, in directory: URL, configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let compiled = directory.appendingPathComponent("\(name).mlmodelc")
        if FileManager.default.fileExists(atPath: compiled.path) {
            return try MLModel(contentsOf: compiled, configuration: configuration)
        }
        let package = directory.appendingPathComponent("\(name).mlpackage")
        guard FileManager.default.fileExists(atPath: package.path) else {
            throw OfflineDiarizationError.processingFailed(
                "TitaNet model \(name) not found in \(directory.path)")
        }
        let compiledURL = try await MLModel.compileModel(at: package)
        return try MLModel(contentsOf: compiledURL, configuration: configuration)
    }

    // MARK: - Entry points (mirror OfflineEmbeddingExtractor)

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
        audioSource: AudioSampleSource,
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

                let chunk = SegmentationChunk(
                    chunkIndex: chunkIndex,
                    chunkOffsetSeconds: chunkOffsetSeconds,
                    frameDuration: segmentation.frameDuration,
                    logProbs: [],
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
        audioSource: AudioSampleSource,
        segmentationStream: S
    ) async throws -> [TimedEmbedding] where S.Element == SegmentationChunk {
        var embeddings: [TimedEmbedding] = []
        let totalSamples = audioSource.sampleCount
        var emptyMaskCount = 0
        var fallbackMaskCount = 0
        var windowCount = 0

        for try await chunk in segmentationStream {
            try Task.checkCancellation()

            let weights = chunk.speakerWeights
            guard !weights.isEmpty, let speakerCount = weights.first?.count, speakerCount > 0
            else { continue }
            let frameCount = weights.count

            let frameDuration: Double
            if chunk.frameDuration > 0 {
                frameDuration = chunk.frameDuration
            } else {
                frameDuration = config.windowDuration / Double(max(frameCount, 1))
            }
            let chunkOffsetSeconds =
                chunk.chunkOffsetSeconds.isFinite
                ? chunk.chunkOffsetSeconds
                : Double(chunk.chunkIndex) * config.windowDuration
            let minFramesForEmbedding = max(1, Int(ceil(config.minSegmentDuration / frameDuration)))

            // Overlap frames: >1 speaker active (mirror of excludeOverlap).
            var overlapFrames = [Bool](repeating: false, count: frameCount)
            if config.embeddingExcludeOverlap {
                for (frame, row) in weights.enumerated() {
                    var active = 0
                    for value in row where value > Self.overlapThreshold {
                        active += 1
                        if active > 1 {
                            overlapFrames[frame] = true
                            break
                        }
                    }
                }
            }

            var encoderOutput: MLMultiArray?  // computed lazily, once per window

            for speakerIndex in 0..<speakerCount {
                var baseMask = [Float](repeating: 0, count: frameCount)
                for frame in 0..<frameCount { baseMask[frame] = weights[frame][speakerIndex] }
                let baseSum = VDSPOperations.sum(baseMask)
                if baseSum <= 0 { continue }

                var cleanMask = baseMask
                if config.embeddingExcludeOverlap {
                    for frame in 0..<frameCount where overlapFrames[frame] {
                        cleanMask[frame] = 0
                    }
                }
                let cleanSum = VDSPOperations.sum(cleanMask)
                if cleanSum < Float(frameCount) * Self.minActiveRatio {
                    emptyMaskCount += 1
                    continue
                }
                let maskToUse: [Float]
                if cleanSum >= Float(minFramesForEmbedding) {
                    maskToUse = cleanMask
                } else {
                    maskToUse = baseMask
                    fallbackMaskCount += 1
                }

                let resampledMask = WeightInterpolation.resample(maskToUse, to: Self.melFrames)
                // Hard guard: the maskdec clamps its pool denominator for fp16
                // safety, so a near-zero mask returns finite garbage, not NaN.
                if VDSPOperations.sum(resampledMask) <= 0 {
                    emptyMaskCount += 1
                    continue
                }

                if encoderOutput == nil {
                    encoderOutput = try encodeWindow(
                        audioSource: audioSource,
                        chunkOffsetSeconds: chunkOffsetSeconds,
                        totalSamples: totalSamples
                    )
                    windowCount += 1
                }
                guard let encoded = encoderOutput else { break }

                let embedding = try runMaskDecoder(encoded: encoded, mask: resampledMask)

                let firstActive = maskToUse.firstIndex(where: { $0 > Self.overlapThreshold }) ?? 0
                let lastActive =
                    maskToUse.lastIndex(where: { $0 > Self.overlapThreshold }) ?? firstActive
                embeddings.append(
                    TimedEmbedding(
                        chunkIndex: chunk.chunkIndex,
                        speakerIndex: speakerIndex,
                        startFrame: firstActive,
                        endFrame: lastActive,
                        frameWeights: maskToUse,
                        startTime: chunkOffsetSeconds + Double(firstActive) * frameDuration,
                        endTime: chunkOffsetSeconds + Double(lastActive + 1) * frameDuration,
                        embedding256: embedding,
                        rho128: []
                    ))
            }
        }

        logger.debug(
            "TitaNet masked extraction: \(embeddings.count) embeddings from \(windowCount) windows (fallbackMasks=\(fallbackMaskCount), emptyMasks=\(emptyMaskCount))"
        )
        return embeddings
    }

    // MARK: - Model stages

    /// audio window → front (mel features) → encoder frames. Runs once per window.
    private func encodeWindow(
        audioSource: AudioSampleSource,
        chunkOffsetSeconds: Double,
        totalSamples: Int
    ) throws -> MLMultiArray {
        let estimatedStartSample = Int((chunkOffsetSeconds * Double(config.sampleRate)).rounded())
        let startSample = max(0, min(estimatedStartSample, totalSamples))
        let available = max(0, min(Self.windowSamples, totalSamples - startSample))

        let audioArray = try MLMultiArray(
            shape: [1, NSNumber(value: Self.windowSamples)], dataType: .float32)
        let audioPointer = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
        vDSP_vclr(audioPointer, 1, vDSP_Length(Self.windowSamples))
        if available > 0 {
            try audioSource.copySamples(into: audioPointer, offset: startSample, count: available)
        }

        let options = MLPredictionOptions()
        audioArray.prefetchToNeuralEngine()
        let frontOutput = try frontModel.prediction(
            from: ZeroCopyDiarizerFeatureProvider(features: [
                "audio": MLFeatureValue(multiArray: audioArray)
            ]),
            options: options
        )
        guard let features = frontOutput.featureValue(for: "feats")?.multiArrayValue else {
            throw OfflineDiarizationError.processingFailed("TitaNet front missing feats output")
        }

        features.prefetchToNeuralEngine()
        let encoderResult = try encoderModel.prediction(
            from: ZeroCopyDiarizerFeatureProvider(features: [
                "feats": MLFeatureValue(multiArray: features)
            ]),
            options: options
        )
        guard let encoded = encoderResult.featureValue(for: "enc")?.multiArrayValue else {
            throw OfflineDiarizationError.processingFailed("TitaNet encoder missing enc output")
        }
        return encoded
    }

    /// encoder frames + per-speaker mask → 192-d embedding. Runs once per speaker.
    private func runMaskDecoder(encoded: MLMultiArray, mask: [Float]) throws -> [Float] {
        let maskArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: Self.melFrames)], dataType: .float32)
        let maskPointer = maskArray.dataPointer.assumingMemoryBound(to: Float.self)
        mask.withUnsafeBufferPointer { buffer in
            maskPointer.update(from: buffer.baseAddress!, count: Self.melFrames)
        }

        let options = MLPredictionOptions()
        encoded.prefetchToNeuralEngine()
        maskArray.prefetchToNeuralEngine()
        let output = try maskdecModel.prediction(
            from: ZeroCopyDiarizerFeatureProvider(features: [
                "enc": MLFeatureValue(multiArray: encoded),
                "mask": MLFeatureValue(multiArray: maskArray),
            ]),
            options: options
        )
        guard let embeddingArray = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw OfflineDiarizationError.processingFailed("TitaNet maskdec missing embedding output")
        }
        let pointer = embeddingArray.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: embeddingArray.count))
    }
}
