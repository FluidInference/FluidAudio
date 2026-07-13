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

    // The front MUST stay fp32: at fp16 its power-spectrum/log-mel math
    // underflows on quiet far-field windows (embedding cosine vs fp32 drops
    // to 0.21). It is ~570K of DSP, so fp32 costs nothing; only the heavy
    // encoder benefits from fp16/ANE.
    private static let frontName = "TitaNet10s_front_fp32"
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

        // `.cleanWaveform` re-featurizes each clean crop on CPU (vDSP); build the
        // featurizer (with its FFT setup) once per extraction, only when needed.
        let waveCropFeaturizer: TitaNetWaveCropFeaturizer? =
            config.embedding.titanetPooling == .cleanWaveform ? TitaNetWaveCropFeaturizer() : nil

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

            var frontFeats: MLMultiArray?  // audio → mel, computed lazily once per window
            var maskedWindowEnc: MLMultiArray?  // full-window enc, .maskedFull only
            var windowAudio: [Float]?  // available window samples, .cleanWaveform only

            func ensureFrontFeats() throws -> MLMultiArray {
                if let feats = frontFeats { return feats }
                let feats = try runFront(
                    audioSource: audioSource,
                    chunkOffsetSeconds: chunkOffsetSeconds,
                    totalSamples: totalSamples)
                frontFeats = feats
                windowCount += 1
                return feats
            }

            func ensureWindowAudio() throws -> [Float] {
                if let audio = windowAudio { return audio }
                let audio = try readWindowSamples(
                    audioSource: audioSource,
                    chunkOffsetSeconds: chunkOffsetSeconds,
                    totalSamples: totalSamples)
                windowAudio = audio
                windowCount += 1
                return audio
            }

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

                let embedding: [Float]
                let metaMask: [Float]  // drives frameWeights metadata (+ timing fallback)
                // Explicit frame bounds for segment timing; only .cleanWaveform sets
                // it (its clean frames use a 0.5 threshold, not metaMask's soft one).
                var explicitBounds: (first: Int, last: Int)?

                switch config.embedding.titanetPooling {
                case .cleanWaveform:
                    // Reference-faithful path: crop the clean single-speaker
                    // WAVEFORM, re-featurize each crop on its own (clean-only
                    // normalization, no cross-talk in any STFT window), drop the
                    // edge frames, concat in the feature domain, encode once.
                    guard let featurizer = waveCropFeaturizer else {
                        emptyMaskCount += 1
                        continue
                    }
                    let sr = config.sampleRate
                    let audio = try ensureWindowAudio()
                    guard
                        let selection = CleanRegionSelector.selectCleanSampleRegions(
                            weights: weights,
                            speaker: speakerIndex,
                            availableSamples: audio.count,
                            regionMinSamples: Self.sampleCount(ms: config.embedding.regionMinMs, sampleRate: sr),
                            targetSamples: Self.sampleCount(ms: config.embedding.embedTargetMs, sampleRate: sr),
                            embedMinSamples: Self.sampleCount(ms: config.embedding.embedMinMs, sampleRate: sr),
                            threshold: 0.5)
                    else {
                        emptyMaskCount += 1
                        continue
                    }
                    var cols: [(feats: [Float], frames: Int)] = []
                    for region in selection.regions {
                        let crop = Array(audio[region.start..<region.end])
                        if let featured = featurizer.featurize(crop: crop) { cols.append(featured) }
                    }
                    let packedFlat = CleanRegionSelector.packConcatFeatures(
                        columns: cols, melBins: TitaNetWaveCropFeaturizer.melBins,
                        dstFrames: Self.melFrames)
                    guard packedFlat.usedFrames > 0 else {
                        emptyMaskCount += 1
                        continue
                    }
                    let packed = try makeFeats(flat: packedFlat.feats)
                    let enc = try runEncoder(feats: packed)
                    var contiguousMask = [Float](repeating: 0, count: Self.melFrames)
                    for p in 0..<min(packedFlat.usedFrames, Self.melFrames) { contiguousMask[p] = 1 }
                    embedding = try runMaskDecoder(encoded: enc, mask: contiguousMask)
                    metaMask = cleanMask
                    explicitBounds = (selection.firstFrame, selection.lastFrame)

                case .cleanRegion:
                    // Resample the OVERLAP-EXCLUDED mask onto the mel grid, pick the
                    // clean frames, and pack them contiguously so the encoder never
                    // convolves over overlap (spec §3.2). Encoder runs per speaker.
                    let cleanMel = WeightInterpolation.resample(cleanMask, to: Self.melFrames)
                    guard
                        let frames = CleanRegionSelector.selectCleanMelFrames(
                            cleanMaskMel: cleanMel,
                            collarLead: Self.frameCount(ms: config.embedding.collarLeadMs),
                            collarTrail: Self.frameCount(ms: config.embedding.collarTrailMs),
                            regionMin: Self.frameCount(ms: config.embedding.regionMinMs),
                            embedMin: Self.frameCount(ms: config.embedding.embedMinMs),
                            embedTarget: Self.frameCount(ms: config.embedding.embedTargetMs))
                    else {
                        emptyMaskCount += 1
                        continue
                    }
                    let feats = try ensureFrontFeats()
                    let packed = try packCleanFeats(feats: feats, frames: frames)
                    let enc = try runEncoder(feats: packed)
                    var contiguousMask = [Float](repeating: 0, count: Self.melFrames)
                    for p in 0..<min(frames.count, Self.melFrames) { contiguousMask[p] = 1 }
                    embedding = try runMaskDecoder(encoded: enc, mask: contiguousMask)
                    metaMask = cleanMask

                case .maskedFull:
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
                    let feats = try ensureFrontFeats()
                    if maskedWindowEnc == nil {
                        maskedWindowEnc = try runEncoder(feats: feats)
                    }
                    guard let enc = maskedWindowEnc else {
                        emptyMaskCount += 1
                        continue
                    }
                    embedding = try runMaskDecoder(encoded: enc, mask: resampledMask)
                    metaMask = maskToUse
                }

                let firstActive: Int
                let lastActive: Int
                if let bounds = explicitBounds {
                    firstActive = bounds.first
                    lastActive = bounds.last
                } else {
                    firstActive = metaMask.firstIndex(where: { $0 > Self.overlapThreshold }) ?? 0
                    lastActive =
                        metaMask.lastIndex(where: { $0 > Self.overlapThreshold }) ?? firstActive
                }
                embeddings.append(
                    TimedEmbedding(
                        chunkIndex: chunk.chunkIndex,
                        speakerIndex: speakerIndex,
                        startFrame: firstActive,
                        endFrame: lastActive,
                        frameWeights: metaMask,
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

    /// audio window → front (mel features `[1, mel, melFrames]`). Runs once per window.
    private func runFront(
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
        return features
    }

    /// mel features `[1, mel, melFrames]` → encoder frames `[1, C, melFrames]`.
    /// `.maskedFull` runs this once per window; `.cleanRegion` once per (window, speaker).
    private func runEncoder(feats: MLMultiArray) throws -> MLMultiArray {
        let options = MLPredictionOptions()
        feats.prefetchToNeuralEngine()
        let encoderResult = try encoderModel.prediction(
            from: ZeroCopyDiarizerFeatureProvider(features: [
                "feats": MLFeatureValue(multiArray: feats)
            ]),
            options: options
        )
        guard let encoded = encoderResult.featureValue(for: "enc")?.multiArrayValue else {
            throw OfflineDiarizationError.processingFailed("TitaNet encoder missing enc output")
        }
        return encoded
    }

    /// Packs the selected mel frames of `feats` (`[1, mel, melFrames]`) contiguously
    /// into a fresh zero-padded `[1, mel, melFrames]` buffer, so the encoder's
    /// convolutions only ever span clean, overlap-free frames (spec §3.2, §5
    /// `concat_domain = feature`).
    private func packCleanFeats(feats: MLMultiArray, frames: [Int]) throws -> MLMultiArray {
        let shape = feats.shape.map { $0.intValue }
        guard shape.count == 3, shape[2] == Self.melFrames else {
            throw OfflineDiarizationError.processingFailed(
                "TitaNet front feats shape \(shape) is not [1, mel, \(Self.melFrames)]")
        }
        let melBins = shape[1]
        let strides = feats.strides.map { $0.intValue }
        let binStride = strides[1]
        let frameStride = strides[2]

        // Read feats into a contiguous [melBins, melFrames] buffer (handles any
        // CoreML output striding once), then pack via the unit-tested core.
        let src = feats.dataPointer.assumingMemoryBound(to: Float.self)
        var flat = [Float](repeating: 0, count: melBins * Self.melFrames)
        for c in 0..<melBins {
            let srcRow = c * binStride
            let dstRow = c * Self.melFrames
            for t in 0..<Self.melFrames { flat[dstRow + t] = src[srcRow + t * frameStride] }
        }
        let packedFlat = CleanRegionSelector.packColumns(
            src: flat, melBins: melBins, srcFrames: Self.melFrames,
            dstFrames: Self.melFrames, frames: frames)

        let packed = try MLMultiArray(shape: feats.shape, dataType: .float32)
        let dst = packed.dataPointer.assumingMemoryBound(to: Float.self)
        packedFlat.withUnsafeBufferPointer { buffer in
            dst.update(from: buffer.baseAddress!, count: packedFlat.count)
        }
        return packed
    }

    /// Milliseconds → mel-frame count on the 100 fps grid (melFrames over the 10 s window).
    private static func frameCount(ms: Int) -> Int {
        let windowSeconds = Double(windowSamples) / 16_000.0
        return max(0, Int((Double(ms) * Double(melFrames) / (windowSeconds * 1000.0)).rounded()))
    }

    /// Milliseconds → sample count at the given sample rate (for `.cleanWaveform`
    /// region thresholds, which operate at sample resolution).
    private static func sampleCount(ms: Int, sampleRate: Int) -> Int {
        max(0, ms * sampleRate / 1000)
    }

    /// Reads the window's available samples (≤ windowSamples, NOT zero-padded) as a
    /// `[Float]` for `.cleanWaveform` cropping. Runs once per window.
    private func readWindowSamples(
        audioSource: AudioSampleSource,
        chunkOffsetSeconds: Double,
        totalSamples: Int
    ) throws -> [Float] {
        let estimatedStartSample = Int((chunkOffsetSeconds * Double(config.sampleRate)).rounded())
        let startSample = max(0, min(estimatedStartSample, totalSamples))
        let available = max(0, min(Self.windowSamples, totalSamples - startSample))
        guard available > 0 else { return [] }
        var buffer = [Float](repeating: 0, count: available)
        try buffer.withUnsafeMutableBufferPointer { pointer in
            try audioSource.copySamples(
                into: pointer.baseAddress!, offset: startSample, count: available)
        }
        return buffer
    }

    /// Wraps a mel-major `[melBins * melFrames]` buffer as an encoder `feats`
    /// `[1, melBins, melFrames]` MLMultiArray (`.cleanWaveform` path).
    private func makeFeats(flat: [Float]) throws -> MLMultiArray {
        let arr = try MLMultiArray(
            shape: [
                1, NSNumber(value: TitaNetWaveCropFeaturizer.melBins),
                NSNumber(value: Self.melFrames),
            ],
            dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        flat.withUnsafeBufferPointer { buffer in
            dst.update(from: buffer.baseAddress!, count: min(flat.count, arr.count))
        }
        return arr
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
