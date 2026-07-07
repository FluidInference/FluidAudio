@preconcurrency import CoreML
import Foundation
import OSLog

// MARK: - Configuration

/// Configuration for the offline (whole-window) Sortformer diarizer.
///
/// The offline model is a single fused graph `mel -> speaker_preds` over a fixed window
/// (3072 mel frames -> 384 output frames, 30.72s) with no streaming state. Long audio is tiled
/// into overlapping windows and the per-window speaker columns are stitched back together by
/// ``SortformerSpeakerStitcher``.
public struct OfflineSortformerConfig: Sendable {
    /// Head-weight precision (selects the HuggingFace subdirectory).
    public var precision: ModelNames.Sortformer.ModelPrecision

    /// Output (post-subsampling) frames per window. Fixed by the exported model.
    public let windowOutputFrames: Int = 384

    /// Encoder subsampling factor (mel frames per output frame).
    public let subsamplingFactor: Int = 8

    /// Number of speaker slots (fixed at 4 for current models).
    public let numSpeakers: Int = 4

    /// Mel filterbank feature count.
    public let melFeatures: Int = 128

    /// Model sample rate in Hz.
    public let sampleRate: Int = 16000

    /// Mel window / stride in samples (25ms / 10ms).
    public let melWindow: Int = 400
    public let melStride: Int = 160

    /// Duration of one output frame in seconds (subsampling * stride / sampleRate).
    public var frameDurationSeconds: Float {
        Float(subsamplingFactor) * Float(melStride) / Float(sampleRate)
    }

    /// Output-frame overlap between consecutive windows used to stitch speaker identities.
    /// ~8s of context — enough to correlate speaker activity, well under one window.
    public var overlapOutputFrames: Int = 100

    /// Mel frames per window (`windowOutputFrames * subsamplingFactor`).
    public var windowMelFrames: Int { windowOutputFrames * subsamplingFactor }

    public init(
        precision: ModelNames.Sortformer.ModelPrecision = .fp16,
        overlapOutputFrames: Int = 100
    ) {
        self.precision = precision
        self.overlapOutputFrames = overlapOutputFrames
    }

    /// Default offline configuration (fp16, ~8s overlap).
    public static let offlineV2_1 = OfflineSortformerConfig()
}

// MARK: - Model Container

/// Loads and runs the fused offline Sortformer model (`mel -> speaker_preds`).
public struct OfflineSortformerModels {
    public let mainModel: MLModel
    public let compilationDuration: TimeInterval

    private let config: OfflineSortformerConfig
    private let memoryOptimizer: ANEMemoryOptimizer
    private let melArray: MLMultiArray
    private let melLengthArray: MLMultiArray

    public init(
        config: OfflineSortformerConfig,
        main: MLModel,
        compilationDuration: TimeInterval = 0
    ) throws {
        self.config = config
        self.mainModel = main
        self.compilationDuration = compilationDuration
        self.memoryOptimizer = .init()
        // Model input is channels-first [1, melFeatures, windowMelFrames].
        self.melArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: config.windowMelFrames)],
            dataType: .float32)
        self.melLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
    }

    /// Run the offline model on one window.
    ///
    /// - Parameters:
    ///   - melTimeMajor: Mel features for the window, flat `[validMelFrames * melFeatures]` in
    ///     time-major order (frame `t`, feature `c` at `t * melFeatures + c`) as produced by
    ///     ``AudioMelSpectrogram/computeFlatTransposed(audio:)``.
    ///   - validMelFrames: Number of valid mel frames (≤ window). Shorter windows are zero-padded
    ///     and masked via `mel_length`.
    /// - Returns: Per-frame speaker probabilities, flat `[windowOutputFrames * numSpeakers]`
    ///   (frame-major).
    public func runOffline(melTimeMajor: [Float], validMelFrames: Int) throws -> [Float] {
        let featureCount = config.melFeatures
        let windowFrames = config.windowMelFrames
        let frames = min(validMelFrames, windowFrames)

        let dst = melArray.dataPointer.assumingMemoryBound(to: Float.self)
        // Transpose time-major [t, c] -> channels-first [c, t]; zero-pad the tail.
        for t in 0..<frames {
            let srcBase = t * featureCount
            for c in 0..<featureCount {
                dst[c * windowFrames + t] = melTimeMajor[srcBase + c]
            }
        }
        if frames < windowFrames {
            for c in 0..<featureCount {
                let rowBase = c * windowFrames
                for t in frames..<windowFrames {
                    dst[rowBase + t] = 0
                }
            }
        }
        melLengthArray[0] = NSNumber(value: Int32(frames))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
            "mel_length": MLFeatureValue(multiArray: melLengthArray),
        ])
        let output = try mainModel.prediction(from: input)

        guard
            let preds = output.featureValue(for: "speaker_preds")?
                .shapedArrayValue(of: Float32.self)?.scalars
        else {
            throw SortformerError.inferenceFailed("Missing speaker_preds")
        }
        return preds
    }
}

// MARK: - Model Loading

extension OfflineSortformerModels {

    private static let logger = AppLogger(category: "OfflineSortformerModels")

    /// Load from a local compiled/`.mlpackage` model.
    ///
    /// - Parameter computeUnits: CoreML compute units to use (default `.all`). On RAM-constrained
    ///   or pre-M1 devices (e.g. A14) where `.all` is slow or incompatible, pass `.cpuOnly`.
    public static func load(
        config: OfflineSortformerConfig = .offlineV2_1,
        modelPath: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> OfflineSortformerModels {
        let start = Date()
        let compiledURL: URL
        if modelPath.pathExtension == "mlmodelc" {
            compiledURL = modelPath
        } else {
            compiledURL = try await MLModel.compileModel(at: modelPath)
        }
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        let model = try MLModel(contentsOf: compiledURL, configuration: mlConfig)
        let duration = Date().timeIntervalSince(start)
        logger.info(
            "Loaded offline Sortformer model in \(String(format: "%.2f", duration))s "
                + "(computeUnits=\(mlConfig.computeUnits.rawValue))")
        return try OfflineSortformerModels(config: config, main: model, compilationDuration: duration)
    }

    /// Download (if needed) and load the offline model from HuggingFace.
    public static func loadFromHuggingFace(
        config: OfflineSortformerConfig = .offlineV2_1,
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ProgressHandler? = nil
    ) async throws -> OfflineSortformerModels {
        let start = Date()

        let directory: URL
        if let cache = cacheDirectory {
            directory = cache
        } else {
            directory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("FluidAudio/Models")
        }

        let bundle = ModelNames.Sortformer.offlineBundle(precision: config.precision)
        logger.info("Downloading offline Sortformer model: \(bundle)...")

        let models = try await ModelHub.loadModels(
            .sortformer,
            modelNames: [bundle],
            directory: directory,
            computeUnits: computeUnits,
            variant: bundle,
            progressHandler: progressHandler
        )

        guard let model = models[bundle] else {
            throw SortformerError.modelLoadFailed("Failed to load offline Sortformer model from HuggingFace")
        }

        let duration = Date().timeIntervalSince(start)
        logger.info("Offline Sortformer model loaded from HuggingFace in \(String(format: "%.2f", duration))s")
        return try OfflineSortformerModels(config: config, main: model, compilationDuration: duration)
    }
}

// MARK: - Diarizer

/// Whole-file (offline) speaker diarizer built on the fused Sortformer graph.
///
/// Faster than the streaming path when the entire audio is available up front: each 30.72s window
/// is one CoreML call (no per-chunk state threading). Windows overlap and are stitched into a
/// single, globally speaker-consistent ``DiarizerTimeline`` via ``SortformerSpeakerStitcher``.
public final class OfflineSortformerDiarizer {

    private let logger = AppLogger(category: "OfflineSortformerDiarizer")
    public let config: OfflineSortformerConfig
    private let timelineConfig: DiarizerTimelineConfig
    private let melSpectrogram: AudioMelSpectrogram
    private let lock = NSLock()
    private var _models: OfflineSortformerModels?

    public var isAvailable: Bool {
        withLock { _models != nil }
    }

    /// Execute a closure while holding the lock (async-safe: the lock ops stay in a sync frame).
    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    public init(
        config: OfflineSortformerConfig = .offlineV2_1,
        timelineConfig: DiarizerTimelineConfig? = nil
    ) {
        self.config = config
        self.timelineConfig =
            timelineConfig
            ?? DiarizerTimelineConfig.default(
                numSpeakers: config.numSpeakers,
                frameDurationSeconds: config.frameDurationSeconds)
        self.melSpectrogram = AudioMelSpectrogram()
    }

    /// Initialize from a local model path.
    ///
    /// - Parameter computeUnits: CoreML compute units to use (default `.all`). Pass `.cpuOnly`
    ///   on RAM-constrained or pre-M1 devices (e.g. A14) where `.all` is slow or incompatible.
    public func initialize(modelPath: URL, computeUnits: MLComputeUnits = .all) async throws {
        let models = try await OfflineSortformerModels.load(
            config: config, modelPath: modelPath, computeUnits: computeUnits)
        withLock { _models = models }
    }

    /// Initialize from HuggingFace.
    public func initializeFromHuggingFace(
        computeUnits: MLComputeUnits = .all,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        let models = try await OfflineSortformerModels.loadFromHuggingFace(
            config: config, computeUnits: computeUnits, progressHandler: progressHandler)
        withLock { _models = models }
    }

    /// Initialize with pre-loaded models.
    public func initialize(models: OfflineSortformerModels) {
        withLock { _models = models }
    }

    /// Diarize a complete audio buffer.
    ///
    /// - Parameters:
    ///   - samples: Mono audio buffer.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    /// - Returns: A finalized timeline with globally consistent speaker IDs.
    public func processComplete(
        _ samples: [Float],
        sourceSampleRate: Double? = nil
    ) throws -> DiarizerTimeline {
        lock.lock()
        defer { lock.unlock() }
        guard let models = _models else { throw SortformerError.notInitialized }

        let normalized = try normalizeSamples(samples, sourceSampleRate: sourceSampleRate)
        guard !normalized.isEmpty else {
            return DiarizerTimeline(config: timelineConfig)
        }

        let (melFlat, _, numMelFrames) = melSpectrogram.computeFlatTransposed(audio: normalized)
        let featureCount = config.melFeatures
        let windowMel = config.windowMelFrames
        let outPerWindow = config.windowOutputFrames
        let speakers = config.numSpeakers
        let sub = config.subsamplingFactor

        guard numMelFrames > 0 else {
            return DiarizerTimeline(config: timelineConfig)
        }

        let overlapOut = max(0, min(config.overlapOutputFrames, outPerWindow - 1))
        let hopOut = outPerWindow - overlapOut
        let hopMel = hopOut * sub

        let totalOut = (numMelFrames + sub - 1) / sub  // ceil to cover the tail
        var global = [Float](repeating: 0, count: totalOut * speakers)
        var filled = [Bool](repeating: false, count: totalOut)

        var melStart = 0
        var windowIndex = 0
        while melStart < numMelFrames {
            let validMel = min(windowMel, numMelFrames - melStart)
            let slice = Array(melFlat[(melStart * featureCount)..<((melStart + validMel) * featureCount)])
            let preds = try models.runOffline(melTimeMajor: slice, validMelFrames: validMel)

            let validOut = min(outPerWindow, (validMel + sub - 1) / sub)
            let gStart = melStart / sub

            // Align this window's speaker columns to the global timeline over the overlap region.
            var mapping = Array(0..<speakers)
            if windowIndex > 0, overlapOut > 0 {
                let ov = min(overlapOut, validOut, max(0, totalOut - gStart))
                if ov > 0 {
                    var gOverlap = [Float](repeating: 0, count: ov * speakers)
                    var wOverlap = [Float](repeating: 0, count: ov * speakers)
                    for j in 0..<ov {
                        let gBase = (gStart + j) * speakers
                        let wBase = j * speakers
                        for s in 0..<speakers {
                            gOverlap[wBase + s] = global[gBase + s]
                            wOverlap[wBase + s] = preds[wBase + s]
                        }
                    }
                    mapping = SortformerSpeakerStitcher.alignment(
                        global: gOverlap, window: wOverlap, frames: ov, numSpeakers: speakers)
                }
            }

            // Write predictions in global speaker IDs; average where windows overlap.
            for j in 0..<validOut {
                let gf = gStart + j
                guard gf < totalOut else { break }
                let outBase = gf * speakers
                let inBase = j * speakers
                if filled[gf] {
                    for w in 0..<speakers {
                        let idx = outBase + mapping[w]
                        global[idx] = (global[idx] + preds[inBase + w]) * 0.5
                    }
                } else {
                    for w in 0..<speakers {
                        global[outBase + mapping[w]] = preds[inBase + w]
                    }
                    filled[gf] = true
                }
            }

            windowIndex += 1
            if validMel < windowMel { break }  // consumed the tail
            melStart += hopMel
        }

        logger.info("Offline diarization: \(windowIndex) window(s), \(totalOut) frames")

        let timeline = DiarizerTimeline(config: timelineConfig)
        _ = try timeline.rebuild(
            finalizedPredictions: global,
            tentativePredictions: [],
            keepingSpeakers: false,
            isComplete: true
        )
        return timeline
    }

    /// Diarize a complete audio file (read + resampled to the model rate).
    public func processComplete(audioFileURL: URL) throws -> DiarizerTimeline {
        let converter = AudioConverter(sampleRate: Double(config.sampleRate))
        let audio = try converter.resampleAudioFile(audioFileURL)
        return try processComplete(audio, sourceSampleRate: nil)
    }

    private func normalizeSamples(_ samples: [Float], sourceSampleRate: Double?) throws -> [Float] {
        guard let sourceSampleRate, sourceSampleRate != Double(config.sampleRate) else {
            return samples
        }
        return try AudioConverter(sampleRate: Double(config.sampleRate))
            .resample(samples, from: sourceSampleRate)
    }
}
