@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "ForcedAlignerManager")

/// Public API for Qwen3-ForcedAligner forced alignment.
///
/// Produces per-word timestamps by aligning audio against a known transcript.
/// Uses a 3-model CoreML pipeline (audio encoder + embedding + decoder with LM head)
/// in a single non-autoregressive prefill pass.
///
/// Usage:
/// ```swift
/// let manager = ForcedAlignerManager()
/// try await manager.downloadAndLoadModels()
/// let result = try await manager.align(audioSamples: samples, text: "hello world")
/// for word in result.alignments {
///     print("\(word.word): \(word.startMs) - \(word.endMs) ms")
/// }
/// ```
@available(macOS 14, iOS 17, *)
public actor ForcedAlignerManager {
    private var models: ForcedAlignerModels?
    private var tokenizer: ForcedAlignerTokenizer?

    public init() {}

    /// Load models from a local directory.
    ///
    /// The directory must contain the 3 CoreML models and tokenizer files
    /// (vocab.json, merges.txt).
    public func loadModels(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws {
        models = try await ForcedAlignerModels.load(
            from: directory,
            computeUnits: computeUnits
        )

        // Load tokenizer from the same directory or download
        let vocabURL = directory.appendingPathComponent("vocab.json")
        let mergesURL = directory.appendingPathComponent("merges.txt")

        if FileManager.default.fileExists(atPath: vocabURL.path)
            && FileManager.default.fileExists(atPath: mergesURL.path)
        {
            tokenizer = try ForcedAlignerTokenizer(vocabURL: vocabURL, mergesURL: mergesURL)
        } else {
            let (downloadedVocab, downloadedMerges) = try await ForcedAlignerTokenizer.download(
                to: directory)
            tokenizer = try ForcedAlignerTokenizer(
                vocabURL: downloadedVocab, mergesURL: downloadedMerges)
        }

        logger.info("ForcedAligner models and tokenizer loaded successfully")
    }

    /// Download models from HuggingFace and load them.
    public func downloadAndLoadModels(
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws {
        let cacheDir = try await ForcedAlignerModels.download()

        // Download tokenizer files to the same cache directory
        let (vocabURL, mergesURL) = try await ForcedAlignerTokenizer.download(to: cacheDir)
        tokenizer = try ForcedAlignerTokenizer(vocabURL: vocabURL, mergesURL: mergesURL)

        models = try await ForcedAlignerModels.load(
            from: cacheDir,
            computeUnits: computeUnits
        )
        logger.info("ForcedAligner models downloaded and loaded successfully")
    }

    /// Run forced alignment on audio samples with a known transcript.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono Float32 audio samples.
    ///   - text: Transcript text to align against the audio.
    /// - Returns: Per-word timestamp alignments with latency info.
    public func align(audioSamples: [Float], text: String) throws -> ForcedAlignmentResult {
        guard let models = models, let tokenizer = tokenizer else {
            throw ForcedAlignerError.modelsNotLoaded
        }

        let start = CFAbsoluteTimeGetCurrent()

        let inference = ForcedAlignerInference(models: models, tokenizer: tokenizer)
        let alignments = try inference.align(audioSamples: audioSamples, text: text)

        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        logger.info(
            "Alignment complete: \(alignments.count) words in \(String(format: "%.0f", elapsed))ms"
        )

        return ForcedAlignmentResult(alignments: alignments, latencyMs: elapsed)
    }
}
