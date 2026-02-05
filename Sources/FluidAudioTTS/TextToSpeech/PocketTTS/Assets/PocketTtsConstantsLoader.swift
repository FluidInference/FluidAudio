import FluidAudio
import Foundation
import OSLog

/// Pre-loaded binary constants for PocketTTS inference.
public struct PocketTtsConstantsBundle: Sendable {
    public let bosEmbedding: [Float]
    public let textEmbedTable: [Float]
    public let tokenizer: SentencePieceTokenizer
    /// Latent normalization mean [32].
    public let embMean: [Float]
    /// Latent normalization std [32].
    public let embStd: [Float]
    /// Quantizer projection weight [512, 32] (flattened row-major).
    public let quantizerWeight: [Float]
}

/// Pre-loaded voice conditioning data.
public struct PocketTtsVoiceData: Sendable {
    /// Flattened audio prompt: [1, promptLength, 1024]
    public let audioPrompt: [Float]
    /// Number of voice conditioning tokens (typically 125).
    public let promptLength: Int
}

/// Loads PocketTTS constants from raw `.bin` Float32 files on disk.
public enum PocketTtsConstantsLoader {

    private static let logger = AppLogger(category: "PocketTtsConstantsLoader")

    public enum LoadError: Error {
        case fileNotFound(String)
        case invalidSize(String, expected: Int, actual: Int)
        case tokenizerLoadFailed(String)
    }

    /// Load all constants from the given directory.
    public static func load(from directory: URL) throws -> PocketTtsConstantsBundle {
        let constantsDir = directory.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)

        let bosEmb = try loadFloatArray(
            from: constantsDir.appendingPathComponent("bos_emb.bin"),
            expectedCount: PocketTtsConstants.latentDim,
            name: "bos_emb"
        )
        let embedTable = try loadFloatArray(
            from: constantsDir.appendingPathComponent("text_embed_table.bin"),
            expectedCount: PocketTtsConstants.vocabSize * PocketTtsConstants.embeddingDim,
            name: "text_embed_table"
        )

        let embMean = try loadFloatArray(
            from: constantsDir.appendingPathComponent("emb_mean.bin"),
            expectedCount: PocketTtsConstants.latentDim,
            name: "emb_mean"
        )
        let embStd = try loadFloatArray(
            from: constantsDir.appendingPathComponent("emb_std.bin"),
            expectedCount: PocketTtsConstants.latentDim,
            name: "emb_std"
        )
        let quantizerWeight = try loadFloatArray(
            from: constantsDir.appendingPathComponent("quantizer_weight.bin"),
            expectedCount: PocketTtsConstants.quantizerDim * PocketTtsConstants.latentDim,
            name: "quantizer_weight"
        )

        let tokenizerURL = constantsDir.appendingPathComponent("tokenizer.model")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw LoadError.fileNotFound("tokenizer.model")
        }
        let tokenizerData = try Data(contentsOf: tokenizerURL)
        let tokenizer: SentencePieceTokenizer
        do {
            tokenizer = try SentencePieceTokenizer(modelData: tokenizerData)
        } catch {
            throw LoadError.tokenizerLoadFailed(error.localizedDescription)
        }

        logger.info("Loaded PocketTTS constants from \(directory.lastPathComponent)")

        return PocketTtsConstantsBundle(
            bosEmbedding: bosEmb,
            textEmbedTable: embedTable,
            tokenizer: tokenizer,
            embMean: embMean,
            embStd: embStd,
            quantizerWeight: quantizerWeight
        )
    }

    /// Load voice conditioning data from the given directory.
    ///
    /// HuggingFace layout: `constants_bin/<voice>_audio_prompt.bin`
    public static func loadVoice(
        _ voice: String, from directory: URL
    ) throws -> PocketTtsVoiceData {
        // Sanitize voice name to prevent path traversal
        let sanitized = voice.components(separatedBy: CharacterSet.alphanumerics.inverted).joined()
        guard !sanitized.isEmpty else {
            throw LoadError.fileNotFound("invalid voice name: \(voice)")
        }

        let constantsDir = directory.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)

        let audioPrompt = try loadFloatArray(
            from: constantsDir.appendingPathComponent("\(sanitized)_audio_prompt.bin"),
            expectedCount: PocketTtsConstants.voicePromptLength * PocketTtsConstants.embeddingDim,
            name: "\(sanitized)_audio_prompt"
        )

        logger.info("Loaded PocketTTS voice '\(sanitized)' conditioning data")

        return PocketTtsVoiceData(
            audioPrompt: audioPrompt,
            promptLength: PocketTtsConstants.voicePromptLength
        )
    }

    // MARK: - Private

    /// Load a raw Float32 binary file into a [Float] array.
    private static func loadFloatArray(
        from url: URL, expectedCount: Int, name: String
    ) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadError.fileNotFound(name)
        }

        let data = try Data(contentsOf: url)
        let actualCount = data.count / MemoryLayout<Float>.size

        guard actualCount == expectedCount else {
            throw LoadError.invalidSize(name, expected: expectedCount, actual: actualCount)
        }

        return data.withUnsafeBytes { rawBuffer in
            let floatBuffer = rawBuffer.bindMemory(to: Float.self)
            return Array(floatBuffer)
        }
    }
}
