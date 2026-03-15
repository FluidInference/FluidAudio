import Foundation
import OSLog

/// Pre-loaded binary constants for PocketTTS inference.
public struct PocketTtsConstantsBundle: Sendable {
    public let bosEmbedding: [Float]
    public let textEmbedTable: [Float]
    public let tokenizer: SentencePieceTokenizer
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
            tokenizer: tokenizer
        )
    }

    /// Load voice conditioning data from the given directory.
    ///
    /// Supports variable-length voice prompts — the prompt length is derived
    /// from the file size (`floatCount / embeddingDim`).
    ///
    /// HuggingFace layout: `constants_bin/<voice>_audio_prompt.bin`
    public static func loadVoice(
        _ voice: String, from directory: URL
    ) throws -> PocketTtsVoiceData {
        // Sanitize voice name to prevent path traversal
        let sanitized = voice.filter { $0.isLetter || $0.isNumber || $0 == "_" }
        guard !sanitized.isEmpty else {
            throw LoadError.fileNotFound("invalid voice name: \(voice)")
        }

        let constantsDir = directory.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)
        let voiceURL = constantsDir.appendingPathComponent("\(sanitized)_audio_prompt.bin")

        guard FileManager.default.fileExists(atPath: voiceURL.path) else {
            throw LoadError.fileNotFound("\(sanitized)_audio_prompt")
        }

        let data = try Data(contentsOf: voiceURL)
        let embDim = PocketTtsConstants.embeddingDim
        let floatCount = data.count / MemoryLayout<Float>.size

        guard floatCount > 0, floatCount % embDim == 0 else {
            throw LoadError.invalidSize(
                "\(sanitized)_audio_prompt",
                expected: embDim,
                actual: floatCount
            )
        }

        let promptLength = floatCount / embDim
        guard promptLength <= PocketTtsVoiceCloner.maxVoiceFrames else {
            throw LoadError.invalidSize(
                "\(sanitized)_audio_prompt",
                expected: PocketTtsVoiceCloner.maxVoiceFrames,
                actual: promptLength
            )
        }
        let audioPrompt = data.withUnsafeBytes { rawBuffer in
            let floatBuffer = rawBuffer.bindMemory(to: Float.self)
            return Array(floatBuffer)
        }

        logger.info("Loaded PocketTTS voice '\(sanitized)' conditioning data")

        return PocketTtsVoiceData(audioPrompt: audioPrompt, promptLength: promptLength)
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
