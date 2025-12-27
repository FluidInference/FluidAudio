import Foundation
import MLX
import MLXNN

/// Chinese TTS synthesizer using Kokoro model with MLX
///
/// This integrates:
/// - ChineseG2P for text-to-phoneme conversion (Bopomofo)
/// - KokoroModel for phoneme-to-audio synthesis (MLX)
///
/// Usage:
/// ```swift
/// let synthesizer = KokoroChineseSynthesizer()
/// try await synthesizer.loadModel(from: modelURL, config: configURL)
/// try await synthesizer.loadG2P(jiebaURL: jiebaURL, pinyinURL: pinyinURL)
/// let audio = try synthesizer.synthesize("你好世界", voice: voiceEmbedding)
/// ```
public final class KokoroChineseSynthesizer {

    // MARK: - Properties

    private var model: KokoroModel?
    private let g2p = ChineseG2P.shared
    private var voiceEmbeddings: [String: MLXArray] = [:]

    public var isModelLoaded: Bool { model != nil }
    public var isG2PLoaded: Bool { g2p.vocabulary.count > 0 }

    // MARK: - Initialization

    public init() {}

    // MARK: - Model Loading

    /// Load the Kokoro model from safetensors file
    public func loadModel(weightsURL: URL, configURL: URL) throws {
        let config = try KokoroModel.loadConfig(from: configURL)
        model = try KokoroModel.load(from: weightsURL, config: config)
    }

    /// Load the Kokoro model with pre-loaded config
    public func loadModel(weightsURL: URL, config: KokoroModelConfig) throws {
        model = try KokoroModel.load(from: weightsURL, config: config)
    }

    // MARK: - G2P Loading

    /// Load G2P dictionaries from URLs
    public func loadG2P(
        jiebaURL: URL,
        pinyinSingleURL: URL,
        pinyinPhrasesURL: URL? = nil
    ) throws {
        try g2p.initialize(
            jiebaURL: jiebaURL,
            pinyinSingleURL: pinyinSingleURL,
            pinyinPhrasesURL: pinyinPhrasesURL
        )
    }

    /// Load G2P dictionaries from data
    public func loadG2P(
        jiebaData: Data,
        pinyinSingleData: Data,
        pinyinPhrasesData: Data? = nil
    ) throws {
        try g2p.initialize(
            jiebaData: jiebaData,
            pinyinSingleData: pinyinSingleData,
            pinyinPhrasesData: pinyinPhrasesData
        )
    }

    // MARK: - Voice Loading

    /// Load a voice embedding from .npy file
    public func loadVoice(name: String, from url: URL) throws {
        let voiceArray = try MLX.loadArray(url: url)
        voiceEmbeddings[name] = voiceArray
    }

    /// Load a voice embedding from MLXArray
    public func setVoice(name: String, embedding: MLXArray) {
        voiceEmbeddings[name] = embedding
    }

    /// Get available voice names
    public var availableVoices: [String] {
        Array(voiceEmbeddings.keys)
    }

    // MARK: - Synthesis

    /// Synthesize speech from Chinese text
    ///
    /// - Parameters:
    ///   - text: Chinese text to synthesize
    ///   - voice: Voice name (must be loaded with loadVoice)
    ///   - speed: Speech speed (1.0 = normal, <1.0 = slower, >1.0 = faster)
    /// - Returns: Audio samples as Float32 array
    public func synthesize(
        _ text: String,
        voice: String,
        speed: Float = 1.0
    ) throws -> SynthesisResult {
        guard let model = model else {
            throw KokoroSynthesisError.modelNotLoaded
        }

        guard let voiceEmbedding = voiceEmbeddings[voice] else {
            throw KokoroSynthesisError.voiceNotFound(voice)
        }

        // Convert text to phonemes
        let phonemes = try g2p.convert(text)

        if phonemes.isEmpty {
            throw KokoroSynthesisError.emptyPhonemes
        }

        // Generate audio
        let output = model(phonemes: phonemes, refS: voiceEmbedding, speed: speed)

        // Convert MLXArray to Float array
        let audioData = output.audio.asArray(Float.self)

        return SynthesisResult(
            audio: audioData,
            sampleRate: model.config.sampleRate,
            phonemes: phonemes,
            durations: output.predDur.asArray(Int32.self)
        )
    }

    /// Synthesize speech and return raw MLXArray (for chaining with other MLX operations)
    public func synthesizeRaw(
        _ text: String,
        voice: String,
        speed: Float = 1.0
    ) throws -> KokoroOutput {
        guard let model = model else {
            throw KokoroSynthesisError.modelNotLoaded
        }

        guard let voiceEmbedding = voiceEmbeddings[voice] else {
            throw KokoroSynthesisError.voiceNotFound(voice)
        }

        let phonemes = try g2p.convert(text)

        if phonemes.isEmpty {
            throw KokoroSynthesisError.emptyPhonemes
        }

        return model(phonemes: phonemes, refS: voiceEmbedding, speed: speed)
    }

    /// Synthesize speech from raw phonemes (bypassing G2P)
    ///
    /// - Parameters:
    ///   - phonemes: Bopomofo phoneme string
    ///   - voice: Voice name (must be loaded with loadVoice)
    ///   - speed: Speech speed (1.0 = normal, <1.0 = slower, >1.0 = faster)
    /// - Returns: Audio samples as Float32 array
    public func synthesizeFromPhonemes(
        _ phonemes: String,
        voice: String,
        speed: Float = 1.0
    ) throws -> SynthesisResult {
        guard let model = model else {
            throw KokoroSynthesisError.modelNotLoaded
        }

        guard let voiceEmbedding = voiceEmbeddings[voice] else {
            throw KokoroSynthesisError.voiceNotFound(voice)
        }

        if phonemes.isEmpty {
            throw KokoroSynthesisError.emptyPhonemes
        }

        // Generate audio directly from phonemes
        let output = model(phonemes: phonemes, refS: voiceEmbedding, speed: speed)

        // Convert MLXArray to Float array
        let audioData = output.audio.asArray(Float.self)

        return SynthesisResult(
            audio: audioData,
            sampleRate: model.config.sampleRate,
            phonemes: phonemes,
            durations: output.predDur.asArray(Int32.self)
        )
    }
}

// MARK: - Synthesis Result

public struct SynthesisResult {
    /// Audio samples (Float32, mono)
    public let audio: [Float]

    /// Sample rate (Hz)
    public let sampleRate: Int

    /// Phoneme sequence used
    public let phonemes: String

    /// Predicted durations for each phoneme
    public let durations: [Int32]

    /// Duration of audio in seconds
    public var duration: Double {
        Double(audio.count) / Double(sampleRate)
    }
}

// MARK: - Errors

public enum KokoroSynthesisError: Error, LocalizedError {
    case modelNotLoaded
    case voiceNotFound(String)
    case emptyPhonemes
    case synthesisError(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Kokoro model not loaded. Call loadModel() first."
        case .voiceNotFound(let name):
            return "Voice '\(name)' not found. Load with loadVoice() first."
        case .emptyPhonemes:
            return "No phonemes generated from input text."
        case .synthesisError(let message):
            return "Synthesis error: \(message)"
        }
    }
}
