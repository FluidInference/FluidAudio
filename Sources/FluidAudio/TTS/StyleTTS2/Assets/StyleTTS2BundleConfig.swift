import Foundation

/// Decoded view of `config.json` shipped alongside the StyleTTS2 mlmodelc
/// bundle on HuggingFace.
///
/// The on-disk file is the source of truth for the runtime contract between
/// the host (Swift) and the converted CoreML stages. We decode the
/// must-have fields here and validate them against `StyleTTS2Constants`
/// during `initialize()` to catch "wrong bundle version" / "swapped cache
/// dir" errors before the first synthesis call.
public struct StyleTTS2BundleConfig: Sendable, Codable {

    public struct Audio: Sendable, Codable {
        public let sampleRate: Int
        public let nFFT: Int
        public let winLength: Int
        public let hopLength: Int
        public let nMels: Int

        enum CodingKeys: String, CodingKey {
            case sampleRate = "sample_rate"
            case nFFT = "n_fft"
            case winLength = "win_length"
            case hopLength = "hop_length"
            case nMels = "n_mels"
        }
    }

    public struct Tokenizer: Sendable, Codable {
        public let type: String
        public let vocabFile: String
        public let nTokens: Int
        public let padToken: String
        public let padId: Int

        enum CodingKeys: String, CodingKey {
            case type
            case vocabFile = "vocab_file"
            case nTokens = "n_tokens"
            case padToken = "pad_token"
            case padId = "pad_id"
        }
    }

    public struct Model: Sendable, Codable {
        public let styleDim: Int
        public let hiddenDim: Int
        public let nLayer: Int
        public let maxDur: Int
        public let refStyleDim: Int

        enum CodingKeys: String, CodingKey {
            case styleDim = "style_dim"
            case hiddenDim = "hidden_dim"
            case nLayer = "n_layer"
            case maxDur = "max_dur"
            case refStyleDim = "ref_s_dim"
        }
    }

    public struct Sampler: Sendable, Codable {
        public let type: String
        public let schedule: String
        public let numSteps: Int
        public let classifierFreeGuidance: Bool
        public let cfgScaleDefault: Float

        enum CodingKeys: String, CodingKey {
            case type
            case schedule
            case numSteps = "num_steps"
            case classifierFreeGuidance = "classifier_free_guidance"
            case cfgScaleDefault = "cfg_scale_default"
        }
    }

    public let modelType: String
    public let audio: Audio
    public let tokenizer: Tokenizer
    public let model: Model
    public let sampler: Sampler

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audio
        case tokenizer
        case model
        case sampler
    }

    /// Load and decode `config.json`.
    public static func load(from url: URL) throws -> StyleTTS2BundleConfig {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StyleTTS2Error.modelNotFound(url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        do {
            return try JSONDecoder().decode(StyleTTS2BundleConfig.self, from: data)
        } catch {
            throw StyleTTS2Error.invalidConfiguration(
                "\(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    /// Validate the bundle matches `StyleTTS2Constants`.
    ///
    /// Throws if any must-have value drifts from what the host expects.
    /// Catches: wrong cache directory, partial download, ship-version mismatch.
    public func validate() throws {
        guard modelType == "styletts2" else {
            throw StyleTTS2Error.invalidConfiguration(
                "model_type=\"\(modelType)\", expected \"styletts2\"")
        }
        try expect(
            audio.sampleRate, equals: StyleTTS2Constants.audioSampleRate,
            field: "audio.sample_rate")
        try expect(
            audio.hopLength, equals: StyleTTS2Constants.hopSize,
            field: "audio.hop_length")
        try expect(
            audio.nMels, equals: StyleTTS2Constants.melChannels,
            field: "audio.n_mels")
        try expect(
            audio.nFFT, equals: StyleTTS2Constants.nFFT, field: "audio.n_fft")
        try expect(
            audio.winLength, equals: StyleTTS2Constants.winLength,
            field: "audio.win_length")

        try expect(
            tokenizer.nTokens, equals: StyleTTS2Constants.vocabSize,
            field: "tokenizer.n_tokens")
        try expect(
            tokenizer.padId, equals: StyleTTS2Constants.padTokenId,
            field: "tokenizer.pad_id")

        try expect(
            model.styleDim, equals: StyleTTS2Constants.styleDim,
            field: "model.style_dim")
        try expect(
            model.hiddenDim, equals: StyleTTS2Constants.hiddenDim,
            field: "model.hidden_dim")
        try expect(
            model.refStyleDim, equals: StyleTTS2Constants.refStyleDim,
            field: "model.ref_s_dim")
    }

    private func expect(_ actual: Int, equals expected: Int, field: String) throws {
        guard actual == expected else {
            throw StyleTTS2Error.invalidConfiguration(
                "\(field)=\(actual), expected \(expected)")
        }
    }
}
