import Foundation

/// Configuration for Nemotron Speech Streaming Multilingual 0.6B
///
/// Loaded from `metadata.json`. Differs from the English variant in three ways:
///   1. Larger vocab (13,087 tokens) and matching `blank_idx`.
///   2. Smaller channel cache: `[1, 24, 56, 1024]` (att_context_size=[56, 0]).
///   3. The encoder takes an additional `prompt_id` int32 [1] input per chunk
///      which selects the language hint embedding. The model also emits a leading
///      `<xx-XX>` language-tag token whose IDs are listed in `lang_tag_token_ids`.
public struct NemotronMultilingualStreamingConfig: Sendable {
    /// Sample rate in Hz
    public let sampleRate: Int
    /// Number of mel spectrogram features
    public let melFeatures: Int
    /// Mel frames per chunk
    public let chunkMelFrames: Int
    /// Chunk duration in milliseconds (derived from chunkMelFrames * 10ms)
    public let chunkMs: Int
    /// Pre-encode cache size in mel frames (for encoder context)
    public let preEncodeCache: Int
    /// Total mel frames for encoder input (cache + chunk)
    public let totalMelFrames: Int
    /// Vocabulary size (excluding blank)
    public let vocabSize: Int
    /// Blank token index (== vocab_size)
    public let blankIdx: Int
    /// Encoder output dimension
    public let encoderDim: Int
    /// Decoder hidden size
    public let decoderHidden: Int
    /// Number of decoder LSTM layers
    public let decoderLayers: Int
    /// Encoder cache shapes
    public let cacheChannelShape: [Int]
    public let cacheTimeShape: [Int]

    /// Total number of prompt IDs supported by the model embedding table.
    public let numPrompts: Int
    /// Prompt ID used when caller does not specify a language ("auto" mode).
    public let defaultPromptId: Int
    /// Map from language code (e.g. `"en-US"`, `"zh-CN"`, `"auto"`) → prompt id.
    public let promptDictionary: [String: Int]
    /// Token IDs corresponding to `<xx-XX>` language-tag tokens emitted by the model.
    /// These should be filtered from the transcript text and the first one (if any)
    /// surfaced separately as the detected language.
    public let langTagTokenIds: Set<Int>

    /// Audio samples per chunk (10ms mel hop @ 16kHz → 160 samples/frame)
    public var chunkSamples: Int { chunkMelFrames * 160 }

    /// Default config for the 1120ms multilingual build.
    public init() {
        self.sampleRate = 16000
        self.melFeatures = 128
        self.chunkMelFrames = 112
        self.chunkMs = 1120
        self.preEncodeCache = 9
        self.totalMelFrames = 121
        self.vocabSize = 13087
        self.blankIdx = 13087
        self.encoderDim = 1024
        self.decoderHidden = 640
        self.decoderLayers = 2
        self.cacheChannelShape = [1, 24, 56, 1024]
        self.cacheTimeShape = [1, 24, 1024, 8]
        self.numPrompts = 128
        self.defaultPromptId = 101
        self.promptDictionary = ["auto": 101]
        self.langTagTokenIds = []
    }

    /// Load config from `metadata.json` produced by the multilingual conversion pipeline.
    public init(from metadataURL: URL) throws {
        let data = try Data(contentsOf: metadataURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ASRError.processingFailed("Invalid multilingual metadata.json format")
        }

        let chunkMelFrames = json["chunk_mel_frames"] as? Int ?? 112
        self.sampleRate = json["sample_rate"] as? Int ?? 16000
        self.melFeatures = json["mel_features"] as? Int ?? 128
        self.chunkMelFrames = chunkMelFrames
        self.chunkMs = (json["chunk_ms"] as? Int) ?? (chunkMelFrames * 10)
        self.preEncodeCache = json["pre_encode_cache"] as? Int ?? 9
        self.totalMelFrames = json["total_mel_frames"] as? Int ?? (chunkMelFrames + 9)
        self.vocabSize = json["vocab_size"] as? Int ?? 13087
        self.blankIdx = json["blank_idx"] as? Int ?? (json["vocab_size"] as? Int ?? 13087)
        self.encoderDim = json["encoder_dim"] as? Int ?? 1024
        self.decoderHidden = json["decoder_hidden"] as? Int ?? 640
        self.decoderLayers = json["decoder_layers"] as? Int ?? 2
        self.cacheChannelShape = json["cache_channel_shape"] as? [Int] ?? [1, 24, 56, 1024]
        self.cacheTimeShape = json["cache_time_shape"] as? [Int] ?? [1, 24, 1024, 8]
        self.numPrompts = json["num_prompts"] as? Int ?? 128
        self.defaultPromptId = json["default_prompt_id"] as? Int ?? 101

        let dict = json["prompt_dictionary"] as? [String: Int] ?? ["auto": 101]
        self.promptDictionary = dict

        let langTagIds = json["lang_tag_token_ids"] as? [Int] ?? []
        self.langTagTokenIds = Set(langTagIds)
    }

    /// Resolve a caller-supplied language identifier to a prompt id.
    ///
    /// - Parameter language: A language code (`"en-US"`, `"zh-CN"`, `"auto"`, …)
    ///   or `nil` to use `defaultPromptId`. Lookup is case-sensitive against
    ///   `promptDictionary`; if the exact key is missing, common fallbacks are
    ///   tried (`"en"` from `"en_us"`, etc.), then `defaultPromptId` is returned.
    public func promptId(forLanguage language: String?) -> Int {
        guard let language, !language.isEmpty else { return defaultPromptId }

        if let direct = promptDictionary[language] {
            return direct
        }

        // Try common normalizations: "en_us" -> "en-US", "EN-us" -> "en-US"
        let dashed = language.replacingOccurrences(of: "_", with: "-")
        if let id = promptDictionary[dashed] {
            return id
        }

        // Normalize casing on region: "en-us" -> "en-US"
        let parts = dashed.split(separator: "-", maxSplits: 1).map(String.init)
        if parts.count == 2 {
            let normalized = parts[0].lowercased() + "-" + parts[1].uppercased()
            if let id = promptDictionary[normalized] {
                return id
            }
        }

        // Bare language fallback: "en-US" -> "en"
        if let bare = parts.first, let id = promptDictionary[bare.lowercased()] {
            return id
        }

        return defaultPromptId
    }
}
