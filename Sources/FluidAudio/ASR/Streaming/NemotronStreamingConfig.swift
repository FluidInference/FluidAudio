import Foundation

/// Configuration for Nemotron Speech Streaming 0.6B
/// Loaded from metadata.json for each chunk size variant
public struct NemotronStreamingConfig: Sendable {
    /// Sample rate in Hz
    public let sampleRate: Int
    /// Number of mel spectrogram features
    public let melFeatures: Int
    /// Mel frames per chunk
    public let chunkMelFrames: Int
    /// Chunk duration in milliseconds
    public let chunkMs: Int
    /// Pre-encode cache size in mel frames (for encoder context)
    public let preEncodeCache: Int
    /// Total mel frames for encoder input (cache + chunk)
    public let totalMelFrames: Int
    /// Vocabulary size
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

    /// Audio samples per chunk
    public var chunkSamples: Int { chunkMelFrames * 160 }

    /// Default config for 1120ms chunks (backward compatibility)
    public init() {
        self.sampleRate = 16000
        self.melFeatures = 128
        self.chunkMelFrames = 112
        self.chunkMs = 1120
        self.preEncodeCache = 9
        self.totalMelFrames = 121
        self.vocabSize = 1024
        self.blankIdx = 1024
        self.encoderDim = 1024
        self.decoderHidden = 640
        self.decoderLayers = 2
        self.cacheChannelShape = [1, 24, 70, 1024]
        self.cacheTimeShape = [1, 24, 1024, 8]
    }

    /// Load config from metadata.json
    public init(from metadataURL: URL) throws {
        let data = try Data(contentsOf: metadataURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ASRError.processingFailed("Invalid metadata.json format")
        }

        self.sampleRate = json["sample_rate"] as? Int ?? 16000
        self.melFeatures = json["mel_features"] as? Int ?? 128
        self.chunkMelFrames = json["chunk_mel_frames"] as? Int ?? 112
        self.chunkMs = json["chunk_ms"] as? Int ?? 1120
        self.preEncodeCache = json["pre_encode_cache"] as? Int ?? 9
        self.totalMelFrames = json["total_mel_frames"] as? Int ?? 121
        self.vocabSize = json["vocab_size"] as? Int ?? 1024
        self.blankIdx = json["blank_idx"] as? Int ?? 1024
        self.encoderDim = json["encoder_dim"] as? Int ?? 1024
        self.decoderHidden = json["decoder_hidden"] as? Int ?? 640
        self.decoderLayers = json["decoder_layers"] as? Int ?? 2
        self.cacheChannelShape = json["cache_channel_shape"] as? [Int] ?? [1, 24, 70, 1024]
        self.cacheTimeShape = json["cache_time_shape"] as? [Int] ?? [1, 24, 1024, 8]
    }
}
