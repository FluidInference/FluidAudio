import Foundation

/// Encoder weight precision for Parakeet Unified 0.6B.
///
/// int8 (per-channel linear symmetric) is the default: identical LibriSpeech
/// test-clean WER to fp16 (offline 1.83% vs 1.82%, streaming 2.14% vs 2.15%),
/// identical ANE latency on M-series, and half the download/disk (~565 MB vs
/// ~1.1 GB per encoder).
public enum UnifiedEncoderPrecision: String, Sendable, CaseIterable {
    case int8
    case fp16
}

/// Shared configuration for the Parakeet Unified 0.6B backend, used by both the
/// streaming (`StreamingUnifiedAsrManager`) and offline-batch (`UnifiedAsrManager`)
/// managers.
///
/// The `leftFrames`/`chunkFrames`/`rightFrames` fields describe the streaming
/// chunked-attention window (the model re-runs its stateless encoder over a
/// `[left | chunk | right]` window whose attention mask was baked in at CoreML
/// conversion time); the offline batch path ignores them and uses its own fixed
/// 15 s window. The remaining fields (sample rate, frame size, decoder/joint
/// shapes) are common to both. Context sizes are in 80 ms encoder frames
/// (1280 samples @ 16 kHz), matching the conversion pipeline in mobius
/// `models/stt/parakeet-unified-en-0.6b/coreml`.
public struct UnifiedConfig: Sendable {
    /// Left context in encoder frames (history visible to each streaming chunk).
    public let leftFrames: Int
    /// Chunk size in encoder frames (new audio decoded per streaming step).
    public let chunkFrames: Int
    /// Right context in encoder frames (streaming look-ahead; adds latency).
    public let rightFrames: Int

    public let sampleRate: Int
    /// Samples per encoder frame (80 ms @ 16 kHz).
    public let frameSamples: Int
    public let melFeatures: Int
    public let decoderLayers: Int
    public let decoderHidden: Int
    public let blankIdx: Int
    public let maxSymbolsPerFrame: Int

    /// Default streaming export: [70, 13, 13] = 5.6 s left, 1.04 s chunk,
    /// 1.04 s right (2.08 s theoretical latency — the model card's best-WER
    /// streaming mode). The offline batch path is unaffected by these values.
    public init(
        leftFrames: Int = 70,
        chunkFrames: Int = 13,
        rightFrames: Int = 13,
        sampleRate: Int = 16000,
        frameSamples: Int = 1280,
        melFeatures: Int = 128,
        decoderLayers: Int = 2,
        decoderHidden: Int = 640,
        blankIdx: Int = 1024,
        maxSymbolsPerFrame: Int = 10
    ) {
        self.leftFrames = leftFrames
        self.chunkFrames = chunkFrames
        self.rightFrames = rightFrames
        self.sampleRate = sampleRate
        self.frameSamples = frameSamples
        self.melFeatures = melFeatures
        self.decoderLayers = decoderLayers
        self.decoderHidden = decoderHidden
        self.blankIdx = blankIdx
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
    }

    // MARK: Streaming-window derived sizes

    /// Total streaming encoder window in samples (left + chunk + right).
    public var windowSamples: Int { (leftFrames + chunkFrames + rightFrames) * frameSamples }
    public var chunkSamples: Int { chunkFrames * frameSamples }
    public var rightSamples: Int { rightFrames * frameSamples }
    /// Theoretical streaming latency in milliseconds (chunk + right context).
    public var latencyMs: Int { (chunkFrames + rightFrames) * frameSamples * 1000 / sampleRate }
    /// Suffix used in streaming encoder file names (e.g. "70_13_13").
    public var contextSuffix: String { "\(leftFrames)_\(chunkFrames)_\(rightFrames)" }
}
