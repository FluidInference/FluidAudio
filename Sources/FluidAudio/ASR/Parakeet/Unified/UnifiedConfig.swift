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

/// Language prompt for `nvidia/nemotron-labs-unified-asr-0.6b` (25 European languages + `auto`).
///
/// The multilingual checkpoint applies a language-ID prompt after the encoder; the CoreML encoders take it as an
/// int32 `prompt_id` input. Ids come from the checkpoint's `prompt_dictionary` (also in the HF repo's
/// `coreml/metadata.json`). For low-latency streaming NVIDIA recommends passing the language explicitly.
public enum NemotronUnifiedLanguage: String, Sendable, CaseIterable {
    case auto
    case bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk

    public var promptId: Int {
        switch self {
        case .auto: return 127
        case .en: return 0
        case .es: return 3
        case .fr: return 8
        case .de: return 9
        case .ru: return 11
        case .pt: return 13
        case .it: return 15
        case .nl: return 16
        case .pl: return 17
        case .uk: return 19
        case .ro: return 20
        case .el: return 21
        case .cs: return 22
        case .hu: return 23
        case .sv: return 24
        case .da: return 25
        case .fi: return 26
        case .sk: return 28
        case .hr: return 29
        case .bg: return 30
        case .lt: return 31
        case .et: return 60
        case .lv: return 61
        case .sl: return 62
        case .mt: return 102
        }
    }
}

/// Which Unified FastConformer-RNNT checkpoint the Unified managers run.
///
/// - `parakeetEnglish`: `nvidia/parakeet-unified-en-0.6b` (English, 1024 tokens + blank).
/// - `nemotronMultilingual`: `nvidia/nemotron-labs-unified-asr-0.6b` (25 European languages, 4095 tokens + blank,
///   language prompt fed to the encoder as `prompt_id`).
///
/// Both share the encoder layout, mel front end, decoder/joint contract and streaming contexts; the family picks
/// the HF repo, bundle names, blank index and prompt.
public enum UnifiedModelFamily: Sendable, Equatable {
    case parakeetEnglish
    case nemotronMultilingual(NemotronUnifiedLanguage)

    public var repo: Repo {
        switch self {
        case .parakeetEnglish: return .parakeetUnified
        case .nemotronMultilingual: return .nemotronUnified
        }
    }

    /// Blank token id (= vocabulary size).
    public var blankIdx: Int {
        switch self {
        case .parakeetEnglish: return 1024
        case .nemotronMultilingual: return 4095
        }
    }

    /// Encoder `prompt_id` input, or nil for the English checkpoint (no prompt input).
    public var promptId: Int? {
        switch self {
        case .parakeetEnglish: return nil
        case .nemotronMultilingual(let language): return language.promptId
        }
    }

    public var displayName: String {
        switch self {
        case .parakeetEnglish: return "Parakeet Unified 0.6B"
        case .nemotronMultilingual(let language): return "Nemotron Unified ASR 0.6B (\(language.rawValue))"
        }
    }

    var filePrefix: String {
        switch self {
        case .parakeetEnglish: return "parakeet_unified"
        case .nemotronMultilingual: return "nemotron_unified"
        }
    }

    public func streamingEncoderFile(precision: UnifiedEncoderPrecision, contextSuffix: String) -> String {
        let base = "\(filePrefix)_encoder_streaming_\(contextSuffix)"
        return precision == .int8 ? "\(base)_int8.mlmodelc" : "\(base).mlmodelc"
    }

    public func offlineEncoderFile(precision: UnifiedEncoderPrecision) -> String {
        precision == .int8 ? "\(filePrefix)_encoder_int8.mlmodelc" : "\(filePrefix)_encoder.mlmodelc"
    }

    public var decoderFile: String { "\(filePrefix)_decoder.mlmodelc" }
    public var jointDecisionFile: String { "\(filePrefix)_joint_decision_single_step.mlmodelc" }
    public var vocab: String { "vocab.json" }
    public var metadata: String { "metadata.json" }
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
    /// Checkpoint family (English Parakeet or multilingual Nemotron) and, for Nemotron, the language prompt.
    public let family: UnifiedModelFamily
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

    /// Offline batch path: take a seam token's timing from the window that
    /// decoded it away from its own edge (default `true`), as
    /// `ASRConfig.seamTimingRealignment` does for the TDT path — see "Seam
    /// Timing" in Documentation/ASR/LongTranscription.md.
    public let seamTimingRealignment: Bool

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
        blankIdx: Int? = nil,
        maxSymbolsPerFrame: Int = 10,
        seamTimingRealignment: Bool = true,
        family: UnifiedModelFamily = .parakeetEnglish
    ) {
        self.family = family
        self.leftFrames = leftFrames
        self.chunkFrames = chunkFrames
        self.rightFrames = rightFrames
        self.sampleRate = sampleRate
        self.frameSamples = frameSamples
        self.melFeatures = melFeatures
        self.decoderLayers = decoderLayers
        self.decoderHidden = decoderHidden
        self.blankIdx = blankIdx ?? family.blankIdx
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
        self.seamTimingRealignment = seamTimingRealignment
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
