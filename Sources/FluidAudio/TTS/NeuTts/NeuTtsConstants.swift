import Foundation

/// Compile-time constants for the NeuTTS-2E backend (emotional English TTS).
///
/// Pipeline: Qwen3 236M backbone emits NeuCodec speech tokens autoregressively
/// (50 codes/s), decoded to 24 kHz audio by the NeuCodec decoder. Models are
/// converted in mobius (`models/tts/neutts-2e/coreml`) and published to
/// `FluidInference/neutts-2e-coreml`.
///
/// Note: upstream applies a Perth watermark to generated audio in the host
/// app; that postprocessing is not implemented here.
public enum NeuTtsConstants {
    public static let sampleRate = 24_000
    /// Audio samples per speech code (50 codes/s at 24 kHz).
    public static let hopLength = 480
    /// Codec code vocabulary (FSQ 8×4 levels).
    public static let codebookSize = 65_536

    /// Static prefill window baked into the prefill model.
    public static let prefillLength = 768
    /// KV-cache capacity baked into the decode model (prompt + generation).
    public static let maxContext = 2048
    /// Codec decoder flexible-length bounds (RangeDim).
    public static let maxCodecCodes = 2000

    /// Upstream sampling defaults.
    public static let temperature: Float = 1.0
    public static let topK = 50
    /// Upstream forces at least this many new tokens before EOS is allowed.
    public static let minNewTokens = 50

    public static let speakers = ["emily", "paul", "sophie", "steven"]
    public static let emotions = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

    public static let defaultSpeaker = "emily"
    public static let defaultEmotion = "neutral"

    /// Number of transformer layers (one kv_k/kv_v state pair each).
    public static let layerCount = 28
    public static let kvHeads = 4
    public static let headDim = 128
}
