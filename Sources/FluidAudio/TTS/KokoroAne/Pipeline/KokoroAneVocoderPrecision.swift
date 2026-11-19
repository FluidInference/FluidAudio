#if TTS
import Foundation

/// Compute precision of the legacy chain's Vocoder stage.
public enum KokoroAneVocoderPrecision: String, CaseIterable, Sendable {
    /// `KokoroVocoder` (fp16 compute). The default.
    case float16
    /// `KokoroVocoder_v2` (fp32 compute, same I/O and size). On the Core ML CPU
    /// path of SME chips (A19 Pro, A20, M5) the fp16 Vocoder corrupts the first
    /// ~1 s of long utterances (robotic onset); fp32 fixes it at ~2.8x the
    /// Vocoder time on iPhone 17 Pro Max. Legacy runtime only (#1135).
    case float32
}
#endif
