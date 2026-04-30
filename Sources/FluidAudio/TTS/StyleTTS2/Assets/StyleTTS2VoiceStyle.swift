import Foundation

/// A precomputed `ref_s` style+prosody reference vector for StyleTTS2.
///
/// `ref_s` is the 256-dim concat of:
///   - `style_encoder(mel)` — first 128 dims, "acoustic" branch
///   - `predictor_encoder(mel)` — last 128 dims, "prosody" branch
///
/// Upstream computes this in PyTorch from a reference WAV and feeds it into
/// the diffusion sampler + the text predictor. Until the StyleTTS2 style
/// encoders are exported as a CoreML stage (follow-up work), the Swift host
/// ships a precomputed `ref_s.bin` blob per voice — produced offline by
/// `mobius-styletts2/scripts/06_dump_ref_s.py` from any 24 kHz mono WAV.
///
/// File format: 256 little-endian fp32 values, 1024 bytes total. No header.
public struct StyleTTS2VoiceStyle: Sendable, Equatable {

    /// Full 256-dim `ref_s` vector (acoustic ⧺ prosody).
    public let concatenated: [Float]

    public init(concatenated: [Float]) {
        precondition(
            concatenated.count == StyleTTS2Constants.refStyleDim,
            "ref_s must be \(StyleTTS2Constants.refStyleDim) floats, got \(concatenated.count)")
        self.concatenated = concatenated
    }

    /// 128-dim acoustic style branch (`ref_s[:, :128]`).
    public var acoustic: ArraySlice<Float> {
        concatenated.prefix(StyleTTS2Constants.styleDim)
    }

    /// 128-dim prosody branch (`ref_s[:, 128:]`).
    public var prosody: ArraySlice<Float> {
        concatenated.suffix(StyleTTS2Constants.styleDim)
    }

    /// Load a precomputed `ref_s.bin` blob (256 fp32 little-endian, 1024 bytes).
    public static func load(from url: URL) throws -> StyleTTS2VoiceStyle {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StyleTTS2Error.modelNotFound(url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        let expectedBytes = StyleTTS2Constants.refStyleDim * MemoryLayout<Float>.size
        guard data.count == expectedBytes else {
            throw StyleTTS2Error.invalidConfiguration(
                "\(url.lastPathComponent): expected \(expectedBytes) bytes "
                    + "(\(StyleTTS2Constants.refStyleDim) fp32), got \(data.count)")
        }

        // The blob is little-endian fp32 by contract (see Python dumper).
        // On Apple Silicon / Intel that matches host byte order, so a direct
        // memory load is correct. We still copy out into a `[Float]` so the
        // value is fully owned and Sendable.
        var floats = [Float](repeating: 0, count: StyleTTS2Constants.refStyleDim)
        floats.withUnsafeMutableBufferPointer { dst in
            _ = data.copyBytes(to: dst)
        }
        return StyleTTS2VoiceStyle(concatenated: floats)
    }
}
