@preconcurrency import CoreML
import Foundation

/// Wraps `nanocodec_decoder*.mlmodelc`. Three builds are supported,
/// dispatched automatically from the model's input shape (the precision
/// of a t24 build is opaque to this wrapper — `MagpieModelStore` decides
/// which file to load):
///
/// - **Monolithic T=256** (`nanocodec_decoder.mlmodelc`): single call, the
///   input row count = 256 codec frames, output = 262144 audio samples.
///   Runs on CPU only because the activation tensor exceeds the ANE's
///   `W ≤ 16384` limit on the space-to-batch lowering of dilated convs.
///   Legacy fallback only.
///
/// - **Chunked T_in=24, fp32** (`nanocodec_decoder_t24_v2.mlmodelc`,
///   default): 24-frame input, 24576 audio samples per call. fp32
///   weights — pinned to CPU because ANE is fp16-only. Audibly clean,
///   matches the PyTorch sin² reference within the Snake-approximation
///   noise floor. Nanocodec wall ~8.5–9.7 s on a ~11 s utterance (M2),
///   still real-time at RTFx ~1.3× end-to-end.
///
/// - **Chunked T_in=24, fp16** (`nanocodec_decoder_t24.mlmodelc`): same
///   shape contract as the fp32 build. Runs ~43 % ANE-resident at
///   ~38.4 ms / 24-frame call, so ~4× faster than fp32, but fp16 weight
///   quantization adds ~27 dB of speech-correlated noise that silence-
///   RMS metrics hide. Phase F (per-op + per-location mixed-precision
///   sweep, see `mobius/.../per_module/results/STATUS.md`) confirmed
///   no mixed-precision island recovers cleanliness. Use only when
///   throughput dominates quality.
///
/// In both chunked builds, the runtime slides the 24-frame window with
/// stride 8 and overlap 16 frames over the codec sequence and
/// concatenates the trailing 8192 audio samples of each call.
///
/// The 16-frame overlap is the dilated-conv stack's input-side receptive
/// field; below 16 frames of left context, [`per_module/chunked_parity.py`]
/// in `mobius` measures < 6 dB SNR vs the single-call reference, which is
/// audibly broken at boundaries. At 16 frames overlap SNR matches the
/// Taylor5Clipped Snake noise floor (~11.5 dB).
public struct MagpieNanocodec {

    public let model: MLModel
    public let numCodebooks: Int
    /// Input frames per `model.prediction(...)` call. 256 for the monolithic
    /// build, 24 for the chunked build. Detected from the model's input
    /// description at init time.
    public let tIn: Int
    /// Fresh frames produced per call. Equals `tIn` for the monolithic
    /// build (no chunking) and `tIn - overlap` for the chunked build.
    public let stride: Int
    public let samplesPerFrame: Int

    /// Receptive field of the dilated-conv stack at the codec input level,
    /// measured empirically in `mobius/.../per_module/chunked_parity.py`.
    private static let receptiveFieldFrames: Int = 16

    public init(
        model: MLModel,
        numCodebooks: Int = MagpieConstants.numCodebooks,
        samplesPerFrame: Int = MagpieConstants.codecSamplesPerFrame
    ) {
        self.model = model
        self.numCodebooks = numCodebooks
        self.samplesPerFrame = samplesPerFrame
        let detected = Self.detectTIn(
            model: model, fallback: MagpieConstants.maxNanocodecFrames)
        self.tIn = detected
        // Monolithic single-call build (tIn = full sequence): no chunking.
        // Chunked build: stride = tIn - 16 receptive-field overlap.
        if detected >= MagpieConstants.maxNanocodecFrames {
            self.stride = detected
        } else {
            self.stride = max(1, detected - Self.receptiveFieldFrames)
        }
    }

    private var overlap: Int { tIn - stride }

    /// - Parameter frames: row-major `[numCodebooks][Ttotal]` codes.
    /// - Returns: `Ttotal * samplesPerFrame` fp32 PCM samples.
    public func decode(frames: [[Int32]]) throws -> [Float] {
        precondition(frames.count == numCodebooks, "expected \(numCodebooks) codebook rows")
        let tTotal = frames[0].count
        if tTotal == 0 {
            return []
        }
        let totalSamples = tTotal * samplesPerFrame
        var output = Swift.Array<Float>(repeating: 0, count: totalSamples)

        // Reusable input tensor — same shape every call.
        let tokens = try MLMultiArray(
            shape: [1, NSNumber(value: numCodebooks), NSNumber(value: tIn)],
            dataType: .int32)

        // Slide a tIn-frame window with `stride` fresh frames per call.
        // Out-of-range indices use **edge replication** rather than zero
        // padding: the first call's left context replays `row[0]`, the
        // last call's right context replays `row[tTotal - 1]`. Code 0 is
        // a real codebook entry the codec was never trained to see in
        // sequence, so zero-padding produces a sharp pop in the first
        // ~30 ms of the rendered audio (measured at peak 0.64 vs mono's
        // 0.001). Edge replication makes the dilated convs see a
        // stationary signal that matches the AR loop's near-silent first
        // frame, killing the transient.
        var outFrame = 0
        let overlap = self.overlap
        let lastIdx = tTotal - 1
        while outFrame < tTotal {
            let ctxStart = outFrame - overlap
            tokens.withUnsafeMutableBytes { rawPtr, _ in
                let base = rawPtr.bindMemory(to: Int32.self).baseAddress!
                for cb in 0..<numCodebooks {
                    let row = frames[cb]
                    let rowOffset = cb * tIn
                    for t in 0..<tIn {
                        let src = ctxStart + t
                        let clamped = max(0, min(lastIdx, src))
                        base[rowOffset + t] = row[clamped]
                    }
                }
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "tokens": MLFeatureValue(multiArray: tokens)
            ])
            let result = try model.prediction(from: provider)
            guard let audio = result.featureValue(for: "audio")?.multiArrayValue else {
                throw MagpieError.inferenceFailed(
                    stage: "nanocodec", underlying: "missing 'audio' output key")
            }

            // Audio buffer is (1, tIn * samplesPerFrame) fp32. Discard the
            // first `overlap * samplesPerFrame` samples (dilated convs'
            // warmup region) and copy the next `stride * samplesPerFrame`
            // samples into the output, clamped to the audio buffer length
            // and to the requested totalSamples.
            let writeStart = outFrame * samplesPerFrame
            let keepStart = overlap * samplesPerFrame
            let writeCount = min(
                stride * samplesPerFrame,
                totalSamples - writeStart,
                audio.count - keepStart)
            if writeCount > 0 {
                audio.withUnsafeBytes { raw in
                    let ptr = raw.bindMemory(to: Float.self)
                    for i in 0..<writeCount {
                        output[writeStart + i] = ptr[keepStart + i]
                    }
                }
            }
            outFrame += stride
        }
        return output
    }

    /// Read the `tokens` input shape from the model description and return
    /// the third dimension (frame count). Falls back to `fallback` if the
    /// description is missing or malformed.
    private static func detectTIn(model: MLModel, fallback: Int) -> Int {
        guard let description = model.modelDescription.inputDescriptionsByName["tokens"],
            let constraint = description.multiArrayConstraint
        else {
            return fallback
        }
        let shape = constraint.shape
        guard shape.count >= 3 else {
            return fallback
        }
        let value = shape[2].intValue
        return value > 0 ? value : fallback
    }
}
