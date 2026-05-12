@preconcurrency import CoreML
import Foundation

/// Wraps `nanocodec_decoder*.mlmodelc`. Dispatches by input shape:
///
/// - v1 (`nanocodec_decoder`): T=256 mono, fp16, CPU. Legacy fallback.
/// - v2 (`nanocodec_decoder_v2`): T_in=24 chunked, fp16. Fast, noisy.
/// - v3 (`nanocodec_decoder_v3`, default): T_in=24 chunked, fp32, CPU. Clean.
///
/// Chunked builds slide a 24-frame window with stride 8 / overlap 16
/// (= dilated-conv input receptive field).
///
/// **Upstream behavior.** NeMo's reference inference
/// (`MagpieTTSModel.infer_batch` / `do_tts` in
/// `nemo/collections/tts/models/magpietts.py`, ≈ lines 5334–5351 and
/// 6889–6891) invokes `_codec_helper.codes_to_audio(...)` exactly once
/// per utterance, after the AR loop has accumulated every codebook frame.
/// The v2/v3 chunked sliding-window dispatch implemented here is purely
/// a memory + first-audio-latency optimization on the CoreML side; it is
/// not an indication that the codec or the model as a whole is streaming.
public struct MagpieNanocodec {

    public let model: MLModel
    public let numCodebooks: Int
    /// Input frames per call (256 mono, 24 chunked). Detected from input shape.
    public let tIn: Int
    /// Fresh frames produced per call (= `tIn - overlap`, or `tIn` for mono).
    public let stride: Int
    public let samplesPerFrame: Int

    /// Dilated-conv input receptive field. Empirical floor for clean seams.
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
        if detected >= MagpieConstants.maxNanocodecFrames {
            self.stride = detected  // mono: no chunking
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

        // Slide a tIn-frame window. Out-of-range indices use edge
        // replication; zero-padding produces an audible pop because
        // code 0 is a real, untrained-in-sequence codebook entry.
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

            // Drop the overlap warmup, copy the next `stride` frames of audio.
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

    /// Read frame count from the `tokens` input shape's third dimension.
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
