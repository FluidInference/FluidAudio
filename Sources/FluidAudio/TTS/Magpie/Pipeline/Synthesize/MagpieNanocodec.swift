@preconcurrency import CoreML
import Foundation

/// Wraps the `nanocodec_decoder.mlmodelc` model. Takes `(numCodebooks, Ttotal)`
/// int32 codes, pads to `maxFrames = 256`, runs the decoder, returns fp32 PCM.
public struct MagpieNanocodec {

    public let model: MLModel
    public let numCodebooks: Int
    public let maxFrames: Int
    public let samplesPerFrame: Int

    public init(
        model: MLModel,
        numCodebooks: Int = MagpieConstants.numCodebooks,
        maxFrames: Int = MagpieConstants.maxNanocodecFrames,
        samplesPerFrame: Int = MagpieConstants.codecSamplesPerFrame
    ) {
        self.model = model
        self.numCodebooks = numCodebooks
        self.maxFrames = maxFrames
        self.samplesPerFrame = samplesPerFrame
    }

    /// - Parameter frames: row-major `[numCodebooks][Ttotal]` codes.
    public func decode(frames: [[Int32]]) throws -> [Float] {
        precondition(frames.count == numCodebooks, "expected \(numCodebooks) codebook rows")
        let tTotal = min(frames[0].count, maxFrames)

        // Build tokens tensor: (1, numCodebooks, maxFrames) int32, zero-padded.
        let tokens = try MLMultiArray(
            shape: [1, NSNumber(value: numCodebooks), NSNumber(value: maxFrames)],
            dataType: .int32)
        tokens.withUnsafeMutableBytes { ptr, strides in
            let base = ptr.bindMemory(to: Int32.self).baseAddress!
            let total = numCodebooks * maxFrames
            for i in 0..<total { base[i] = 0 }
            for cb in 0..<numCodebooks {
                for t in 0..<tTotal {
                    base[cb * maxFrames + t] = frames[cb][t]
                }
            }
            _ = strides
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLFeatureValue(multiArray: tokens)
        ])
        let output = try model.prediction(from: provider)
        guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
            throw MagpieError.inferenceFailed(
                stage: "nanocodec", underlying: "missing 'audio' output key")
        }

        let expected = tTotal * samplesPerFrame
        var samples = Swift.Array<Float>(repeating: 0, count: expected)
        audio.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: Float.self)
            let available = min(expected, audio.count)
            for i in 0..<available {
                samples[i] = ptr[i]
            }
        }
        return samples
    }
}
