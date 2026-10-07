@preconcurrency import CoreML
import Foundation

/// Drives the two-model Paradee pipeline for one chunk:
/// `ParadeeText` → host duration rounding + column expansion + source noise →
/// `ParadeeAcoustic`. Both CoreML graphs are deterministic; the only
/// randomness is the harmonic source noise generated here.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
enum ParadeeSynthesizer {

    /// Synthesize one chunk of `[0, ...ids, 0]` tokens. Returns 24 kHz mono PCM.
    static func synthesize(
        inputIds: [Int32],
        speed: Float,
        noiseSeed: UInt64,
        text: MLModel,
        acoustic: MLModel
    ) throws -> [Float] {
        let tokens = inputIds.count
        guard tokens >= 3, tokens <= ParadeeConstants.maxPhonemeLength + 2 else {
            throw ParadeeError.inputProcessingFailed(
                "token count \(tokens) out of range (3...\(ParadeeConstants.maxPhonemeLength + 2))")
        }
        guard speed > 0 else {
            throw ParadeeError.inputProcessingFailed("speed must be > 0, got \(speed)")
        }

        let textOut = try predict(text, ["input_ids": try multiArray(inputIds, shape: [1, tokens])])
        let durations = try readChannelsByTime(textOut, "duration", channels: 1, time: tokens)
        let d = try readChannelsByTime(textOut, "d", channels: ParadeeConstants.prosodyChannels, time: tokens)
        let asrTok = try readChannelsByTime(
            textOut, "asr_tok", channels: ParadeeConstants.textChannels, time: tokens)

        let counts = frameCounts(durations: durations, speed: speed)
        let frames = counts.reduce(0, +)
        guard frames <= ParadeeConstants.maxFrames else {
            throw ParadeeError.durationOverflow(frames: frames, maxFrames: ParadeeConstants.maxFrames)
        }

        let en = expand(d, channels: ParadeeConstants.prosodyChannels, counts: counts)
        let asr = expand(asrTok, channels: ParadeeConstants.textChannels, counts: counts)
        var noise = [Float](repeating: 0, count: frames * ParadeeConstants.samplesPerFrame)
        var rng = InflectNoise(seed: noiseSeed)
        rng.fill(&noise)

        let acousticOut = try predict(
            acoustic,
            [
                "en": try multiArray(en, shape: [1, ParadeeConstants.prosodyChannels, frames]),
                "asr": try multiArray(asr, shape: [1, ParadeeConstants.textChannels, frames]),
                "noise": try multiArray(noise, shape: [1, 1, noise.count]),
            ])
        return try readChannelsByTime(acousticOut, "audio", channels: 1, time: noise.count)
    }

    // MARK: - Host steps

    /// `max(1, round(duration / speed))` per token, rounding half to even like
    /// `torch.round` in the upstream graph.
    static func frameCounts(durations: [Float], speed: Float) -> [Int] {
        durations.map { max(1, Int(($0 / speed).rounded(.toNearestOrEven))) }
    }

    /// Repeat column `i` of a row-major `[channels, T]` matrix `counts[i]`
    /// times → `[channels, sum(counts)]` (the upstream `x @ alignment`).
    static func expand(_ source: [Float], channels: Int, counts: [Int]) -> [Float] {
        let tokens = counts.count
        let frames = counts.reduce(0, +)
        var out = [Float](repeating: 0, count: channels * frames)
        source.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for c in 0..<channels {
                    var f = c * frames
                    let row = c * tokens
                    for i in 0..<tokens {
                        let v = src[row + i]
                        for _ in 0..<counts[i] {
                            dst[f] = v
                            f += 1
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: - CoreML glue

    private static func predict(_ model: MLModel, _ inputs: [String: MLMultiArray]) throws -> MLFeatureProvider {
        do {
            let provider = try MLDictionaryFeatureProvider(
                dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
            return try model.prediction(from: provider)
        } catch {
            throw ParadeeError.predictionFailed("\(error)")
        }
    }

    private static func multiArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: values.count) }
        return arr
    }

    private static func multiArray(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
        let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        values.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: values.count) }
        return arr
    }

    /// Read a float32 output shaped `[1, channels, time]` or `[1, time]` into a
    /// row-major `[channels, time]` buffer, honouring the array's strides
    /// (dynamic-shape outputs are not guaranteed to be densely packed).
    private static func readChannelsByTime(
        _ provider: MLFeatureProvider, _ name: String, channels: Int, time: Int
    ) throws -> [Float] {
        guard let arr = provider.featureValue(for: name)?.multiArrayValue else {
            throw ParadeeError.predictionFailed("missing output '\(name)'")
        }
        guard arr.dataType == .float32 else {
            throw ParadeeError.predictionFailed("output '\(name)' is not float32")
        }
        let shape = arr.shape.map(\.intValue)
        let strides = arr.strides.map(\.intValue)
        guard shape.last == time, shape.reduce(1, *) == channels * time else {
            throw ParadeeError.predictionFailed("output '\(name)' has shape \(shape), expected \(channels)×\(time)")
        }
        let timeStride = strides[strides.count - 1]
        let channelStride = shape.count >= 2 ? strides[strides.count - 2] : 0
        let src = arr.dataPointer.bindMemory(to: Float.self, capacity: 1)
        var out = [Float](repeating: 0, count: channels * time)
        for c in 0..<channels {
            for t in 0..<time {
                out[c * time + t] = src[c * channelStride + t * timeStride]
            }
        }
        return out
    }
}
