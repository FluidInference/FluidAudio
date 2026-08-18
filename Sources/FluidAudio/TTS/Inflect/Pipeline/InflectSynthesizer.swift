@preconcurrency import CoreML
import Foundation

/// Drives the two-stage Inflect CoreML pipeline for a single chunk:
/// encoder → host duration expansion + prior sampling → bucketed synthesizer.
/// Everything stochastic or dynamically shaped runs here in Swift; both CoreML
/// graphs are deterministic and fixed-shape.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
struct InflectSynthesizer {

    private let store: InflectModelStore
    private let variant: InflectVariant

    init(store: InflectModelStore, variant: InflectVariant) {
        self.store = store
        self.variant = variant
    }

    /// Synthesize one chunk of already-encoded, blank-interspersed token ids.
    /// `tokens.count` must be ≤ `InflectConstants.encoderTokens`.
    func synthesize(
        tokens: [Int32],
        noiseScale: Float,
        speed: Float,
        noiseSeed: UInt64
    ) async throws -> [Float] {
        let tText = InflectConstants.encoderTokens
        let channels = InflectConstants.interChannels(for: variant)
        let n = tokens.count
        guard n > 0, n <= tText else {
            throw InflectError.inputProcessingFailed(
                "token count \(n) out of range (1...\(tText))")
        }

        // --- Encoder ---
        var tokenBuf = [Int32](repeating: InflectSymbols.blankID, count: tText)
        var maskBuf = [Float](repeating: 0, count: tText)
        for i in 0..<n {
            tokenBuf[i] = tokens[i]
            maskBuf[i] = 1
        }
        let encoder = try await store.encoder()
        let encoderOut = try predict(
            model: encoder,
            inputs: [
                "tokens": try multiArray(tokenBuf, shape: [1, tText]),
                "x_mask": try multiArray(maskBuf, shape: [1, 1, tText]),
            ])
        let mP = try floats(encoderOut, "m_p")  // [1, C, tText]
        let logsP = try floats(encoderOut, "logs_p")
        let logw = try floats(encoderOut, "logw")  // [1, 1, tText]

        // --- Host: durations (ceil(exp(logw) / speed)) over the valid tokens ---
        let lengthScale = 1.0 / speed
        var durations = [Int](repeating: 0, count: n)
        var yLen = 0
        for i in 0..<n {
            let frames = Int((Foundation.exp(logw[i]) * lengthScale).rounded(.up))
            let clamped = max(0, frames)
            durations[i] = clamped
            yLen += clamped
        }
        guard yLen >= 1 else {
            throw InflectError.inputProcessingFailed("predicted zero total duration")
        }
        guard yLen <= InflectConstants.maxFrames else {
            throw InflectError.durationOverflow(frames: yLen, maxFrames: InflectConstants.maxFrames)
        }

        let (synthModel, bucket) = try await store.synthesizer(forFrames: yLen)

        // --- Host: expand prior to frame rate + sample z_p (padded to bucket) ---
        var noise = InflectNoise(seed: noiseSeed)
        var zp = [Float](repeating: 0, count: channels * bucket)
        var maskY = [Float](repeating: 0, count: bucket)
        for f in 0..<yLen { maskY[f] = 1 }
        // Walk expanded frames, mapping each back to its source token column.
        var frame = 0
        for i in 0..<n where durations[i] > 0 {
            let col = i
            for _ in 0..<durations[i] {
                for c in 0..<channels {
                    let src = c * tText + col
                    let g = noise.nextGaussian()
                    zp[c * bucket + frame] = mP[src] + g * Foundation.exp(logsP[src]) * noiseScale
                }
                frame += 1
            }
        }

        // --- Synthesizer ---
        let synthOut = try predict(
            model: synthModel,
            inputs: [
                "z_p": try multiArray(zp, shape: [1, channels, bucket]),
                "y_mask": try multiArray(maskY, shape: [1, 1, bucket]),
            ])
        let audioFull = try floats(synthOut, "audio")  // [1, 1, bucket * hop]
        let sampleCount = min(yLen * InflectConstants.hopLength, audioFull.count)
        var audio = Array(audioFull[0..<sampleCount])
        edgeFade(&audio)
        for i in audio.indices { audio[i] = min(1, max(-1, audio[i])) }
        return audio
    }

    // MARK: - Edge fade (matches upstream `edge_fade`)

    private func edgeFade(_ audio: inout [Float]) {
        let frames = min(
            Int((Double(InflectConstants.sampleRate) * InflectConstants.edgeFadeMs / 1000.0).rounded()),
            audio.count / 2)
        guard frames > 0 else { return }
        for i in 0..<frames {
            let ramp = Float(i) / Float(frames)
            audio[i] *= ramp
            audio[audio.count - 1 - i] *= ramp
        }
    }

    // MARK: - CoreML glue

    private func predict(model: MLModel, inputs: [String: MLMultiArray]) throws -> MLFeatureProvider {
        do {
            let provider = try MLDictionaryFeatureProvider(
                dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
            return try model.prediction(from: provider)
        } catch {
            throw InflectError.predictionFailed("\(error)")
        }
    }

    private func multiArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: values.count) }
        return arr
    }

    private func multiArray(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
        let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        values.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: values.count) }
        return arr
    }

    /// Read a named output as `[Float]`, handling fp16 or fp32 backing.
    private func floats(_ provider: MLFeatureProvider, _ name: String) throws -> [Float] {
        guard let arr = provider.featureValue(for: name)?.multiArrayValue else {
            throw InflectError.predictionFailed("missing output '\(name)'")
        }
        let count = arr.count
        var out = [Float](repeating: 0, count: count)
        switch arr.dataType {
        case .float32:
            let src = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            out.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: src, count: count) }
        case .float16:
            let src = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count { out[i] = Float(float16Bits: src[i]) }
        case .double:
            let src = arr.dataPointer.bindMemory(to: Double.self, capacity: count)
            for i in 0..<count { out[i] = Float(src[i]) }
        default:
            throw InflectError.predictionFailed("unsupported dtype for '\(name)'")
        }
        return out
    }
}

extension Float {
    /// Decode an IEEE-754 half stored in a `UInt16`.
    fileprivate init(float16Bits bits: UInt16) {
        let sign = Float((bits >> 15) & 0x1)
        let exponent = Int((bits >> 10) & 0x1F)
        let mantissa = Float(bits & 0x3FF)
        let value: Float
        if exponent == 0 {
            value = mantissa / 1024.0 * Float(pow(2.0, -14.0))
        } else if exponent == 0x1F {
            value = mantissa == 0 ? Float.infinity : Float.nan
        } else {
            value = (1.0 + mantissa / 1024.0) * Float(pow(2.0, Double(exponent - 15)))
        }
        self = (sign == 1 ? -1 : 1) * value
    }
}
