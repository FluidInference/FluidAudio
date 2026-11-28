#if TTS
@preconcurrency import CoreML
import Foundation

/// Native equivalent of the `KokoroAlignment` stage: each token's features repeated over
/// the frames its duration gives it.
///
/// The Core ML graph builds a 0/1 alignment mask from the durations and multiplies `d` and
/// `t_en` by it. Every frame belongs to exactly one token, so each product is a copy of one
/// fp16 value, and copying the values gives the same bits without the graph (#1172). Under
/// `.cpuAndNeuralEngine` the graph runs on BNNS, which traps (`EXC_BREAKPOINT` in
/// `BNNSGraphContextExecute_v2`) whenever the token count equals the model's default shape
/// of 48: the ~5% of sentences that come to 46 phonemes.
enum KokoroAneV3Alignment {

    /// - Parameters:
    ///   - d: PostAlbert's `d`, fp16 `[1, tokens, channels]`.
    ///   - tEn: PostAlbert's `t_en`, fp16 `[1, textChannels, tokens]`.
    ///   - durations: Frames per token, each zero or more.
    /// - Returns: `en` as fp16 `[1, channels, frames]` and `asr` as fp16 `[1, textChannels, frames]`.
    static func expand(
        d: MLMultiArray, tEn: MLMultiArray, durations: [Int32]
    ) throws -> (en: MLMultiArray, asr: MLMultiArray) {
        let tokens = durations.count
        let dShape = d.shape.map(\.intValue)
        let tShape = tEn.shape.map(\.intValue)
        guard d.dataType == .float16, tEn.dataType == .float16,
            dShape.count == 3, dShape[0] == 1, dShape[1] == tokens,
            tShape.count == 3, tShape[0] == 1, tShape[2] == tokens,
            durations.allSatisfy({ $0 >= 0 })
        else {
            throw KokoroAneV3Error.invalidInput("Alignment expects fp16 d and t_en matching the durations")
        }
        let channels = dShape[2]
        let textChannels = tShape[1]
        // The token each frame belongs to.
        var owners: [Int] = []
        owners.reserveCapacity(durations.reduce(0) { $0 + Int($1) })
        for (token, duration) in durations.enumerated() {
            owners.append(contentsOf: repeatElement(token, count: Int(duration)))
        }
        let frames = owners.count
        let en = try KokoroAneArrays.makeArray(shape: [1, channels, frames], dataType: .float16)
        let asr = try KokoroAneArrays.makeArray(shape: [1, textChannels, frames], dataType: .float16)
        let dStrides = d.strides.map(\.intValue)
        let tStrides = tEn.strides.map(\.intValue)
        let dSource = d.dataPointer.assumingMemoryBound(to: UInt16.self)
        let tSource = tEn.dataPointer.assumingMemoryBound(to: UInt16.self)
        let enValues = en.dataPointer.assumingMemoryBound(to: UInt16.self)
        let asrValues = asr.dataPointer.assumingMemoryBound(to: UInt16.self)
        for channel in 0..<channels {
            let row = enValues + channel * frames
            for (frame, token) in owners.enumerated() {
                row[frame] = dSource[token * dStrides[1] + channel * dStrides[2]]
            }
        }
        for channel in 0..<textChannels {
            let row = asrValues + channel * frames
            for (frame, token) in owners.enumerated() {
                row[frame] = tSource[channel * tStrides[1] + token * tStrides[2]]
            }
        }
        return (en, asr)
    }
}
#endif
