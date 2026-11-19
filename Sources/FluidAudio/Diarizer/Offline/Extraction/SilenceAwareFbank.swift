#if Diarizer
import Foundation

/// Re-centers FBANK features on the frames that hold signal (#981).
///
/// The bundled `FBank.mlmodelc` ends with `features = mel - reduce_mean(mel, frames)`.
/// A frame of exact digital silence sits at the log floor, so in a window that is
/// mostly silence (a muted call track, end-of-file padding) that mean is dominated by
/// the floor and every speech frame is shifted by it: the window's embedding loses the
/// voice. Because the model only subtracts a per-band constant, subtracting the mean of
/// its output over the non-silent frames gives exactly the features of a model whose
/// mean excluded the silent frames. Windows without a silent frame are left untouched.
enum SilenceAwareFbank {
    /// FBANK frame geometry: 25 ms frames every 10 ms at 16 kHz.
    static let frameLength = 400
    static let frameShift = 160

    /// Marks each frame whose samples are all exactly zero.
    static func silentFrames(audio: UnsafeBufferPointer<Float>, frameCount: Int) -> [Bool] {
        var nonZeroBefore = [Int](repeating: 0, count: audio.count + 1)
        for index in 0..<audio.count {
            nonZeroBefore[index + 1] = nonZeroBefore[index] + (audio[index] != 0 ? 1 : 0)
        }
        return (0..<frameCount).map { frame in
            let start = min(frame * frameShift, audio.count)
            let end = min(start + frameLength, audio.count)
            return nonZeroBefore[end] == nonZeroBefore[start]
        }
    }

    /// Subtracts from each band its mean over the non-silent frames.
    ///
    /// - Returns: `false`, leaving `features` untouched, when no frame or every frame is silent.
    @discardableResult
    static func recenter(
        features: UnsafeMutablePointer<Float>,
        bandCount: Int,
        bandStride: Int,
        frameStride: Int,
        silent: [Bool]
    ) -> Bool {
        let voiced = silent.indices.filter { !silent[$0] }
        guard !voiced.isEmpty, voiced.count < silent.count else { return false }
        for band in 0..<bandCount {
            let row = features + band * bandStride
            var sum: Float = 0
            for frame in voiced {
                sum += row[frame * frameStride]
            }
            let mean = sum / Float(voiced.count)
            for frame in silent.indices {
                row[frame * frameStride] -= mean
            }
        }
        return true
    }
}
#endif
