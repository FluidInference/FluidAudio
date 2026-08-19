import Accelerate
@preconcurrency import CoreML

/// Greedy per-frame argmax over `[1, frames, vocab]` logits, shared by the
/// SenseVoice CTC and Paraformer decoders.
///
/// Uses `vDSP_maxvi` (SIMD) instead of per-element `MLMultiArray` subscript
/// reads — the boxed `NSNumber` path is several hundred ms for ~1M elements.
/// Reads use the tensor's real row stride (CoreML pads rows for ANE alignment,
/// e.g. 8404 -> 8408) and scan only `vocab` elements per row, so the padding
/// slots are never consulted. fp16 logits are widened to fp32 in a single
/// vImage pass first.
enum LogitsArgmax {

    /// Returns the argmax token id for each of the first `frames` rows.
    static func argmaxPerFrame(logits: MLMultiArray, frames: Int) -> [Int] {
        let vocab = logits.shape[2].intValue
        let frameStride = logits.strides[1].intValue
        var ids: [Int] = []
        ids.reserveCapacity(frames)

        func run(_ p: UnsafePointer<Float>) {
            for t in 0..<frames {
                var bestVal: Float = 0
                var bestIdx = vDSP_Length(0)
                vDSP_maxvi(p + t * frameStride, 1, &bestVal, &bestIdx, vDSP_Length(vocab))
                ids.append(Int(bestIdx))
            }
        }

        if logits.dataType == .float32 {
            run(logits.dataPointer.assumingMemoryBound(to: Float32.self))
        } else {
            let count = frames * frameStride
            let src = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            var buf = [Float](repeating: 0, count: count)
            // The vImage buffers must not outlive the pointers they wrap, so the
            // convert + argmax both run inside the withUnsafe scope.
            buf.withUnsafeMutableBufferPointer { dst in
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src),
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float16>.size)
                var dstBuf = vImage_Buffer(
                    data: dst.baseAddress!,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.size)
                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                run(dst.baseAddress!)
            }
        }
        return ids
    }
}
