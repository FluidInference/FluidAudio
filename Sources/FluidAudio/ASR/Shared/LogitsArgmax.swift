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
            // Address fp16 storage as raw bit patterns: Swift's `Float16` is
            // unavailable on macOS x86_64.
            let src = logits.dataPointer.assumingMemoryBound(to: UInt16.self)
            var buf = [Float](repeating: 0, count: count)
            buf.withUnsafeMutableBufferPointer { dst in
                Float16Conversion.toFloat32(src: src, dst: dst.baseAddress!, count: count)
                run(dst.baseAddress!)
            }
        }
        return ids
    }

    /// The first `count` rows as full-vocabulary fp32 vectors (row padding dropped).
    static func leadingRows(logits: MLMultiArray, count: Int) -> [[Float]] {
        let rows = min(count, logits.shape[1].intValue)
        let vocab = logits.shape[2].intValue
        let frameStride = logits.strides[1].intValue
        guard rows > 0 else { return [] }

        func copy(_ p: UnsafePointer<Float>) -> [[Float]] {
            (0..<rows).map { t in Array(UnsafeBufferPointer(start: p + t * frameStride, count: vocab)) }
        }

        if logits.dataType == .float32 {
            return copy(logits.dataPointer.assumingMemoryBound(to: Float32.self))
        }
        let total = rows * frameStride
        let src = logits.dataPointer.assumingMemoryBound(to: UInt16.self)
        var buf = [Float](repeating: 0, count: total)
        return buf.withUnsafeMutableBufferPointer { dst in
            guard let base = dst.baseAddress else { return [] }
            Float16Conversion.toFloat32(src: src, dst: base, count: total)
            return copy(UnsafePointer(base))
        }
    }
}
