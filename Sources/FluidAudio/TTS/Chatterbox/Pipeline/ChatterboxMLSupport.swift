@preconcurrency import CoreML
import Foundation

/// CoreML plumbing shared by the Multilingual and Nano synthesizers.
enum ChatterboxMLSupport {

    /// Copy an MLMultiArray into a dense row-major `[Float]`, honoring the
    /// array's strides and dtype. GPU-backed CoreML outputs routinely arrive
    /// as fp16 IOSurfaces with padded row strides (e.g. a [1, 80, 1000] mel
    /// with strides [80640, 1008, 1]) even when the model declares fp32
    /// outputs — a naive contiguous read smears every row.
    static func floatBuffer(_ array: MLMultiArray) throws -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let count = shape.reduce(1, *)
        let rowLen = shape.last ?? 1
        guard strides.last == 1 else {
            throw ChatterboxError.processingFailed("innermost stride \(strides) unsupported")
        }
        let rows = count / max(rowLen, 1)
        let outerShape = shape.dropLast()
        let outerStrides = strides.dropLast()

        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBufferPointer { dst in
            let dstBase = dst.baseAddress!
            for row in 0..<rows {
                // Mixed-radix decode of the outer index → source offset.
                var remainder = row
                var srcOffset = 0
                for (dim, stride) in zip(outerShape, outerStrides).reversed() {
                    srcOffset += (remainder % dim) * stride
                    remainder /= dim
                }
                switch array.dataType {
                case .float32:
                    let src = array.dataPointer.assumingMemoryBound(to: Float.self)
                    dstBase.advanced(by: row * rowLen)
                        .update(from: src.advanced(by: srcOffset), count: rowLen)
                case .float16:
                    let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
                    Float16Conversion.toFloat32(
                        src: src.advanced(by: srcOffset),
                        dst: dstBase.advanced(by: row * rowLen), count: rowLen)
                default:
                    break
                }
            }
        }
        switch array.dataType {
        case .float32, .float16:
            return out
        default:
            throw ChatterboxError.processingFailed(
                "unexpected MLMultiArray dataType \(array.dataType.rawValue)")
        }
    }

    /// Copy prefill KV (`[layers, batch, heads, maxContext, headDim]`,
    /// possibly strided fp16) into the decode model's per-layer fp16
    /// `MLState` buffers (`kv_k_<i>` / `kv_v_<i>`).
    @available(macOS 15.0, iOS 18.0, *)
    static func seedState(
        _ state: MLState, kvK: MLMultiArray, kvV: MLMultiArray,
        layerCount: Int, layerElements: Int
    ) throws {
        let kFloats = try floatBuffer(kvK)
        let vFloats = try floatBuffer(kvV)

        for layer in 0..<layerCount {
            for (name, src) in [("kv_k_\(layer)", kFloats), ("kv_v_\(layer)", vFloats)] {
                src.withUnsafeBufferPointer { srcBuf in
                    let base = srcBuf.baseAddress!.advanced(by: layer * layerElements)
                    state.withMultiArray(for: name) { array in
                        array.withUnsafeMutableBytes { rawBuffer, _ in
                            guard let dstBase = rawBuffer.baseAddress else { return }
                            // Address fp16 storage as raw bit patterns: Swift's
                            // `Float16` is unavailable on macOS x86_64.
                            let dst = dstBase.assumingMemoryBound(to: UInt16.self)
                            Float16Conversion.fromFloat32(
                                src: base, dst: dst, count: layerElements)
                        }
                    }
                }
            }
        }
    }
}
