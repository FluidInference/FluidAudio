import Accelerate

/// Portable fp16 ↔ fp32 buffer conversions.
///
/// Swift's `Float16` is unavailable on macOS x86_64, so half-precision
/// buffers are addressed as `UInt16` bit patterns and converted with
/// vImage's Planar16F routines, which compile on every architecture.
enum Float16Conversion {

    /// Widen `count` half-precision values (as raw bit patterns) to Float32.
    static func toFloat32(
        src: UnsafePointer<UInt16>, dst: UnsafeMutablePointer<Float>, count: Int
    ) {
        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src), height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<UInt16>.stride)
        var dstBuf = vImage_Buffer(
            data: dst, height: 1, width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.stride)
        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
    }

    /// Narrow `count` Float32 values to half precision (as raw bit patterns).
    static func fromFloat32(
        src: UnsafePointer<Float>, dst: UnsafeMutablePointer<UInt16>, count: Int
    ) {
        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src), height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.stride)
        var dstBuf = vImage_Buffer(
            data: dst, height: 1, width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<UInt16>.stride)
        vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
    }
}
