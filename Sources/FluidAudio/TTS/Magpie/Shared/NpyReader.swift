import Foundation

/// Minimal NumPy `.npy` (format 1.0) loader.
///
/// Magpie ships its tensor constants (speaker embeddings, audio codebook embeddings,
/// local-transformer weights) as `.npy` files. We only need to read them once at
/// load time into a flat `[Float]` in fp32, so the reader supports exactly the
/// dtypes the Python exporter emits: `<f2` (fp16), `<f4` (fp32), and `<i4` (int32).
///
/// The NPY format spec is trivial: magic + version + header (Python literal dict)
/// + raw little-endian data in C-order. We do not support Fortran-order or
/// structured dtypes — they would be hidden bugs in the exporter, not features.
public enum NpyReader {

    public enum DType {
        case float16
        case float32
        case int32

        public var bytesPerElement: Int {
            switch self {
            case .float16: return 2
            case .float32: return 4
            case .int32: return 4
            }
        }
    }

    public struct Array {
        public let shape: [Int]
        public let dtype: DType
        public let data: [Float]  // always converted to fp32 for ease of consumption

        public var count: Int { data.count }

        public func assertShape(_ expected: [Int], label: String) throws {
            if shape != expected {
                throw MagpieError.invalidNpyFile(
                    path: label,
                    reason: "expected shape \(expected), got \(shape)"
                )
            }
        }
    }

    public static func read(from url: URL) throws -> Array {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        return try parse(data: data, sourceLabel: url.lastPathComponent)
    }

    public static func parse(data: Data, sourceLabel: String) throws -> Array {
        guard data.count >= 10 else {
            throw MagpieError.invalidNpyFile(path: sourceLabel, reason: "file too small")
        }

        // Magic: \x93NUMPY
        let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]
        for (i, expected) in magic.enumerated() where data[i] != expected {
            throw MagpieError.invalidNpyFile(path: sourceLabel, reason: "bad magic byte \(i)")
        }

        let major = data[6]
        let minor = data[7]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            _ = minor
            let low = Int(data[8])
            let high = Int(data[9])
            headerLen = low | (high << 8)
            headerStart = 10
        } else if major == 2 || major == 3 {
            guard data.count >= 12 else {
                throw MagpieError.invalidNpyFile(path: sourceLabel, reason: "truncated v2 header")
            }
            let b0 = Int(data[8])
            let b1 = Int(data[9])
            let b2 = Int(data[10])
            let b3 = Int(data[11])
            headerLen = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            headerStart = 12
        } else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "unsupported NPY version \(major).\(minor)")
        }

        let headerEnd = headerStart + headerLen
        guard headerEnd <= data.count else {
            throw MagpieError.invalidNpyFile(path: sourceLabel, reason: "header out of range")
        }

        guard let header = String(data: data.subdata(in: headerStart..<headerEnd), encoding: .ascii)
        else {
            throw MagpieError.invalidNpyFile(path: sourceLabel, reason: "non-ASCII header")
        }

        let (dtype, shape, fortranOrder) = try parseHeader(header, sourceLabel: sourceLabel)
        if fortranOrder {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "Fortran-order arrays are not supported")
        }

        let elementCount = shape.reduce(1, *)
        let payloadBytes = elementCount * dtype.bytesPerElement
        let payloadStart = headerEnd
        guard payloadStart + payloadBytes == data.count else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel,
                reason: "payload size mismatch (expected \(payloadBytes), file has \(data.count - payloadStart))"
            )
        }

        let floats = try convertToFloat32(
            data: data, offset: payloadStart, count: elementCount, dtype: dtype,
            sourceLabel: sourceLabel)
        return Array(shape: shape, dtype: dtype, data: floats)
    }

    // MARK: - Header parsing

    private static func parseHeader(
        _ header: String, sourceLabel: String
    ) throws -> (
        DType, [Int], Bool
    ) {
        // Header is a Python dict literal, padded with spaces and terminated by '\n'.
        // Example: {'descr': '<f4', 'fortran_order': False, 'shape': (256, 768), }
        let dtype = try extractString(key: "descr", in: header, sourceLabel: sourceLabel)
        let fortran = try extractBool(key: "fortran_order", in: header, sourceLabel: sourceLabel)
        let shape = try extractShape(in: header, sourceLabel: sourceLabel)

        let parsedDtype: DType
        switch dtype {
        case "<f2", "|f2", "=f2":
            parsedDtype = .float16
        case "<f4", "|f4", "=f4":
            parsedDtype = .float32
        case "<i4", "|i4", "=i4":
            parsedDtype = .int32
        default:
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "unsupported dtype '\(dtype)'")
        }
        return (parsedDtype, shape, fortran)
    }

    private static func extractString(
        key: String, in header: String, sourceLabel: String
    ) throws
        -> String
    {
        guard let range = header.range(of: "'\(key)'") else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "missing header key '\(key)'")
        }
        let rest = header[range.upperBound...]
        guard let openQuote = rest.firstIndex(of: "'") else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "missing value for '\(key)'")
        }
        let afterOpen = rest.index(after: openQuote)
        guard let closeQuote = rest[afterOpen...].firstIndex(of: "'") else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "unterminated value for '\(key)'")
        }
        return String(rest[afterOpen..<closeQuote])
    }

    private static func extractBool(
        key: String, in header: String, sourceLabel: String
    ) throws
        -> Bool
    {
        guard let range = header.range(of: "'\(key)'") else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "missing header key '\(key)'")
        }
        let rest = header[range.upperBound...]
        if rest.range(of: "True") != nil,
            let trueIdx = rest.range(of: "True")?.lowerBound,
            let falseIdx = rest.range(of: "False")?.lowerBound
        {
            return trueIdx < falseIdx
        }
        if rest.range(of: "True") != nil { return true }
        if rest.range(of: "False") != nil { return false }
        throw MagpieError.invalidNpyFile(
            path: sourceLabel, reason: "missing bool value for '\(key)'")
    }

    private static func extractShape(in header: String, sourceLabel: String) throws -> [Int] {
        guard let shapeRange = header.range(of: "'shape'") else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "missing 'shape' key")
        }
        let rest = header[shapeRange.upperBound...]
        guard let openIdx = rest.firstIndex(of: "(") else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "missing '(' in shape")
        }
        let afterOpen = rest.index(after: openIdx)
        guard let closeIdx = rest[afterOpen...].firstIndex(of: ")") else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "missing ')' in shape")
        }
        let inside = String(rest[afterOpen..<closeIdx])
        // Handles "(N,)" and "(N, M)" and "(N, M, K, )"
        let dims = inside.split(separator: ",").compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
        if dims.isEmpty {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "could not parse shape '\(inside)'")
        }
        return dims
    }

    // MARK: - Dtype conversion

    private static func convertToFloat32(
        data: Data, offset: Int, count: Int, dtype: DType, sourceLabel: String
    ) throws -> [Float] {
        let payloadRange = offset..<(offset + count * dtype.bytesPerElement)
        let slice = data.subdata(in: payloadRange)

        switch dtype {
        case .float32:
            return slice.withUnsafeBytes { raw -> [Float] in
                let ptr = raw.bindMemory(to: Float.self)
                return Swift.Array<Float>(ptr)
            }
        case .float16:
            return slice.withUnsafeBytes { raw -> [Float] in
                let ptr = raw.bindMemory(to: UInt16.self)
                return ptr.map { Self.float16ToFloat32(bits: $0) }
            }
        case .int32:
            return slice.withUnsafeBytes { raw -> [Float] in
                let ptr = raw.bindMemory(to: Int32.self)
                return ptr.map { Float($0) }
            }
        }
    }

    /// Convert IEEE-754 binary16 bits to Float32. Pure Swift (no Accelerate
    /// dependency) so tests can run without Darwin-specific guards.
    @inline(__always)
    static func float16ToFloat32(bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exp = UInt32((bits & 0x7C00) >> 10)
        let mant = UInt32(bits & 0x03FF)
        var result: UInt32

        if exp == 0 {
            if mant == 0 {
                result = sign
            } else {
                // Subnormal: normalize.
                var e: UInt32 = 127 - 15 + 1
                var m = mant
                while (m & 0x0400) == 0 {
                    m <<= 1
                    e -= 1
                }
                m &= 0x03FF
                result = sign | (e << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            // Inf / NaN
            result = sign | 0x7F80_0000 | (mant << 13)
        } else {
            let newExp = UInt32(Int(exp) - 15 + 127)
            result = sign | (newExp << 23) | (mant << 13)
        }

        return Float(bitPattern: result)
    }
}
