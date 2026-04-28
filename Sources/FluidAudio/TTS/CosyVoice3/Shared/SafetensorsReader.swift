import Foundation

/// Minimal zero-dependency safetensors parser.
///
/// File format (little-endian):
/// - `u64` header length N
/// - N bytes of UTF-8 JSON: `{ "<name>": {"dtype": "...", "shape": [...], "data_offsets": [start, end]}, ... }`
/// - raw tensor payload (referenced by offsets above)
///
/// Used for Phase 1 fixture + speech embedding table mmap.
public final class SafetensorsFile {

    public enum DType: String, Sendable {
        case f16 = "F16"
        case bf16 = "BF16"
        case f32 = "F32"
        case f64 = "F64"
        case i8 = "I8"
        case i16 = "I16"
        case i32 = "I32"
        case i64 = "I64"
        case u8 = "U8"
        case u16 = "U16"
        case u32 = "U32"
        case u64 = "U64"
        case bool = "BOOL"

        public var byteSize: Int {
            switch self {
            case .f16, .bf16, .i16, .u16: return 2
            case .f32, .i32, .u32: return 4
            case .f64, .i64, .u64: return 8
            case .i8, .u8, .bool: return 1
            }
        }
    }

    public struct TensorInfo: Sendable {
        public let dtype: DType
        public let shape: [Int]
        public let dataStart: Int  // absolute offset in file
        public let dataEnd: Int
        public var byteCount: Int { dataEnd - dataStart }
    }

    private let data: Data
    private let payloadStart: Int
    public let tensors: [String: TensorInfo]

    public init(url: URL) throws {
        let data = try Data(contentsOf: url, options: [.alwaysMapped])
        guard data.count >= 8 else {
            throw CosyVoice3Error.invalidSafetensors("file smaller than 8 byte header: \(url.path)")
        }
        self.data = data

        let headerLen: UInt64 = data.withUnsafeBytes { buf in
            var v: UInt64 = 0
            memcpy(&v, buf.baseAddress!, 8)
            return UInt64(littleEndian: v)
        }
        let headerEnd = 8 + Int(headerLen)
        guard headerEnd <= data.count else {
            throw CosyVoice3Error.invalidSafetensors(
                "header length \(headerLen) exceeds file size \(data.count)")
        }
        let headerData = data.subdata(in: 8..<headerEnd)
        self.payloadStart = headerEnd

        guard
            let json = try JSONSerialization.jsonObject(with: headerData, options: [])
                as? [String: Any]
        else {
            throw CosyVoice3Error.invalidSafetensors("header is not a JSON object")
        }

        var parsed: [String: TensorInfo] = [:]
        for (name, value) in json where name != "__metadata__" {
            guard
                let entry = value as? [String: Any],
                let dtypeStr = entry["dtype"] as? String,
                let dtype = DType(rawValue: dtypeStr),
                let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Int],
                offsets.count == 2
            else {
                throw CosyVoice3Error.invalidSafetensors("bad entry for tensor \(name)")
            }
            parsed[name] = TensorInfo(
                dtype: dtype,
                shape: shape,
                dataStart: payloadStart + offsets[0],
                dataEnd: payloadStart + offsets[1])
        }
        self.tensors = parsed
    }

    /// Returns the raw bytes slice for a tensor (zero-copy reference into the mmap'd file).
    public func rawBytes(_ name: String) throws -> Data {
        guard let info = tensors[name] else {
            throw CosyVoice3Error.invalidSafetensors("tensor not found: \(name)")
        }
        return data.subdata(in: info.dataStart..<info.dataEnd)
    }

    public func info(_ name: String) throws -> TensorInfo {
        guard let info = tensors[name] else {
            throw CosyVoice3Error.invalidSafetensors("tensor not found: \(name)")
        }
        return info
    }

    // MARK: - Typed accessors (copying)

    public func asFloat32(_ name: String) throws -> [Float] {
        let info = try self.info(name)
        let bytes = try rawBytes(name)
        switch info.dtype {
        case .f32:
            return bytes.withUnsafeBytes { buf -> [Float] in
                let count = buf.count / 4
                let ptr = buf.bindMemory(to: Float.self)
                return Array(UnsafeBufferPointer(start: ptr.baseAddress, count: count))
            }
        case .f64:
            return bytes.withUnsafeBytes { buf -> [Float] in
                let count = buf.count / 8
                let ptr = buf.bindMemory(to: Double.self)
                return (0..<count).map { Float(ptr[$0]) }
            }
        default:
            throw CosyVoice3Error.invalidSafetensors(
                "asFloat32 unsupported for dtype \(info.dtype.rawValue)")
        }
    }

    public func asInt32(_ name: String) throws -> [Int32] {
        let info = try self.info(name)
        let bytes = try rawBytes(name)
        switch info.dtype {
        case .i32:
            return bytes.withUnsafeBytes { buf -> [Int32] in
                let count = buf.count / 4
                let ptr = buf.bindMemory(to: Int32.self)
                return Array(UnsafeBufferPointer(start: ptr.baseAddress, count: count))
            }
        case .i64:
            return bytes.withUnsafeBytes { buf -> [Int32] in
                let count = buf.count / 8
                let ptr = buf.bindMemory(to: Int64.self)
                return (0..<count).map { Int32(truncatingIfNeeded: ptr[$0]) }
            }
        default:
            throw CosyVoice3Error.invalidSafetensors(
                "asInt32 unsupported for dtype \(info.dtype.rawValue)")
        }
    }

    /// Scalar integer (shape [] or [1]), for tensors like `seed` or `t_pre`.
    public func asInt(_ name: String) throws -> Int {
        let values = try asInt32(name)
        guard let first = values.first else {
            throw CosyVoice3Error.invalidSafetensors("tensor \(name) is empty")
        }
        return Int(first)
    }
}
