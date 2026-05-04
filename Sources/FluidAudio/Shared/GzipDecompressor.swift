import Compression
import Foundation

/// Minimal RFC 1952 gzip → raw bytes decompressor.
///
/// Apple's `Compression` framework exposes raw DEFLATE (RFC 1951) via
/// `COMPRESSION_ZLIB`, but does not parse the gzip wrapper. This helper
/// strips the gzip header (handling FNAME/FCOMMENT/FEXTRA/FHCRC flag
/// bits) plus the 8-byte CRC32+ISIZE trailer, then runs the inner
/// DEFLATE stream through `compression_decode_buffer`.
///
/// Used by KokoroAne Mandarin G2P to load `pinyin_*.bin.gz` and
/// `jieba.bin.gz` shipped on HuggingFace. Single-shot decode — the whole
/// compressed payload must fit in memory (the largest file is ~3.5 MB
/// compressed → ~9 MB uncompressed, well within budget).
public enum GzipDecompressor {

    public enum Error: Swift.Error, LocalizedError {
        case invalidMagic
        case unsupportedMethod(UInt8)
        case truncatedHeader
        case decodeFailed
        case sizeMismatch(declared: UInt32, actual: Int)

        public var errorDescription: String? {
            switch self {
            case .invalidMagic: return "Not a gzip stream (missing 1f 8b magic)"
            case .unsupportedMethod(let m):
                return "Gzip compression method \(m) is not supported (only deflate=8)"
            case .truncatedHeader: return "Gzip header is truncated"
            case .decodeFailed: return "Failed to inflate gzip payload"
            case .sizeMismatch(let d, let a):
                return "Decompressed size \(a) does not match gzip ISIZE field \(d)"
            }
        }
    }

    /// Decompress a complete gzip stream into raw bytes. Verifies the
    /// trailing ISIZE field (uncompressed size mod 2³²) matches the
    /// produced output size — a cheap sanity check that catches truncated
    /// downloads.
    public static func decompress(_ data: Data) throws -> Data {
        guard data.count >= 18 else { throw Error.truncatedHeader }
        guard data[0] == 0x1f, data[1] == 0x8b else { throw Error.invalidMagic }
        guard data[2] == 0x08 else { throw Error.unsupportedMethod(data[2]) }

        let flg = data[3]
        var pos = 10  // fixed header

        // FEXTRA: 2-byte XLEN + XLEN bytes extra
        if flg & 0x04 != 0 {
            guard pos + 2 <= data.count else { throw Error.truncatedHeader }
            let xlen = Int(data[pos]) | (Int(data[pos + 1]) << 8)
            pos += 2 + xlen
        }
        // FNAME: null-terminated string
        if flg & 0x08 != 0 {
            pos = try advancePastNull(data, from: pos)
        }
        // FCOMMENT: null-terminated string
        if flg & 0x10 != 0 {
            pos = try advancePastNull(data, from: pos)
        }
        // FHCRC: 2-byte header CRC16
        if flg & 0x02 != 0 {
            pos += 2
        }

        guard pos + 8 <= data.count else { throw Error.truncatedHeader }
        let payloadEnd = data.count - 8  // strip CRC32 (4) + ISIZE (4)
        guard pos < payloadEnd else { throw Error.truncatedHeader }

        // ISIZE is little-endian uint32 mod 2^32 of the original size.
        let isizeOffset = data.count - 4
        let isize: UInt32 =
            UInt32(data[isizeOffset])
            | (UInt32(data[isizeOffset + 1]) << 8)
            | (UInt32(data[isizeOffset + 2]) << 16)
            | (UInt32(data[isizeOffset + 3]) << 24)

        // Worst-case raw DEFLATE expansion is ~1032×, but for our fixture
        // bundle compressed size already brackets the upper bound (≤ 32 MB
        // gives ≤ 1 GB output, way more than we'd ever ship). Use ISIZE as
        // the buffer hint when it looks plausible (< 64 MB), otherwise fall
        // back to a 64 MB cap to absorb pathological inputs.
        let hinted = Int(isize)
        let bufferSize = (hinted > 0 && hinted < 64 * 1024 * 1024) ? hinted : 64 * 1024 * 1024

        let payloadLength = payloadEnd - pos
        let result = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Data? in
            guard let base = raw.baseAddress else { return nil }
            let src = base.advanced(by: pos).assumingMemoryBound(to: UInt8.self)
            let dst = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
            defer { dst.deallocate() }
            let written = compression_decode_buffer(
                dst, bufferSize, src, payloadLength, nil, COMPRESSION_ZLIB)
            guard written > 0 else { return nil }
            return Data(bytes: dst, count: written)
        }

        guard let out = result else { throw Error.decodeFailed }
        // ISIZE is mod 2^32; only verify when the declared size fits in our
        // produced buffer (always true here since both sides are ≤ 4 GB).
        if Int(isize) != out.count {
            throw Error.sizeMismatch(declared: isize, actual: out.count)
        }
        return out
    }

    private static func advancePastNull(_ data: Data, from start: Int) throws -> Int {
        var i = start
        while i < data.count {
            if data[i] == 0 { return i + 1 }
            i += 1
        }
        throw Error.truncatedHeader
    }
}
