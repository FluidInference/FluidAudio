import Compression
import FluidAudio
import Foundation

/// Minimal NumPy `.npz` (ZIP-of-`.npy`) loader.
///
/// `np.savez(...)` writes a ZIP archive whose members are `<name>.npy` files —
/// each member is a regular NPY blob that `NpyReader` already handles. We only
/// need a tiny ZIP parser that can locate members and decompress them.
///
/// Supported compression methods:
///   - 0 (STORE)   — raw bytes, used by `np.savez` (default).
///   - 8 (DEFLATE) — raw deflate, used by `np.savez_compressed`.
///
/// Multi-disk archives, encryption, and ZIP64 are not supported (NumPy never
/// emits them for fixture files in our size range).
public enum MagpieNpzReader {

    /// Read the entire NPZ archive into a name → NpyReader.Array map.
    /// Names are stripped of the trailing `.npy` (so `encoder_output.npy`
    /// surfaces as `encoder_output`).
    public static func read(from url: URL) throws -> [String: NpyReader.Array] {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        return try parse(archive: data, sourceLabel: url.lastPathComponent)
    }

    public static func parse(archive: Data, sourceLabel: String) throws -> [String: NpyReader.Array] {
        let entries = try locateEntries(in: archive, sourceLabel: sourceLabel)
        var out: [String: NpyReader.Array] = [:]
        out.reserveCapacity(entries.count)
        for entry in entries {
            let payload = try extractPayload(entry: entry, archive: archive, sourceLabel: sourceLabel)
            let parsed = try NpyReader.parse(data: payload, sourceLabel: entry.name)
            let key =
                entry.name.hasSuffix(".npy")
                ? String(entry.name.dropLast(4))
                : entry.name
            out[key] = parsed
        }
        return out
    }

    // MARK: - ZIP parsing

    private struct Entry {
        let name: String
        let compressionMethod: UInt16
        let compressedSize: Int
        let uncompressedSize: Int
        let localHeaderOffset: Int
    }

    private static func locateEntries(
        in data: Data, sourceLabel: String
    ) throws -> [Entry] {
        // Scan backwards for End Of Central Directory (EOCD) signature 0x06054b50.
        // EOCD is 22 bytes + variable-length comment (typically 0).
        let eocdSig: UInt32 = 0x0605_4b50
        let minEocd = 22
        guard data.count >= minEocd else {
            throw MagpieError.invalidNpyFile(path: sourceLabel, reason: "file too small to be a ZIP")
        }
        let scanStart = max(0, data.count - minEocd - 0xFFFF)
        var eocdOffset: Int? = nil
        var i = data.count - minEocd
        while i >= scanStart {
            if readU32(data, at: i) == eocdSig {
                eocdOffset = i
                break
            }
            i -= 1
        }
        guard let eocd = eocdOffset else {
            throw MagpieError.invalidNpyFile(path: sourceLabel, reason: "EOCD record not found")
        }

        let totalEntries = Int(readU16(data, at: eocd + 10))
        let cdSize = Int(readU32(data, at: eocd + 12))
        let cdOffset = Int(readU32(data, at: eocd + 16))
        guard cdOffset + cdSize <= data.count else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "central directory out of bounds")
        }

        var entries: [Entry] = []
        entries.reserveCapacity(totalEntries)
        var cursor = cdOffset
        let cdSig: UInt32 = 0x0201_4b50
        for _ in 0..<totalEntries {
            guard cursor + 46 <= cdOffset + cdSize else {
                throw MagpieError.invalidNpyFile(
                    path: sourceLabel, reason: "truncated central directory entry")
            }
            guard readU32(data, at: cursor) == cdSig else {
                throw MagpieError.invalidNpyFile(
                    path: sourceLabel, reason: "bad central directory signature")
            }
            let compressionMethod = readU16(data, at: cursor + 10)
            let compressedSize = Int(readU32(data, at: cursor + 20))
            let uncompressedSize = Int(readU32(data, at: cursor + 24))
            let nameLen = Int(readU16(data, at: cursor + 28))
            let extraLen = Int(readU16(data, at: cursor + 30))
            let commentLen = Int(readU16(data, at: cursor + 32))
            let localHeaderOffset = Int(readU32(data, at: cursor + 42))
            let nameStart = cursor + 46
            guard nameStart + nameLen <= data.count else {
                throw MagpieError.invalidNpyFile(
                    path: sourceLabel, reason: "filename out of range")
            }
            guard
                let name = String(
                    data: data.subdata(in: nameStart..<(nameStart + nameLen)), encoding: .utf8)
            else {
                throw MagpieError.invalidNpyFile(
                    path: sourceLabel, reason: "non-UTF8 filename in central directory")
            }
            entries.append(
                Entry(
                    name: name, compressionMethod: compressionMethod,
                    compressedSize: compressedSize, uncompressedSize: uncompressedSize,
                    localHeaderOffset: localHeaderOffset))
            cursor = nameStart + nameLen + extraLen + commentLen
        }
        return entries
    }

    private static func extractPayload(
        entry: Entry, archive: Data, sourceLabel: String
    ) throws -> Data {
        // Local file header is 30 bytes + filename + extra; payload immediately follows.
        let lfhSig: UInt32 = 0x0403_4b50
        let off = entry.localHeaderOffset
        guard off + 30 <= archive.count else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "local header truncated for \(entry.name)")
        }
        guard readU32(archive, at: off) == lfhSig else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "bad local header signature for \(entry.name)")
        }
        let lfhNameLen = Int(readU16(archive, at: off + 26))
        let lfhExtraLen = Int(readU16(archive, at: off + 28))
        let payloadStart = off + 30 + lfhNameLen + lfhExtraLen
        guard payloadStart + entry.compressedSize <= archive.count else {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel, reason: "payload truncated for \(entry.name)")
        }
        let compressed = archive.subdata(in: payloadStart..<(payloadStart + entry.compressedSize))

        switch entry.compressionMethod {
        case 0:
            return compressed
        case 8:
            return try inflateRawDeflate(
                compressed: compressed, expectedSize: entry.uncompressedSize,
                sourceLabel: "\(sourceLabel):\(entry.name)")
        default:
            throw MagpieError.invalidNpyFile(
                path: sourceLabel,
                reason: "unsupported ZIP compression method \(entry.compressionMethod) for \(entry.name)"
            )
        }
    }

    // MARK: - Raw DEFLATE inflate via Compression framework

    private static func inflateRawDeflate(
        compressed: Data, expectedSize: Int, sourceLabel: String
    ) throws -> Data {
        // ZIP's method 8 is raw deflate (no zlib wrapper). On Apple,
        // COMPRESSION_ZLIB is raw deflate per docs.
        var dst = [UInt8](repeating: 0, count: max(expectedSize, 1))
        let written = compressed.withUnsafeBytes { srcRaw -> Int in
            guard let src = srcRaw.bindMemory(to: UInt8.self).baseAddress else { return 0 }
            return dst.withUnsafeMutableBufferPointer { dstBuf -> Int in
                guard let dstBase = dstBuf.baseAddress else { return 0 }
                return compression_decode_buffer(
                    dstBase, expectedSize,
                    src, compressed.count,
                    nil, COMPRESSION_ZLIB)
            }
        }
        if written != expectedSize {
            throw MagpieError.invalidNpyFile(
                path: sourceLabel,
                reason: "DEFLATE inflate produced \(written) bytes, expected \(expectedSize)")
        }
        return Data(dst[0..<written])
    }

    // MARK: - Little-endian readers

    private static func readU16(_ data: Data, at offset: Int) -> UInt16 {
        return UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
    }

    private static func readU32(_ data: Data, at offset: Int) -> UInt32 {
        return UInt32(data[offset])
            | (UInt32(data[offset + 1]) << 8)
            | (UInt32(data[offset + 2]) << 16)
            | (UInt32(data[offset + 3]) << 24)
    }
}
