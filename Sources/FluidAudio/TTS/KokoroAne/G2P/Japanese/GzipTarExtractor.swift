import CZlib
import Foundation

/// Minimal streaming reader for the regular-file/directory subset used by
/// the pinned OpenJTalk dictionary archive. It rejects links and paths that
/// could escape the destination directory.
enum GzipTarExtractor {
    enum ExtractionError: LocalizedError {
        case couldNotOpen(URL)
        case corruptArchive(String)
        case unsafePath(String)
        case unsupportedEntry(String)

        var errorDescription: String? {
            switch self {
            case .couldNotOpen(let url): return "Could not open gzip archive at \(url.path)."
            case .corruptArchive(let detail): return "Corrupt tar.gz archive: \(detail)"
            case .unsafePath(let path): return "Unsafe path in tar archive: \(path)"
            case .unsupportedEntry(let type): return "Unsupported tar entry type: \(type)"
            }
        }
    }

    static func extract(archiveURL: URL, to destination: URL) throws {
        guard let stream = gzopen(archiveURL.path, "rb") else {
            throw ExtractionError.couldNotOpen(archiveURL)
        }
        defer { gzclose(stream) }

        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        var zeroBlocks = 0

        while let header = try readExactly(512, from: stream) {
            if header.allSatisfy({ $0 == 0 }) {
                zeroBlocks += 1
                if zeroBlocks == 2 { return }
                continue
            }
            zeroBlocks = 0

            let name = try field(in: header, range: 0..<100)
            let prefix = try field(in: header, range: 345..<500)
            let path = prefix.isEmpty ? name : "\(prefix)/\(name)"
            let relativePath = try validatedRelativePath(path)
            let size = try octal(in: header, range: 124..<136)
            let type = header[156]
            let entryURL = destination.appendingPathComponent(relativePath)

            switch type {
            case 0, 48:  // NUL or "0": regular file
                try FileManager.default.createDirectory(
                    at: entryURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                guard FileManager.default.createFile(atPath: entryURL.path, contents: nil) else {
                    throw ExtractionError.corruptArchive("could not create \(relativePath)")
                }
                let file = try FileHandle(forWritingTo: entryURL)
                do {
                    try copy(size, from: stream, to: file)
                    try file.close()
                } catch {
                    try? file.close()
                    throw error
                }
            case 53:  // "5": directory
                guard size == 0 else {
                    throw ExtractionError.corruptArchive("directory \(path) has non-zero size")
                }
                try FileManager.default.createDirectory(at: entryURL, withIntermediateDirectories: true)
            default:
                let scalar = UnicodeScalar(Int(type)).map(String.init) ?? String(type)
                throw ExtractionError.unsupportedEntry(scalar)
            }

            let padding = (512 - (size % 512)) % 512
            if padding > 0 {
                guard try readExactly(padding, from: stream) != nil else {
                    throw ExtractionError.corruptArchive("truncated padding after \(path)")
                }
            }
        }

        throw ExtractionError.corruptArchive("missing tar end marker")
    }

    static func validatedRelativePath(_ rawPath: String) throws -> String {
        guard !rawPath.isEmpty, !rawPath.hasPrefix("/") else {
            throw ExtractionError.unsafePath(rawPath)
        }
        let components = rawPath.split(separator: "/", omittingEmptySubsequences: true)
        guard !components.isEmpty,
            components.allSatisfy({ $0 != "." && $0 != ".." && !$0.contains("\\") })
        else {
            throw ExtractionError.unsafePath(rawPath)
        }
        return components.joined(separator: "/")
    }

    private static func field(in header: Data, range: Range<Int>) throws -> String {
        let bytes = header[range].prefix { $0 != 0 }
        guard let result = String(bytes: bytes, encoding: .utf8) else {
            throw ExtractionError.corruptArchive("non-UTF-8 tar path")
        }
        return result
    }

    private static func octal(in header: Data, range: Range<Int>) throws -> Int {
        guard
            let value = String(bytes: header[range], encoding: .ascii)?
                .trimmingCharacters(in: CharacterSet(charactersIn: " \0")),
            let parsed = Int(value.isEmpty ? "0" : value, radix: 8), parsed >= 0
        else {
            throw ExtractionError.corruptArchive("invalid tar entry size")
        }
        return parsed
    }

    private static func copy(_ count: Int, from stream: gzFile, to file: FileHandle) throws {
        var remaining = count
        while remaining > 0 {
            let chunkSize = min(remaining, 1 << 20)
            guard let chunk = try readExactly(chunkSize, from: stream) else {
                throw ExtractionError.corruptArchive("truncated file body")
            }
            try file.write(contentsOf: chunk)
            remaining -= chunk.count
        }
    }

    private static func readExactly(_ count: Int, from stream: gzFile) throws -> Data? {
        if count == 0 { return Data() }
        var result = Data(count: count)
        var offset = 0

        while offset < count {
            let bytesRead: Int32 = result.withUnsafeMutableBytes { rawBuffer in
                guard let baseAddress = rawBuffer.baseAddress else { return 0 }
                return gzread(stream, baseAddress.advanced(by: offset), UInt32(count - offset))
            }
            if bytesRead < 0 {
                var errorNumber: Int32 = 0
                let message = gzerror(stream, &errorNumber).map(String.init(cString:)) ?? "unknown gzip error"
                throw ExtractionError.corruptArchive(message)
            }
            if bytesRead == 0 {
                if offset == 0 { return nil }
                throw ExtractionError.corruptArchive("unexpected end of gzip stream")
            }
            offset += Int(bytesRead)
        }
        return result
    }
}
