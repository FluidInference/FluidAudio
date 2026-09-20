#if os(macOS)
import CryptoKit
import Foundation

/// Loads and validates the fixed LocalVQE ASR benchmark manifest.
enum EnhanceBenchmarkDataset {
    struct Example: Equatable {
        let fileID: String
        let mic: URL
        let lpb: URL
        let clean: URL
        let ser: Int
        let farendNoisy: Bool
        let nearendNoisy: Bool
    }

    enum DatasetError: Error, LocalizedError, Equatable {
        case duplicateFileID(String)
        case emptyMetadata
        case invalidInteger(line: Int, column: String, value: String)
        case malformedRow(line: Int, expected: Int, actual: Int)
        case missingAudio(fileID: String, path: String)
        case missingColumn(String)
        case missingFileID(line: Int)

        var errorDescription: String? {
            switch self {
            case .duplicateFileID(let fileID):
                return "duplicate benchmark fileid '\(fileID)'"
            case .emptyMetadata:
                return "benchmark meta.csv is empty"
            case .invalidInteger(let line, let column, let value):
                return "benchmark meta.csv line \(line) has invalid \(column) value '\(value)'"
            case .malformedRow(let line, let expected, let actual):
                return "benchmark meta.csv line \(line) has \(actual) fields; expected \(expected)"
            case .missingAudio(let fileID, let path):
                return "benchmark fileid \(fileID) is missing \(path)"
            case .missingColumn(let column):
                return "benchmark meta.csv is missing required column '\(column)'"
            case .missingFileID(let line):
                return "benchmark meta.csv line \(line) has an empty fileid"
            }
        }
    }

    private struct MetadataRow {
        let fileID: String
        let ser: Int
        let farendNoisy: Bool
        let nearendNoisy: Bool
    }

    static func loadExamples(
        from directory: URL,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) throws -> [Example] {
        let metadata = try String(contentsOf: directory.appendingPathComponent("meta.csv"), encoding: .utf8)
        return try parseMetadata(metadata).map { row in
            let stem = "fileid_\(row.fileID)"
            let mic = directory.appendingPathComponent("\(stem)_mic.wav")
            let lpb = directory.appendingPathComponent("\(stem)_lpb.wav")
            let clean = directory.appendingPathComponent("\(stem)_clean.wav")
            for url in [mic, lpb, clean] {
                guard fileExists(url.path) else {
                    throw DatasetError.missingAudio(fileID: row.fileID, path: url.lastPathComponent)
                }
            }
            return Example(
                fileID: row.fileID,
                mic: mic,
                lpb: lpb,
                clean: clean,
                ser: row.ser,
                farendNoisy: row.farendNoisy,
                nearendNoisy: row.nearendNoisy
            )
        }
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 1024 * 1024) ?? Data()
            guard !data.isEmpty else { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func parseMetadata(_ text: String) throws -> [MetadataRow] {
        var lines = text.split(omittingEmptySubsequences: true, whereSeparator: \.isNewline).map(String.init)
        guard !lines.isEmpty else { throw DatasetError.emptyMetadata }

        let header = fields(in: lines.removeFirst())
        let requiredColumns = ["fileid", "ser", "is_farend_noisy", "is_nearend_noisy"]
        for required in requiredColumns where !header.contains(required) {
            throw DatasetError.missingColumn(required)
        }

        guard let fileIDIndex = header.firstIndex(of: "fileid"),
            let serIndex = header.firstIndex(of: "ser"),
            let farendNoisyIndex = header.firstIndex(of: "is_farend_noisy"),
            let nearendNoisyIndex = header.firstIndex(of: "is_nearend_noisy")
        else {
            throw DatasetError.missingColumn("internal required-column lookup")
        }

        var seen = Set<String>()
        var rows: [MetadataRow] = []
        for (offset, line) in lines.enumerated() {
            let lineNumber = offset + 2
            let rowFields = fields(in: line)
            guard rowFields.count == header.count else {
                throw DatasetError.malformedRow(line: lineNumber, expected: header.count, actual: rowFields.count)
            }
            let fileID = rowFields[fileIDIndex]
            guard !fileID.isEmpty else { throw DatasetError.missingFileID(line: lineNumber) }
            guard seen.insert(fileID).inserted else { throw DatasetError.duplicateFileID(fileID) }
            let ser = try integer("ser", value: rowFields[serIndex], line: lineNumber)
            let farendNoisy = try integer(
                "is_farend_noisy", value: rowFields[farendNoisyIndex], line: lineNumber)
            let nearendNoisy = try integer(
                "is_nearend_noisy", value: rowFields[nearendNoisyIndex], line: lineNumber)
            rows.append(
                MetadataRow(
                    fileID: fileID,
                    ser: ser,
                    farendNoisy: farendNoisy != 0,
                    nearendNoisy: nearendNoisy != 0
                ))
        }
        return rows.sorted { lhs, rhs in
            guard let lhsID = Int(lhs.fileID), let rhsID = Int(rhs.fileID) else {
                return lhs.fileID < rhs.fileID
            }
            return lhsID < rhsID
        }
    }

    private static func fields(in line: String) -> [String] {
        line.split(separator: ",", omittingEmptySubsequences: false)
            .map { String($0).trimmingCharacters(in: .whitespaces) }
    }

    private static func integer(_ column: String, value rawValue: String, line: Int) throws -> Int {
        guard let value = Int(rawValue) else {
            throw DatasetError.invalidInteger(line: line, column: column, value: rawValue)
        }
        return value
    }
}
#endif
