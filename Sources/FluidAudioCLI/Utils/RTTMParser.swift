#if os(macOS)
import FluidAudio
import Foundation

enum RTTMParserError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidLine(String)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "RTTM file not found at \(path)"
        case .invalidLine(let line):
            return "Invalid RTTM line: \(line)"
        }
    }
}

/// Lightweight RTTM parser for converting ground-truth annotations into `TimedSpeakerSegment`s.
enum RTTMParser {

    static func loadSegments(from path: String) throws -> [TimedSpeakerSegment] {
        guard FileManager.default.fileExists(atPath: path) else {
            throw RTTMParserError.fileNotFound(path)
        }

        let contents = try String(contentsOfFile: path, encoding: .utf8)
        var segments: [TimedSpeakerSegment] = []

        for rawLine in contents.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") {
                continue
            }

            let fields = line.split(whereSeparator: { $0.isWhitespace })
            guard fields.count >= 8, fields[0] == "SPEAKER" else {
                throw RTTMParserError.invalidLine(line)
            }

            guard
                let start = Float(fields[3]),
                let duration = Float(fields[4])
            else {
                throw RTTMParserError.invalidLine(line)
            }

            let speakerId = String(fields[7])
            let endTime = start + duration

            segments.append(
                TimedSpeakerSegment(
                    speakerId: speakerId,
                    embedding: [],
                    startTimeSeconds: start,
                    endTimeSeconds: endTime,
                    qualityScore: 1.0
                )
            )
        }

        return segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
    }
}
#endif
