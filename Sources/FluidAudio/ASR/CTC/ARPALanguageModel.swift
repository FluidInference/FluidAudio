import Foundation

/// ARPA n-gram language model for CTC beam search rescoring.
///
/// Loads unigrams and bigrams from an ARPA text file. Log10 probabilities
/// are converted to natural log for direct combination with CTC log-softmax output.
///
/// Usage:
/// ```swift
/// let lm = try ARPALanguageModel.load(from: arpaFileURL)
/// let score = lm.score(word: "runway", prev: "cleared")
/// ```
///
/// - Note: Only plain-text ARPA files are supported (not gzipped or binary KenLM).
///   Trigrams and higher-order n-grams are ignored.
public struct ARPALanguageModel: Sendable {

    private static let logger = AppLogger(category: "ARPALanguageModel")

    public struct Entry: Sendable {
        /// Log-probability in natural log (converted from log10)
        public let logProb: Float
        /// Backoff weight in natural log
        public let backoff: Float
    }

    /// Conversion factor from log10 to natural log
    public static let log10ToNat: Float = Float(log(10.0))

    /// Fallback log-probability for out-of-vocabulary words (≈ log(1e-10))
    public static let unkLogProb: Float = -23.026

    /// Unigram entries keyed by word
    public var unigrams: [String: Entry] = [:]
    /// Bigram entries keyed by context word → target word
    public var bigrams: [String: [String: Entry]] = [:]

    public init() {}

    /// Load an ARPA language model from a text file.
    ///
    /// Parses unigram and bigram sections. Trigrams and higher are skipped.
    /// Log10 probabilities are converted to natural log.
    ///
    /// - Parameter url: Path to an ARPA-format text file.
    /// - Returns: A populated `ARPALanguageModel`.
    /// - Throws: If the file cannot be opened.
    public static func load(from url: URL) throws -> ARPALanguageModel {
        guard let reader = ARPALineReader(url: url) else {
            throw ARPAError.cannotOpen(url.path)
        }
        var lm = ARPALanguageModel()
        var section = ""
        while let line = reader.readLine() {
            if line.isEmpty || line.hasPrefix("\\data\\") { continue }
            if line == "\\end\\" { break }
            if line.hasPrefix("\\") {
                section = line
                continue
            }
            // Skip header lines like "ngram 1=15" that aren't n-gram entries
            if line.hasPrefix("ngram ") { continue }

            let parts = line.components(separatedBy: "\t")
            guard let log10prob = Float(parts[0]) else {
                Self.logger.warning("Skipping malformed ARPA line: \(line)")
                continue
            }
            let prob = log10prob * log10ToNat
            if section == "\\1-grams:" && parts.count >= 2 {
                let word = parts[1]
                let backoff = parts.count >= 3 ? (Float(parts[2]) ?? 0.0) * log10ToNat : 0.0
                lm.unigrams[word] = Entry(logProb: prob, backoff: backoff)
            } else if section == "\\2-grams:" && parts.count >= 3 {
                let ctx = parts[1]
                let word = parts[2]
                let backoff = parts.count >= 4 ? (Float(parts[3]) ?? 0.0) * log10ToNat : 0.0
                var ctxDict = lm.bigrams[ctx, default: [:]]
                ctxDict[word] = Entry(logProb: prob, backoff: backoff)
                lm.bigrams[ctx] = ctxDict
            }
        }
        return lm
    }

    /// Score P(word | prev) in natural log, backing off to unigram if bigram is absent.
    ///
    /// - Parameters:
    ///   - word: The target word.
    ///   - prev: The preceding context word, or nil for unigram-only scoring.
    /// - Returns: Natural log probability.
    public func score(word: String, prev: String?) -> Float {
        if let p = prev, let bi = bigrams[p]?[word] { return bi.logProb }
        let backoff = prev.flatMap { unigrams[$0]?.backoff } ?? 0.0
        return backoff + (unigrams[word]?.logProb ?? ARPALanguageModel.unkLogProb)
    }
}

// MARK: - Errors

public enum ARPAError: Error, LocalizedError {
    case cannotOpen(String)

    public var errorDescription: String? {
        switch self {
        case .cannotOpen(let path):
            return "Cannot open ARPA file: \(path)"
        }
    }
}

// MARK: - Line Reader

/// Streaming line reader for efficient ARPA file parsing.
private final class ARPALineReader {
    private let fileHandle: FileHandle
    private var buffer = Data()
    private let chunkSize = 65_536
    private var eof = false

    init?(url: URL) {
        guard let fh = FileHandle(forReadingAtPath: url.path) else { return nil }
        fileHandle = fh
    }

    deinit { try? fileHandle.close() }

    func readLine() -> String? {
        while true {
            if let nl = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let slice = buffer[buffer.startIndex..<nl]
                buffer = Data(buffer[(nl + 1)...])
                return String(data: slice, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if eof {
                guard !buffer.isEmpty else { return nil }
                let result = String(data: buffer, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                buffer = Data()
                return result
            }
            let chunk = fileHandle.readData(ofLength: chunkSize)
            if chunk.isEmpty { eof = true } else { buffer.append(chunk) }
        }
    }
}
