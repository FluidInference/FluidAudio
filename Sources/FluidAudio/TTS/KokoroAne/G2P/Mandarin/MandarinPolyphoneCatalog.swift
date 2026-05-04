import Foundation

/// Loader / lookup for `POLYPHONIC_CHARS.txt`, the pronunciation
/// inventory the g2pW model is conditioned on.
///
/// File format (matches `g2pw/data/POLYPHONIC_CHARS.txt` upstream):
///
/// ```
/// <hanzi>\t<bopomofo_with_tone>
/// ```
///
/// One row per (char, valid-pronunciation) pair. Multiple rows for the
/// same char enumerate its allowed phonemes:
///
/// ```
/// 行    ㄒㄧㄥˊ
/// 行    ㄏㄤˊ
/// 行    ㄒㄧㄥˋ
/// ```
///
/// The model emits a softmax over the global label set; only the
/// indices in `candidates(for:)` are valid for a given target char,
/// so the runtime applies that subset as a phoneme mask before
/// argmax-ing.
public struct MandarinPolyphoneCatalog: Sendable {

    /// All characters that have ≥ 2 pronunciations, in the order they
    /// first appear in the file. Used by the model as the target-char
    /// vocabulary (a char index maps to the row index here).
    public let chars: [Character]
    /// Sorted-unique list of every bopomofo label that appears in the
    /// file. The model's output dimension equals `labels.count`.
    public let labels: [String]
    /// Per-char index into `labels` for each valid pronunciation.
    /// Empty array means the char is not polyphonic and should not be
    /// routed through g2pW.
    public let candidatesByChar: [Character: [Int]]
    /// Convenience reverse map for chars → catalog index.
    public let charIndex: [Character: Int]

    public enum LoadError: Swift.Error, LocalizedError {
        case malformed(String)

        public var errorDescription: String? {
            switch self {
            case .malformed(let what):
                return "POLYPHONIC_CHARS.txt parse error: \(what)"
            }
        }
    }

    public static func load(fileURL: URL) throws -> MandarinPolyphoneCatalog {
        let raw = try String(contentsOf: fileURL, encoding: .utf8)
        return try MandarinPolyphoneCatalog(text: raw)
    }

    /// Parse from an in-memory string. Empty / blank / `#`-comment lines
    /// are skipped to keep the file format permissive.
    public init(text: String) throws {
        var seenChars: [Character] = []
        var seenCharSet: Set<Character> = []
        var labelSet: Set<String> = []
        var candidatesRaw: [Character: [String]] = [:]

        // Split on unicode scalars rather than `Character`s — Swift
        // merges `\r\n` into a single grapheme cluster, which would
        // hide the line break from a `Character`-level splitter and
        // cause the entire CRLF file to be treated as one row.
        for rawLine in text.unicodeScalars.split(
            omittingEmptySubsequences: true,
            whereSeparator: { $0 == "\n" || $0 == "\r" })
        {
            let line = String(String.UnicodeScalarView(rawLine))
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }

            // Tolerate one or more whitespace characters as separator
            // (the upstream file uses TAB, but a sanitised copy may use
            // spaces — both are unambiguous because Hanzi never contain
            // ASCII whitespace).
            let parts = trimmed.split(
                maxSplits: 1, omittingEmptySubsequences: true,
                whereSeparator: { $0 == "\t" || $0 == " " })
            guard parts.count == 2 else {
                throw LoadError.malformed(
                    "expected '<hanzi><sep><bopomofo>', got '\(trimmed)'")
            }
            let charStr = String(parts[0])
            guard charStr.count == 1, let ch = charStr.first else {
                throw LoadError.malformed(
                    "expected single hanzi in column 1, got '\(charStr)'")
            }
            let label = String(parts[1]).trimmingCharacters(in: .whitespaces)
            if label.isEmpty {
                throw LoadError.malformed("empty bopomofo for '\(charStr)'")
            }

            if !seenCharSet.contains(ch) {
                seenChars.append(ch)
                seenCharSet.insert(ch)
            }
            labelSet.insert(label)
            candidatesRaw[ch, default: []].append(label)
        }

        let labels = labelSet.sorted()
        let labelToIndex: [String: Int] = Dictionary(
            uniqueKeysWithValues: labels.enumerated().map { ($1, $0) })

        var candidatesByChar: [Character: [Int]] = [:]
        candidatesByChar.reserveCapacity(candidatesRaw.count)
        for (ch, list) in candidatesRaw {
            // Preserve input order, dedupe (some upstream rows are
            // duplicated — a parse-time dedup keeps the mask compact).
            var seen: Set<Int> = []
            var indices: [Int] = []
            for label in list {
                guard let idx = labelToIndex[label] else { continue }
                if seen.insert(idx).inserted { indices.append(idx) }
            }
            candidatesByChar[ch] = indices
        }

        self.chars = seenChars
        self.labels = labels
        self.candidatesByChar = candidatesByChar
        self.charIndex = Dictionary(
            uniqueKeysWithValues: seenChars.enumerated().map { ($1, $0) })
    }

    /// Allowed label indices for `char`, or `nil` when the char isn't
    /// in the polyphone vocabulary (the caller should fall back to the
    /// pinyin dict in that case).
    public func candidates(for char: Character) -> [Int]? {
        candidatesByChar[char]
    }

    /// Reverse lookup: bopomofo string for a given label index.
    public func bopomofo(forLabel idx: Int) -> String? {
        guard idx >= 0, idx < labels.count else { return nil }
        return labels[idx]
    }

    /// Reverse lookup as the digit-suffixed form used by the v1.1-zh
    /// vocab (`MandarinBopomofoMap.encode` output style).
    ///
    /// Converts the trailing tone diacritic (`ˊ` / `ˇ` / `ˋ` / `˙`) to
    /// the numeric tone digit (`2` / `3` / `4` / `5`); a missing
    /// diacritic implies tone `1`. The model's labels are stored in
    /// the upstream diacritic form so the table matches the source
    /// data verbatim — the conversion happens at emission so callers
    /// see a string they can append straight into the phoneme buffer.
    public func bopomofoDigitForm(forLabel idx: Int) -> String? {
        guard let raw = bopomofo(forLabel: idx) else { return nil }
        return Self.toneDigitForm(raw)
    }

    /// Convert `<bopomofo><diacritic?>` → `<bopomofo><digit>`. Public
    /// so callers (and tests) can run the conversion without holding a
    /// catalog instance.
    public static func toneDigitForm(_ bopomofo: String) -> String {
        guard let last = bopomofo.last else { return bopomofo }
        switch last {
        case "ˊ":
            return String(bopomofo.dropLast()) + "2"
        case "ˇ":
            return String(bopomofo.dropLast()) + "3"
        case "ˋ":
            return String(bopomofo.dropLast()) + "4"
        case "˙":
            return String(bopomofo.dropLast()) + "5"
        default:
            // No diacritic — implicit tone 1. The upstream catalog
            // never emits a literal `ˉ` so this branch is the common
            // case for first-tone polyphones.
            return bopomofo + "1"
        }
    }
}
