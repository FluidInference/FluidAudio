import Foundation

/// Long-text chunker that mirrors `chunkText()` from the upstream Supertonic
/// reference `Helper.swift`.
///
/// The chunker splits by paragraphs first, then by sentences (abbreviation-
/// aware), then by commas, and finally falls back to whitespace boundaries
/// so individual chunks never exceed the configured `maxLen`. The default
/// cap is 300 characters for Latin-script input and 120 characters for CJK
/// (Korean / Japanese), matching what the upstream Swift CLI ships.
enum Supertonic3TextChunker {

    private static let abbreviations: [String] = [
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.",
        "St.", "Ave.", "Rd.", "Blvd.", "Dept.", "Inc.", "Ltd.",
        "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D.",
    ]

    static func chunk(text rawText: String, maxLen: Int) -> [String] {
        let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return []
        }

        let paragraphs = splitParagraphs(trimmed)
        var chunks: [String] = []

        for paragraph in paragraphs {
            let para = paragraph.trimmingCharacters(in: .whitespacesAndNewlines)
            if para.isEmpty { continue }

            if para.count <= maxLen {
                chunks.append(para)
                continue
            }

            packSentences(para, maxLen: maxLen, into: &chunks)
        }

        return chunks
    }

    // MARK: - Paragraph split (blank line boundary)

    private static func splitParagraphs(_ text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: "\\n\\s*\\n") else {
            return [text]
        }
        let nsrange = NSRange(text.startIndex..., in: text)
        var lastEnd = text.startIndex
        var paragraphs: [String] = []

        regex.enumerateMatches(in: text, range: nsrange) { match, _, _ in
            if let match = match, let r = Range(match.range, in: text) {
                paragraphs.append(String(text[lastEnd..<r.lowerBound]))
                lastEnd = r.upperBound
            }
        }
        if lastEnd < text.endIndex {
            paragraphs.append(String(text[lastEnd...]))
        }
        return paragraphs.isEmpty ? [text] : paragraphs
    }

    // MARK: - Sentence packing (with comma + word fallbacks)

    private static func packSentences(
        _ paragraph: String, maxLen: Int, into chunks: inout [String]
    ) {
        let sentences = splitSentences(paragraph)
        var current = ""

        for sentence in sentences {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }

            if trimmed.count > maxLen {
                flush(&current, into: &chunks)
                packCommas(trimmed, maxLen: maxLen, into: &chunks)
                continue
            }

            if current.count + trimmed.count + 1 > maxLen, !current.isEmpty {
                chunks.append(current)
                current = ""
            }
            current = current.isEmpty ? trimmed : "\(current) \(trimmed)"
        }
        flush(&current, into: &chunks)
    }

    private static func packCommas(
        _ sentence: String, maxLen: Int, into chunks: inout [String]
    ) {
        var current = ""
        for rawPart in sentence.components(separatedBy: ",") {
            let part = rawPart.trimmingCharacters(in: .whitespacesAndNewlines)
            if part.isEmpty { continue }

            if part.count > maxLen {
                flush(&current, into: &chunks)
                packWords(part, maxLen: maxLen, into: &chunks)
                continue
            }

            if current.count + part.count + 2 > maxLen, !current.isEmpty {
                chunks.append(current)
                current = ""
            }
            current = current.isEmpty ? part : "\(current), \(part)"
        }
        flush(&current, into: &chunks)
    }

    private static func packWords(
        _ phrase: String, maxLen: Int, into chunks: inout [String]
    ) {
        var current = ""
        for word in phrase.split(whereSeparator: { $0.isWhitespace }) {
            let w = String(word)
            if current.count + w.count + 1 > maxLen, !current.isEmpty {
                chunks.append(current)
                current = ""
            }
            current = current.isEmpty ? w : "\(current) \(w)"
        }
        flush(&current, into: &chunks)
    }

    @inline(__always)
    private static func flush(_ buffer: inout String, into chunks: inout [String]) {
        let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            chunks.append(trimmed)
        }
        buffer = ""
    }

    // MARK: - Sentence boundary detection (abbreviation aware)

    private static func splitSentences(_ text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: "([.!?])\\s+") else {
            return [text]
        }
        let nsrange = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, range: nsrange)
        if matches.isEmpty { return [text] }

        var sentences: [String] = []
        var lastEnd = text.startIndex

        for match in matches {
            guard let matchRange = Range(match.range, in: text) else { continue }
            let beforePunc = String(text[lastEnd..<matchRange.lowerBound])
            guard let puncRange = Range(NSRange(location: match.range.location, length: 1), in: text)
            else { continue }
            let punc = String(text[puncRange])
            let combined = beforePunc.trimmingCharacters(in: .whitespaces) + punc

            let isAbbreviation = abbreviations.contains { combined.hasSuffix($0) }
            if !isAbbreviation {
                sentences.append(String(text[lastEnd..<matchRange.upperBound]))
                lastEnd = matchRange.upperBound
            }
        }
        if lastEnd < text.endIndex {
            sentences.append(String(text[lastEnd...]))
        }
        return sentences.isEmpty ? [text] : sentences
    }
}
