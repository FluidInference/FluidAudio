import Foundation

/// Text preprocessor that mirrors the upstream `UnicodeProcessor` from
/// `https://github.com/supertone-inc/supertonic/blob/main/swift/Sources/Helper.swift`.
///
/// Pipeline:
///   1. NFKD-decompose the input (`decomposedStringWithCompatibilityMapping`).
///   2. Strip emoji blocks (wide Unicode ranges the model never saw).
///   3. Replace dashes/quotes/brackets/arrows with ASCII equivalents.
///   4. Drop a fixed set of decorative symbols (`♥`, `☆`, `\\`, …).
///   5. Expand known abbreviations (`@` → " at ", `e.g.,` → "for example,", …).
///   6. Collapse whitespace around punctuation + duplicate quotes.
///   7. Append a period if the text doesn't end with sentence-ending
///      punctuation or a closing bracket/quote.
///   8. Wrap the cleaned string with `<lang>…</lang>` tags.
///   9. Map each Unicode scalar through the loaded `unicode_indexer.json`;
///      unknown scalars receive `-1` so the model can mask them.
///
/// `unicode_indexer.json` ships as a flat `[Int64]` keyed by codepoint, so
/// the lookup is an O(1) index into a Swift array.
struct Supertonic3UnicodeProcessor {

    let indexer: [Int64]

    init(unicodeIndexerURL: URL) throws {
        do {
            let data = try Data(contentsOf: unicodeIndexerURL)
            self.indexer = try JSONDecoder().decode([Int64].self, from: data)
        } catch {
            throw Supertonic3Error.unicodeIndexerLoadFailed("\(error)")
        }
    }

    /// Encode a batch of (text, language) pairs into padded Int64 IDs +
    /// per-row float masks (`[bsz, 1, maxLen]`).
    func encode(
        texts: [String], languages: [String]
    ) throws -> (ids: [[Int64]], mask: [[[Float]]]) {
        precondition(texts.count == languages.count, "texts/languages length mismatch")

        var processed: [String] = []
        processed.reserveCapacity(texts.count)
        for (text, lang) in zip(texts, languages) {
            guard Supertonic3Constants.availableLanguages.contains(lang) else {
                throw Supertonic3Error.unsupportedLanguage(lang)
            }
            let cleaned = Self.preprocess(text: text, lang: lang)
            if cleaned.isEmpty {
                throw Supertonic3Error.emptyText
            }
            processed.append(cleaned)
        }

        // The text_encoder + duration_predictor models pin the T axis at
        // `textTFixed` (128). Truncate longer inputs and zero-pad shorter
        // ones so the bound MLMultiArray shape always matches the spec.
        let maxLen = Supertonic3Constants.textTFixed
        let lengths = processed.map { min($0.unicodeScalars.count, maxLen) }

        var ids: [[Int64]] = []
        ids.reserveCapacity(processed.count)
        for text in processed {
            var row = [Int64](repeating: 0, count: maxLen)
            for (j, scalar) in text.unicodeScalars.prefix(maxLen).enumerated() {
                let value = Int(scalar.value)
                if value < indexer.count {
                    row[j] = indexer[value]
                } else {
                    row[j] = -1
                }
            }
            ids.append(row)
        }

        let mask = Self.mask(from: lengths, maxLen: maxLen)
        return (ids, mask)
    }

    // MARK: - Text normalization (pure function for unit tests)

    static func preprocess(text rawText: String, lang: String) -> String {
        var text = rawText.decomposedStringWithCompatibilityMapping

        // Drop emoji codepoints in the wide Unicode planes.
        text = String(
            text.unicodeScalars.filter { !Self.isEmojiCodepoint($0.value) })

        for (old, new) in Self.symbolReplacements {
            text = text.replacingOccurrences(of: old, with: new)
        }
        for symbol in Self.decorativeSymbols {
            text = text.replacingOccurrences(of: symbol, with: "")
        }
        for (old, new) in Self.expressionReplacements {
            text = text.replacingOccurrences(of: old, with: new)
        }

        // Tighten spacing around terminal punctuation that the input may
        // have over-spaced after NFKD decomposition.
        for old in [" ,", " .", " !", " ?", " ;", " :", " '"] {
            text = text.replacingOccurrences(of: old, with: String(old.dropFirst()))
        }

        for repeated in [("\"\"", "\""), ("''", "'"), ("``", "`")] {
            while text.contains(repeated.0) {
                text = text.replacingOccurrences(of: repeated.0, with: repeated.1)
            }
        }

        if let regex = try? NSRegularExpression(pattern: "\\s+") {
            let range = NSRange(text.startIndex..., in: text)
            text = regex.stringByReplacingMatches(
                in: text, range: range, withTemplate: " ")
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        if !text.isEmpty,
            let regex = try? NSRegularExpression(
                pattern: "[.!?;:,'\"\\u201C\\u201D\\u2018\\u2019)\\]}…。」』】〉》›»]$")
        {
            let range = NSRange(text.startIndex..., in: text)
            if regex.firstMatch(in: text, range: range) == nil {
                text += "."
            }
        }

        return "<\(lang)>\(text)</\(lang)>"
    }

    static func mask(from lengths: [Int], maxLen: Int) -> [[[Float]]] {
        var rows: [[[Float]]] = []
        rows.reserveCapacity(lengths.count)
        for len in lengths {
            var row = [Float](repeating: 0, count: maxLen)
            for j in 0..<min(len, maxLen) {
                row[j] = 1
            }
            rows.append([row])
        }
        return rows
    }

    // MARK: - Tables (verbatim from upstream `helper.py` / `Helper.swift`)

    private static let symbolReplacements: KeyValuePairs<String, String> = [
        "\u{2013}": "-",  // en dash
        "\u{2011}": "-",  // non-breaking hyphen
        "\u{2014}": "-",  // em dash
        "_": " ",
        "\u{201C}": "\"",
        "\u{201D}": "\"",
        "\u{2018}": "'",
        "\u{2019}": "'",
        "\u{00B4}": "'",  // acute
        "`": "'",
        "[": " ",
        "]": " ",
        "|": " ",
        "/": " ",
        "#": " ",
        "\u{2192}": " ",  // right arrow
        "\u{2190}": " ",  // left arrow
    ]

    private static let decorativeSymbols: [String] = [
        "\u{2665}",  // ♥
        "\u{2606}",  // ☆
        "\u{2661}",  // ♡
        "\u{00A9}",  // ©
        "\\",
    ]

    private static let expressionReplacements: KeyValuePairs<String, String> = [
        "@": " at ",
        "e.g.,": "for example, ",
        "i.e.,": "that is, ",
    ]

    private static func isEmojiCodepoint(_ value: UInt32) -> Bool {
        switch value {
        case 0x1F600...0x1F64F: return true
        case 0x1F300...0x1F5FF: return true
        case 0x1F680...0x1F6FF: return true
        case 0x1F700...0x1F77F: return true
        case 0x1F780...0x1F7FF: return true
        case 0x1F800...0x1F8FF: return true
        case 0x1F900...0x1F9FF: return true
        case 0x1FA00...0x1FA6F: return true
        case 0x1FA70...0x1FAFF: return true
        case 0x2600...0x26FF: return true
        case 0x2700...0x27BF: return true
        case 0x1F1E6...0x1F1FF: return true
        default: return false
        }
    }
}
