import Foundation

/// Regex-based parser for SSML tags
/// Supports: <phoneme>, <sub>, <say-as>
enum SSMLTagParser {

    // MARK: - Regex Patterns

    /// Pattern for <phoneme alphabet="ipa" ph="...">content</phoneme>
    /// Captures: (1) alphabet (optional), (2) ph (required), (3) content
    private static let phonemePattern = try! NSRegularExpression(
        pattern: #"<phoneme(?:\s+alphabet\s*=\s*["']([^"']*)["'])?\s+ph\s*=\s*["']([^"']+)["']\s*>([^<]*)</phoneme>"#,
        options: [.caseInsensitive]
    )

    /// Pattern for <sub alias="...">content</sub>
    /// Captures: (1) alias, (2) content
    private static let subPattern = try! NSRegularExpression(
        pattern: #"<sub\s+alias\s*=\s*["']([^"']+)["']\s*>([^<]*)</sub>"#,
        options: [.caseInsensitive]
    )

    /// Pattern for <say-as interpret-as="..." format="...">content</say-as>
    /// Captures: (1) interpret-as, (2) format (optional), (3) content
    private static let sayAsPattern = try! NSRegularExpression(
        pattern:
            #"<say-as\s+interpret-as\s*=\s*["']([^"']+)["'](?:\s+format\s*=\s*["']([^"']*)["'])?\s*>([^<]*)</say-as>"#,
        options: [.caseInsensitive]
    )

    // MARK: - Public API

    /// Parse all SSML tags from text
    /// Returns tags in reverse document order (safe for sequential replacement)
    static func parse(_ text: String) -> [SSMLParsedTag] {
        var tags: [SSMLParsedTag] = []
        let nsText = text as NSString

        // Parse phoneme tags
        tags.append(contentsOf: parsePhoneme(text, nsText: nsText))

        // Parse sub tags
        tags.append(contentsOf: parseSub(text, nsText: nsText))

        // Parse say-as tags
        tags.append(contentsOf: parseSayAs(text, nsText: nsText))

        // Sort in reverse order by position for safe replacement
        return tags.sorted { $0.range.lowerBound > $1.range.lowerBound }
    }

    // MARK: - Private Parsing Methods

    private static func parsePhoneme(_ text: String, nsText: NSString) -> [SSMLParsedTag] {
        var tags: [SSMLParsedTag] = []
        let range = NSRange(location: 0, length: nsText.length)

        phonemePattern.enumerateMatches(in: text, options: [], range: range) { match, _, _ in
            guard let match = match else { return }

            let alphabet = extractGroup(match, group: 1, from: nsText) ?? "ipa"
            guard let ph = extractGroup(match, group: 2, from: nsText),
                let content = extractGroup(match, group: 3, from: nsText),
                let swiftRange = Range(match.range, in: text)
            else { return }

            tags.append(
                SSMLParsedTag(
                    type: .phoneme(alphabet: alphabet, ph: ph, content: content),
                    range: swiftRange
                ))
        }

        return tags
    }

    private static func parseSub(_ text: String, nsText: NSString) -> [SSMLParsedTag] {
        var tags: [SSMLParsedTag] = []
        let range = NSRange(location: 0, length: nsText.length)

        subPattern.enumerateMatches(in: text, options: [], range: range) { match, _, _ in
            guard let match = match else { return }

            guard let alias = extractGroup(match, group: 1, from: nsText),
                let content = extractGroup(match, group: 2, from: nsText),
                let swiftRange = Range(match.range, in: text)
            else { return }

            tags.append(
                SSMLParsedTag(
                    type: .sub(alias: alias, content: content),
                    range: swiftRange
                ))
        }

        return tags
    }

    private static func parseSayAs(_ text: String, nsText: NSString) -> [SSMLParsedTag] {
        var tags: [SSMLParsedTag] = []
        let range = NSRange(location: 0, length: nsText.length)

        sayAsPattern.enumerateMatches(in: text, options: [], range: range) { match, _, _ in
            guard let match = match else { return }

            guard let interpretAs = extractGroup(match, group: 1, from: nsText),
                let content = extractGroup(match, group: 3, from: nsText),
                let swiftRange = Range(match.range, in: text)
            else { return }

            let format = extractGroup(match, group: 2, from: nsText)

            tags.append(
                SSMLParsedTag(
                    type: .sayAs(interpretAs: interpretAs, format: format, content: content),
                    range: swiftRange
                ))
        }

        return tags
    }

    // MARK: - Helpers

    private static func extractGroup(
        _ match: NSTextCheckingResult,
        group: Int,
        from nsText: NSString
    ) -> String? {
        let groupRange = match.range(at: group)
        guard groupRange.location != NSNotFound else { return nil }
        return nsText.substring(with: groupRange)
    }
}
