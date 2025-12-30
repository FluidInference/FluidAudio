import Foundation

/// Regex-based parser for SSML tags
/// Supports: <phoneme>, <sub>, <say-as>
enum SSMLTagParser {

    // MARK: - Regex Patterns

    /// Pattern for <phoneme ...>content</phoneme>
    /// Captures: (1) attributes, (2) content
    private static let phonemePattern = try! NSRegularExpression(
        pattern: #"<phoneme\s+([^>]+)>([^<]*)</phoneme>"#,
        options: [.caseInsensitive]
    )

    /// Pattern for <sub ...>content</sub>
    /// Captures: (1) attributes, (2) content
    private static let subPattern = try! NSRegularExpression(
        pattern: #"<sub\s+([^>]+)>([^<]*)</sub>"#,
        options: [.caseInsensitive]
    )

    /// Pattern for <say-as ...>content</say-as>
    /// Captures: (1) attributes, (2) content
    private static let sayAsPattern = try! NSRegularExpression(
        pattern: #"<say-as\s+([^>]+)>([^<]*)</say-as>"#,
        options: [.caseInsensitive]
    )

    /// Pattern for extracting attribute values
    private static func extractAttribute(_ name: String, from attributes: String) -> String? {
        let pattern = try! NSRegularExpression(
            pattern: #"\b"# + name + #"\s*=\s*["']([^"']*)["']"#,
            options: [.caseInsensitive]
        )
        let nsAttributes = attributes as NSString
        let range = NSRange(location: 0, length: nsAttributes.length)
        guard let match = pattern.firstMatch(in: attributes, options: [], range: range) else { return nil }
        let valueRange = match.range(at: 1)
        guard valueRange.location != NSNotFound else { return nil }
        return nsAttributes.substring(with: valueRange)
    }

    // MARK: - Public API

    /// Parse all SSML tags from text
    /// Returns tags in reverse document order (safe for sequential replacement)
    static func parse(_ text: String) -> [SSMLParsedTag] {
        let nsText = text as NSString
        let range = NSRange(location: 0, length: nsText.length)

        var tags: [SSMLParsedTag] = []

        // Parse all tag types using shared enumeration
        enumerateMatches(phonemePattern, in: text, nsText: nsText, range: range) { match in
            guard let attributes = extractGroup(match, group: 1, from: nsText),
                let content = extractGroup(match, group: 2, from: nsText),
                let ph = extractAttribute("ph", from: attributes)
            else { return nil }
            let alphabet = extractAttribute("alphabet", from: attributes) ?? "ipa"
            return .phoneme(alphabet: alphabet, ph: ph, content: content)
        }.forEach { tags.append($0) }

        enumerateMatches(subPattern, in: text, nsText: nsText, range: range) { match in
            guard let attributes = extractGroup(match, group: 1, from: nsText),
                let content = extractGroup(match, group: 2, from: nsText),
                let alias = extractAttribute("alias", from: attributes)
            else { return nil }
            return .sub(alias: alias, content: content)
        }.forEach { tags.append($0) }

        enumerateMatches(sayAsPattern, in: text, nsText: nsText, range: range) { match in
            guard let attributes = extractGroup(match, group: 1, from: nsText),
                let content = extractGroup(match, group: 2, from: nsText),
                let interpretAs = extractAttribute("interpret-as", from: attributes)
            else { return nil }
            let format = extractAttribute("format", from: attributes)
            return .sayAs(interpretAs: interpretAs, format: format, content: content)
        }.forEach { tags.append($0) }

        // Sort in reverse order by position for safe replacement
        return tags.sorted { $0.range.lowerBound > $1.range.lowerBound }
    }

    // MARK: - Private Parsing Methods

    /// Enumerate regex matches and convert to SSMLParsedTag using provided closure
    private static func enumerateMatches(
        _ pattern: NSRegularExpression,
        in text: String,
        nsText: NSString,
        range: NSRange,
        createTagType: (NSTextCheckingResult) -> SSMLParsedTag.TagType?
    ) -> [SSMLParsedTag] {
        var tags: [SSMLParsedTag] = []

        pattern.enumerateMatches(in: text, options: [], range: range) { match, _, _ in
            guard let match = match,
                let swiftRange = Range(match.range, in: text),
                let tagType = createTagType(match)
            else { return }

            tags.append(SSMLParsedTag(type: tagType, range: swiftRange))
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
