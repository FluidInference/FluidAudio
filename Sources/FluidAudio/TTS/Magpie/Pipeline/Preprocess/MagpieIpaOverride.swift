import Foundation

/// Parses Magpie's `|`-delimited IPA override syntax.
///
/// The Magpie model card describes inline pronunciation overrides as:
///
///     "Hello world from | ˈ n ɛ m o ʊ | Text to Speech."
///
/// Inside each `|…|` region, tokens are **space-separated IPA characters**, each of
/// which is looked up directly in the language's `token2id` map (no G2P). Outside
/// the regions, text flows through the normal language tokenizer.
///
/// This type is a pure lexer — it only segments the input string. The caller
/// (`MagpieTokenizer`) is responsible for tokenizing the segments.
public enum MagpieIpaOverride {

    public enum Segment: Sendable, Equatable {
        /// Plain text to be handled by the language's G2P / tokenizer.
        case text(String)
        /// IPA tokens already space-separated. Look each up in `token2id` directly.
        case ipa(tokens: [String])
    }

    /// Segments `input` into alternating `.text` / `.ipa` runs.
    ///
    /// Rules:
    /// - Pairs of `|` delimit an IPA region. Whitespace inside is treated as a token
    ///   separator; consecutive whitespace collapses to a single split.
    /// - An unpaired trailing `|` is treated as literal text (no silent data loss).
    /// - Empty IPA regions (`||`) collapse to no segment.
    public static func segment(_ input: String) -> [Segment] {
        guard input.contains("|") else {
            return input.isEmpty ? [] : [.text(input)]
        }

        var segments: [Segment] = []
        var cursor = input.startIndex
        var inIpa = false
        var buffer = ""

        while cursor < input.endIndex {
            let ch = input[cursor]
            if ch == "|" {
                if inIpa {
                    let tokens = buffer.split(whereSeparator: { $0.isWhitespace }).map(String.init)
                    if !tokens.isEmpty {
                        segments.append(.ipa(tokens: tokens))
                    }
                } else {
                    if !buffer.isEmpty {
                        segments.append(.text(buffer))
                    }
                }
                buffer.removeAll(keepingCapacity: true)
                inIpa.toggle()
            } else {
                buffer.append(ch)
            }
            cursor = input.index(after: cursor)
        }

        // Trailing content: if we were still inside an IPA region at EOF, the leading
        // `|` was unmatched — emit it plus the buffered content as plain text so we
        // don't silently drop characters.
        if inIpa {
            segments.append(.text("|" + buffer))
        } else if !buffer.isEmpty {
            segments.append(.text(buffer))
        }

        return segments
    }
}
