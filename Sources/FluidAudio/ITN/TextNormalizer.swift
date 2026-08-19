import CNemoTextProcessing
import Foundation
import NaturalLanguage

/// Inverse Text Normalization (ITN) for post-processing ASR output.
///
/// Converts spoken-form text to written form:
/// - "two hundred thirty two" → "232"
/// - "five dollars and fifty cents" → "$5.50"
/// - "january fifth twenty twenty five" → "January 5, 2025"
/// - "period" → "."
///
/// Supports three modes:
/// - `normalize(_:)` — single expression normalization
/// - `normalizeSentence(_:)` — sentence-mode with sliding window span matching
/// - `normalizeSentence(_:maxSpanTokens:)` — sentence-mode with custom span size
///
/// Uses Apple NaturalLanguage framework to avoid false positives on ambiguous words
/// (e.g., "period" as a noun vs. punctuation).
///
/// The native engine (`text-processing-rs`) ships with the package as a binary
/// target and is linked directly — no runtime discovery, always available.
public final class TextNormalizer: Sendable {

    /// Whether the native NeMo library is available.
    ///
    /// Always `true`: the library is statically linked via the bundled
    /// `NemoTextProcessing` binary target. Kept for source compatibility with
    /// releases that resolved the library at runtime (≤ 0.15.6).
    public let isNativeAvailable = true

    /// Whether the linked library exposes the TN (written→spoken) surface used
    /// by the TTS frontends.
    ///
    /// Always `true` with the bundled library. Kept for source compatibility.
    public var isTnAvailable: Bool { true }

    /// Shared instance for convenience.
    public static let shared = TextNormalizer()

    /// Words that are ambiguous — they could be punctuation spoken forms OR normal English words.
    /// When these appear in sentence context, NLTagger is used to check if they're nouns/verbs/adjectives
    /// (natural language) vs. standalone punctuation commands.
    private static let ambiguousWords: Set<String> = [
        "period", "dash", "colon", "pipe", "slash", "dot", "plus", "hash", "percent",
    ]

    public init() {}

    // MARK: - Normalization

    /// Normalize spoken-form text to written form (single expression).
    ///
    /// - Parameter input: Spoken-form text from ASR (e.g., "two hundred")
    /// - Returns: Written-form text (e.g., "200"), or original if no normalization applies
    public func normalize(_ input: String) -> String {
        guard let resultPtr = nemo_normalize(input) else {
            return input
        }
        defer { nemo_free_string(resultPtr) }
        return String(cString: resultPtr)
    }

    // MARK: - Text Normalization (written → spoken)

    /// Normalize written-form text to spoken form (single expression), e.g.
    /// `"$5.50"` → `"five dollars fifty cents"`.
    public func tnNormalize(_ input: String) -> String {
        guard let resultPtr = nemo_tn_normalize(input) else {
            return input
        }
        defer { nemo_free_string(resultPtr) }
        return String(cString: resultPtr)
    }

    /// Normalize a full sentence to spoken form, rewriting written-form spans
    /// in place (`"I paid $5"` → `"I paid five dollars"`).
    public func tnNormalizeSentence(_ input: String) -> String {
        guard let resultPtr = nemo_tn_normalize_sentence(input) else {
            return input
        }
        defer { nemo_free_string(resultPtr) }
        return String(cString: resultPtr)
    }

    /// Normalize a full sentence, replacing spoken-form spans with written form.
    ///
    /// Uses a sliding window to find normalizable spans within the sentence.
    /// Applies NLTagger-based context spotting to avoid false positives on
    /// ambiguous words (e.g., "period" as a noun stays unchanged).
    ///
    /// - Parameter input: Full sentence from ASR
    /// - Returns: Sentence with spoken-form spans replaced
    public func normalizeSentence(_ input: String) -> String {
        let (masked, restore) = maskAmbiguousWords(in: input)
        guard let resultPtr = nemo_normalize_sentence(masked) else {
            return input
        }
        defer { nemo_free_string(resultPtr) }
        return restoreMaskedWords(String(cString: resultPtr), restore)
    }

    /// Normalize a full sentence with a configurable max span size.
    ///
    /// - Parameters:
    ///   - input: Full sentence from ASR
    ///   - maxSpanTokens: Maximum consecutive tokens per normalizable span
    /// - Returns: Sentence with spoken-form spans replaced
    public func normalizeSentence(_ input: String, maxSpanTokens: UInt32) -> String {
        let (masked, restore) = maskAmbiguousWords(in: input)
        guard let resultPtr = nemo_normalize_sentence_with_options(masked, 0, maxSpanTokens, 0) else {
            return input
        }
        defer { nemo_free_string(resultPtr) }
        return restoreMaskedWords(String(cString: resultPtr), restore)
    }

    /// Normalize an ASR result, returning a new result with normalized text.
    ///
    /// - Parameter result: The original ASR result
    /// - Returns: A new ASR result with normalized text
    public func normalize(result: ASRResult) -> ASRResult {
        let normalizedText = normalizeSentence(result.text)

        guard normalizedText != result.text else {
            return result
        }

        return ASRResult(
            text: normalizedText,
            confidence: result.confidence,
            duration: result.duration,
            processingTime: result.processingTime,
            tokenTimings: result.tokenTimings,
            ctcDetectedTerms: result.ctcDetectedTerms,
            ctcAppliedTerms: result.ctcAppliedTerms
        )
    }

    // MARK: - Custom Rules

    /// Add a custom spoken→written normalization rule.
    ///
    /// Custom rules have the highest priority, checked before all built-in taggers.
    /// Matching is case-insensitive on the spoken form.
    ///
    /// - Parameters:
    ///   - spoken: The spoken form to match (e.g., "gee pee tee")
    ///   - written: The written replacement (e.g., "GPT")
    public func addRule(spoken: String, written: String) {
        nemo_add_rule(spoken, written)
    }

    /// Remove a custom normalization rule.
    ///
    /// - Parameter spoken: The spoken form to remove
    /// - Returns: True if the rule was found and removed
    @discardableResult
    public func removeRule(spoken: String) -> Bool {
        nemo_remove_rule(spoken) != 0
    }

    /// Clear all custom normalization rules.
    public func clearRules() {
        nemo_clear_rules()
    }

    /// The number of custom rules currently registered.
    public var ruleCount: Int {
        Int(nemo_rule_count())
    }

    // MARK: - Info

    /// The native library version.
    public var version: String? {
        guard let versionPtr = nemo_version() else {
            return nil
        }
        return String(cString: versionPtr)
    }

    // MARK: - NLTagger Context Spotting

    /// Mask ambiguous words that NLTagger identifies as natural language, so the
    /// native normalizer can't rewrite them (e.g. the noun "period" → ".").
    ///
    /// Each protected word is replaced with a unique Private-Use-Area sentinel
    /// character; the native normalizer passes those through unchanged, and the
    /// caller restores them via the returned map. Returns the (possibly)
    /// rewritten string and a sentinel→original map (empty when nothing was
    /// masked).
    private func maskAmbiguousWords(in input: String) -> (masked: String, restore: [Character: String]) {
        let words = input.split(separator: " ", omittingEmptySubsequences: true)

        // Quick check: are there any ambiguous words at all?
        let hasAmbiguous = words.contains { word in
            Self.ambiguousWords.contains(word.lowercased())
        }
        guard hasAmbiguous else {
            return (input, [:])
        }

        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = input

        var result: [String] = []
        var restore: [Character: String] = [:]
        // Private Use Area (U+E000…) — never appears in real ASR text and is
        // passed through untouched by the native normalizer.
        var nextSentinel: UInt32 = 0xE000
        for word in words {
            let wordLower = word.lowercased()

            guard Self.ambiguousWords.contains(wordLower) else {
                result.append(String(word))
                continue
            }

            // Find this word's range in the original string for NLTagger
            guard let wordRange = input.range(of: word) else {
                result.append(String(word))
                continue
            }

            let tag = tagger.tag(at: wordRange.lowerBound, unit: .word, scheme: .lexicalClass).0

            // A noun/verb/adjective/adverb in a multi-word sentence is being used
            // as natural language — mask it so the normalizer leaves it alone.
            // Standalone or "other" usage is a potential punctuation command;
            // leave it for the normalizer to process.
            let isNaturalLanguage = tag == .noun || tag == .verb || tag == .adjective || tag == .adverb

            if isNaturalLanguage && words.count > 1, let scalar = UnicodeScalar(nextSentinel) {
                let sentinel = Character(scalar)
                nextSentinel += 1
                restore[sentinel] = String(word)
                result.append(String(sentinel))
            } else {
                result.append(String(word))
            }
        }

        return (result.joined(separator: " "), restore)
    }

    /// Restore sentinel characters produced by ``maskAmbiguousWords(in:)``.
    private func restoreMaskedWords(_ text: String, _ restore: [Character: String]) -> String {
        guard !restore.isEmpty else { return text }
        var out = text
        for (sentinel, original) in restore {
            out = out.replacingOccurrences(of: String(sentinel), with: original)
        }
        return out
    }
}
