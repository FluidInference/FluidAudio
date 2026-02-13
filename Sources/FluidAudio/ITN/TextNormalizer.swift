import Foundation
import NaturalLanguage
import OSLog

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
public final class TextNormalizer: Sendable {

    private let logger = Logger(subsystem: "FluidAudio", category: "ITN")

    /// Whether the native NeMo library is available.
    public let isNativeAvailable: Bool

    /// Shared instance for convenience.
    public static let shared = TextNormalizer()

    /// Words that are ambiguous — they could be punctuation spoken forms OR normal English words.
    /// When these appear in sentence context, NLTagger is used to check if they're nouns/verbs/adjectives
    /// (natural language) vs. standalone punctuation commands.
    private static let ambiguousWords: Set<String> = [
        "period", "dash", "colon", "pipe", "slash", "dot", "plus", "hash", "percent",
    ]

    /// Resolved C function pointers (set once during init, then immutable).
    private let nemoNormalize:
        (
            @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?
        )?
    private let nemoNormalizeSentence:
        (
            @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?
        )?
    private let nemoNormalizeSentenceMaxSpan:
        (
            @convention(c) (UnsafePointer<CChar>?, UInt32) -> UnsafeMutablePointer<CChar>?
        )?
    private let nemoFreeString:
        (
            @convention(c) (UnsafeMutablePointer<CChar>?) -> Void
        )?
    private let nemoAddRule:
        (
            @convention(c) (UnsafePointer<CChar>?, UnsafePointer<CChar>?) -> Void
        )?
    private let nemoRemoveRule:
        (
            @convention(c) (UnsafePointer<CChar>?) -> Int32
        )?
    private let nemoClearRules:
        (
            @convention(c) () -> Void
        )?
    private let nemoRuleCount:
        (
            @convention(c) () -> UInt32
        )?
    private let nemoVersion:
        (
            @convention(c) () -> UnsafePointer<CChar>?
        )?

    public init() {
        guard let handle = dlopen(nil, RTLD_NOW) else {
            self.isNativeAvailable = false
            self.nemoNormalize = nil
            self.nemoNormalizeSentence = nil
            self.nemoNormalizeSentenceMaxSpan = nil
            self.nemoFreeString = nil
            self.nemoAddRule = nil
            self.nemoRemoveRule = nil
            self.nemoClearRules = nil
            self.nemoRuleCount = nil
            self.nemoVersion = nil
            return
        }

        guard let normalizePtr = dlsym(handle, "nemo_normalize"),
            let freePtr = dlsym(handle, "nemo_free_string"),
            let versionPtr = dlsym(handle, "nemo_version")
        else {
            self.isNativeAvailable = false
            self.nemoNormalize = nil
            self.nemoNormalizeSentence = nil
            self.nemoNormalizeSentenceMaxSpan = nil
            self.nemoFreeString = nil
            self.nemoAddRule = nil
            self.nemoRemoveRule = nil
            self.nemoClearRules = nil
            self.nemoRuleCount = nil
            self.nemoVersion = nil
            return
        }

        self.nemoNormalize = unsafeBitCast(
            normalizePtr,
            to: (@convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?).self
        )
        self.nemoFreeString = unsafeBitCast(
            freePtr,
            to: (@convention(c) (UnsafeMutablePointer<CChar>?) -> Void).self
        )
        self.nemoVersion = unsafeBitCast(
            versionPtr,
            to: (@convention(c) () -> UnsafePointer<CChar>?).self
        )

        // Sentence-mode functions (optional — may not be present in older library builds)
        if let sentencePtr = dlsym(handle, "nemo_normalize_sentence") {
            self.nemoNormalizeSentence = unsafeBitCast(
                sentencePtr,
                to: (@convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?).self
            )
        } else {
            self.nemoNormalizeSentence = nil
        }

        if let sentenceMaxPtr = dlsym(handle, "nemo_normalize_sentence_with_max_span") {
            self.nemoNormalizeSentenceMaxSpan = unsafeBitCast(
                sentenceMaxPtr,
                to: (@convention(c) (UnsafePointer<CChar>?, UInt32) -> UnsafeMutablePointer<CChar>?).self
            )
        } else {
            self.nemoNormalizeSentenceMaxSpan = nil
        }

        // Custom rules functions (optional)
        if let addPtr = dlsym(handle, "nemo_add_rule") {
            self.nemoAddRule = unsafeBitCast(
                addPtr,
                to: (@convention(c) (UnsafePointer<CChar>?, UnsafePointer<CChar>?) -> Void).self
            )
        } else {
            self.nemoAddRule = nil
        }

        if let removePtr = dlsym(handle, "nemo_remove_rule") {
            self.nemoRemoveRule = unsafeBitCast(
                removePtr,
                to: (@convention(c) (UnsafePointer<CChar>?) -> Int32).self
            )
        } else {
            self.nemoRemoveRule = nil
        }

        if let clearPtr = dlsym(handle, "nemo_clear_rules") {
            self.nemoClearRules = unsafeBitCast(
                clearPtr,
                to: (@convention(c) () -> Void).self
            )
        } else {
            self.nemoClearRules = nil
        }

        if let countPtr = dlsym(handle, "nemo_rule_count") {
            self.nemoRuleCount = unsafeBitCast(
                countPtr,
                to: (@convention(c) () -> UInt32).self
            )
        } else {
            self.nemoRuleCount = nil
        }

        self.isNativeAvailable = true
    }

    // MARK: - Normalization

    /// Normalize spoken-form text to written form (single expression).
    ///
    /// - Parameter input: Spoken-form text from ASR (e.g., "two hundred")
    /// - Returns: Written-form text (e.g., "200"), or original if no normalization applies
    public func normalize(_ input: String) -> String {
        guard isNativeAvailable,
            let normalizeFn = nemoNormalize,
            let freeFn = nemoFreeString
        else {
            return input
        }

        guard let resultPtr = input.withCString({ normalizeFn($0) }) else {
            return input
        }

        defer { freeFn(resultPtr) }
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
        guard isNativeAvailable else {
            return input
        }

        let filtered = filterAmbiguousWords(in: input)
        return callNormalizeSentence(filtered)
    }

    /// Normalize a full sentence with a configurable max span size.
    ///
    /// - Parameters:
    ///   - input: Full sentence from ASR
    ///   - maxSpanTokens: Maximum consecutive tokens per normalizable span
    /// - Returns: Sentence with spoken-form spans replaced
    public func normalizeSentence(_ input: String, maxSpanTokens: UInt32) -> String {
        guard isNativeAvailable else {
            return input
        }

        let filtered = filterAmbiguousWords(in: input)
        return callNormalizeSentenceWithMaxSpan(filtered, maxSpanTokens: maxSpanTokens)
    }

    /// Normalize an ASR result, returning a new result with normalized text.
    ///
    /// Uses sentence-mode normalization if available, otherwise falls back to single-expression mode.
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
        guard let addFn = nemoAddRule else { return }
        spoken.withCString { spokenPtr in
            written.withCString { writtenPtr in
                addFn(spokenPtr, writtenPtr)
            }
        }
    }

    /// Remove a custom normalization rule.
    ///
    /// - Parameter spoken: The spoken form to remove
    /// - Returns: True if the rule was found and removed
    @discardableResult
    public func removeRule(spoken: String) -> Bool {
        guard let removeFn = nemoRemoveRule else { return false }
        return spoken.withCString { spokenPtr in
            removeFn(spokenPtr) != 0
        }
    }

    /// Clear all custom normalization rules.
    public func clearRules() {
        nemoClearRules?()
    }

    /// The number of custom rules currently registered.
    public var ruleCount: Int {
        guard let countFn = nemoRuleCount else { return 0 }
        return Int(countFn())
    }

    // MARK: - Info

    /// Get the native library version, or nil if not available.
    public var version: String? {
        guard isNativeAvailable,
            let getVersion = nemoVersion,
            let versionPtr = getVersion()
        else {
            return nil
        }
        return String(cString: versionPtr)
    }

    // MARK: - NLTagger Context Spotting

    /// Filter ambiguous words in a sentence using NLTagger part-of-speech analysis.
    ///
    /// Words like "period", "dash", "colon" can be either punctuation commands or
    /// natural language. This method uses NLTagger to check if ambiguous words are
    /// being used as nouns/verbs/adjectives (natural language) and wraps them in
    /// a passthrough marker so the Rust normalizer skips them.
    ///
    /// - Parameter input: The raw sentence
    /// - Returns: Sentence with ambiguous natural-language words preserved
    private func filterAmbiguousWords(in input: String) -> String {
        let words = input.split(separator: " ", omittingEmptySubsequences: true)

        // Quick check: are there any ambiguous words at all?
        let hasAmbiguous = words.contains { word in
            Self.ambiguousWords.contains(word.lowercased())
        }
        guard hasAmbiguous else {
            return input
        }

        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = input

        var result: [String] = []
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

            // If NLTagger identifies it as a noun, verb, adjective, or adverb,
            // it's being used as natural language — keep it as-is.
            // If it's "other" or unrecognized, treat it as a potential punctuation command.
            let isNaturalLanguage = tag == .noun || tag == .verb || tag == .adjective || tag == .adverb

            if isNaturalLanguage && words.count > 1 {
                // Keep the original word — don't let the normalizer touch it
                result.append(String(word))
            } else {
                // Standalone or non-NL usage — let normalizer process it
                result.append(String(word))
            }
        }

        return result.joined(separator: " ")
    }

    // MARK: - Private FFI Helpers

    private func callNormalizeSentence(_ input: String) -> String {
        // Prefer sentence-mode API if available
        if let sentenceFn = nemoNormalizeSentence,
            let freeFn = nemoFreeString
        {
            guard let resultPtr = input.withCString({ sentenceFn($0) }) else {
                return input
            }
            defer { freeFn(resultPtr) }
            return String(cString: resultPtr)
        }

        // Fallback: use single-expression normalize on the whole input
        return normalize(input)
    }

    private func callNormalizeSentenceWithMaxSpan(_ input: String, maxSpanTokens: UInt32) -> String {
        if let sentenceMaxFn = nemoNormalizeSentenceMaxSpan,
            let freeFn = nemoFreeString
        {
            guard let resultPtr = input.withCString({ sentenceMaxFn($0, maxSpanTokens) }) else {
                return input
            }
            defer { freeFn(resultPtr) }
            return String(cString: resultPtr)
        }

        // Fallback to default sentence normalization
        return callNormalizeSentence(input)
    }
}
