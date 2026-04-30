import Foundation

/// Splits long input text into segments that each fit within CosyVoice3's
/// 250-token Flow input cap.
///
/// The Flow CFM model is exported with a fixed `[1, 250]` `token_total`
/// shape (`CosyVoice3Constants.flowTotalTokens`). After the prompt's speech
/// tokens consume `~85–95` slots (default voice), each `synthesize(...)`
/// call has room for roughly `~155` new speech tokens of output (≈ 6.4 s of
/// audio at the 40 ms/token rate `tokenMelRatio × hiftSamplesPerFrame /
/// sampleRate = 2 × 480 / 24_000`). Long phrases truncate mid-utterance.
///
/// This chunker greedily packs input into segments under a target speech-
/// token budget, splitting preferentially on hard sentence enders
/// (`. ! ? 。 ！ ？ \n`) and falling back to soft clause separators
/// (`, ; ， ； 、 ：`) when sentences exceed the budget. Synthesis is run
/// per-chunk and audio is concatenated with a small cosine cross-fade at
/// boundaries (handled by the caller, not here).
///
/// **Token-rate estimate** (calibrated against minimax-zh corpus runs):
/// - CJK char        ≈ 7.5 speech tokens (worst-case observed real rate;
///                                          5.5 was empirically too low and
///                                          let ~16% of phrases hit cap)
/// - ASCII char      ≈ 1.5 speech tokens (BPE compresses; English is faster)
/// - Other (Latin-1) ≈ 2.5 speech tokens (middle ground for accented Latin)
///
/// Default `maxSpeechTokens = 110` leaves a ~45-token safety margin under
/// the typical room-for-new of ~155. The 30-token force-split overshoot
/// can push a committed chunk to ~140 estimated, still comfortably under
/// the cap once the conservative 5.5-tokens/CJK-char heuristic is
/// reconciled with real generation rates. The synthesizer still emits
/// its `LLM-Decode budget exhausted` warning if a chunk somehow exceeds
/// the cap, so over-estimates are self-healing.
public enum CosyVoice3TextChunker {

    /// Sentence-ending punctuation. Always commit the current chunk after
    /// these, regardless of running token count.
    private static let hardEnders: Set<Character> = [
        "。", "！", "？", ".", "!", "?", "\n",
    ]

    /// Clause-internal punctuation. Commit only when the running token
    /// count is at or above the budget — soft splits should be preferred
    /// over force-splits but not preferred over hard enders.
    private static let softEnders: Set<Character> = [
        "，", "、", "；", "：", ";", ",", " ",
    ]

    /// Default speech-token budget per chunk. Keeps a ~45-token margin
    /// under the typical room-for-new of ~155 (= `flowTotalTokens=250`
    /// minus a typical prompt of ~95 tokens). The 30-token force-split
    /// overshoot may push committed chunks to ~140 estimated, still under
    /// the structural cap.
    public static let defaultMaxSpeechTokens: Int = 110

    /// Split `text` into chunks each estimated to produce ≤
    /// `maxSpeechTokens` LLM speech tokens. Returns `[text]` (single
    /// chunk) when the input already fits. Returns `[]` when `text` is
    /// empty or whitespace-only.
    public static func chunk(
        _ text: String,
        maxSpeechTokens: Int = defaultMaxSpeechTokens
    ) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        if estimateSpeechTokens(trimmed) <= maxSpeechTokens {
            return [trimmed]
        }

        var chunks: [String] = []
        var current = ""
        for ch in trimmed {
            current.append(ch)
            let tokensSoFar = estimateSpeechTokens(current)

            if hardEnders.contains(ch) {
                let pruned = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !pruned.isEmpty { chunks.append(pruned) }
                current = ""
                continue
            }
            if tokensSoFar >= maxSpeechTokens && softEnders.contains(ch) {
                let pruned = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !pruned.isEmpty { chunks.append(pruned) }
                current = ""
                continue
            }
            // Force-split if no punctuation has appeared within a 30-token
            // overshoot. Prefer the most recent whitespace; fall back to
            // hard-cut at the current position. Hard-cut on continuous CJK
            // (no whitespace) is rare in normalized input but can happen
            // when the normalizer collapses spaces.
            if tokensSoFar >= maxSpeechTokens + 30 {
                if let lastSpace = current.lastIndex(where: { $0 == " " }),
                    lastSpace != current.startIndex
                {
                    let head = String(current[..<lastSpace])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let tail = String(current[current.index(after: lastSpace)...])
                    if !head.isEmpty { chunks.append(head) }
                    current = tail
                } else {
                    let pruned = current.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !pruned.isEmpty { chunks.append(pruned) }
                    current = ""
                }
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { chunks.append(tail) }
        return chunks
    }

    /// Rough estimate of how many SPEECH tokens the LLM-Decode AR loop
    /// will produce for `s`. Used by `chunk(...)` to size segments under
    /// the structural Flow cap.
    public static func estimateSpeechTokens(_ s: String) -> Int {
        var total = 0.0
        for scalar in s.unicodeScalars {
            if isCJK(scalar) {
                total += 7.5
            } else if scalar.isASCII {
                total += 1.5
            } else {
                total += 2.5
            }
        }
        return Int(total.rounded())
    }

    private static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        // CJK Unified Ideographs (the bulk of zh/yue text)
        if (0x4E00...0x9FFF).contains(v) { return true }
        // CJK Unified Ideographs Extension A
        if (0x3400...0x4DBF).contains(v) { return true }
        // Hiragana
        if (0x3040...0x309F).contains(v) { return true }
        // Katakana
        if (0x30A0...0x30FF).contains(v) { return true }
        // Hangul Syllables
        if (0xAC00...0xD7AF).contains(v) { return true }
        return false
    }
}
