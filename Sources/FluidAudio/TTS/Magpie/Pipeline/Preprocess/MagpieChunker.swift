import Foundation
import NaturalLanguage

/// One chunk of input text plus a hint for trailing silence.
public struct MagpieTextChunk: Sendable {
    public let text: String
    public let estimatedCodes: Int
    public let pauseAfterMs: Int

    public init(text: String, estimatedCodes: Int, pauseAfterMs: Int) {
        self.text = text
        self.estimatedCodes = estimatedCodes
        self.pauseAfterMs = pauseAfterMs
    }
}

/// Splits input text into chunks each estimated to fit inside the NanoCodec
/// 256-frame static-shape cap (~11.9 s of audio at 21.5 fps). Approach:
///
/// 1. `NaturalLanguage` sentence tokenizer (handles abbreviations, multilingual
///    punctuation including `。？！` for Chinese / Japanese-style text).
/// 2. Merge adjacent short sentences below `mergeBelowCodes` into one chunk so
///    one-word fragments don't create choppy prosody resets.
/// 3. Split too-long sentences on internal punctuation (`,;:—`), then on
///    English connector words (` and `, ` but `, ` because `, ` however `),
///    then as a last resort on whitespace at the codes-budget boundary.
/// 4. Assign `pauseAfterMs` based on the trailing punctuation of each chunk.
///
/// Capacity is estimated as `chars * codesPerChar` with a per-language ratio
/// (calibrated empirically: English ≈ 2.3 codes/char, Mandarin ≈ 7 codes/char,
/// since one CJK char represents a whole syllable). Estimates are intentionally
/// conservative (over-estimate) so chunks never exceed `maxCodesPerChunk` at
/// synth time; the real cap is enforced by `MagpieConstants.maxNanocodecFrames`
/// truncation in `MagpieNanocodec` if estimation drifts.
public enum MagpieChunker {

    /// Hard upper bound per chunk in codec frames. Set just below the model's
    /// 256-frame static cap to leave headroom for estimation error.
    public static let maxCodesPerChunk: Int = 220
    /// Sentences shorter than this get merged with their neighbor when possible.
    public static let mergeBelowCodes: Int = 30

    /// Pause inserted after each chunk based on its trailing punctuation. These
    /// are appended to the output PCM as zero-filled silence between chunks.
    public static let pauseSentenceMs: Int = 250
    public static let pauseClauseMs: Int = 80
    public static let pauseParagraphMs: Int = 450
    public static let pauseDefaultMs: Int = 100

    /// Per-language codes-per-character ratio. Conservative values (slight
    /// over-estimate) so we never under-cap.
    private static func codesPerChar(_ language: MagpieLanguage) -> Double {
        switch language {
        case .mandarin, .hindi: return 7.0
        case .vietnamese: return 3.0
        default: return 2.3
        }
    }

    /// Soft cap for the first chunk in streaming mode (codec frames).
    /// 50 frames ≈ 2.3 s of audio at 21.5 fps — a clause-sized head that
    /// trades a little prosody scope for low time-to-first-audio.
    public static let streamingFirstChunkCap: Int = 50

    /// Streaming variant: same as `chunk(text:language:)` but forces the first
    /// chunk to be small (≤ `firstChunkCap` codes) so the first audio yield
    /// arrives quickly. Subsequent chunks pack normally up to
    /// `maxCodesPerChunk`. If the first sentence is already small enough, this
    /// returns the same result as `chunk(text:language:)`.
    public static func chunkForStreaming(
        text: String,
        language: MagpieLanguage,
        firstChunkCap: Int = streamingFirstChunkCap
    ) -> [MagpieTextChunk] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        // Sentence-tokenize once; only re-shape the first sentence.
        let sentences = splitSentences(trimmed, language: language)
        guard let firstSentence = sentences.first else { return [] }

        let ratio = codesPerChar(language)
        let estimate: (String) -> Int = { Int((Double($0.count) * ratio).rounded(.up)) }

        // If the first sentence already fits in the cap, normal chunking wins.
        if estimate(firstSentence) <= firstChunkCap {
            return chunk(text: trimmed, language: language)
        }

        // Try internal punctuation (commas, semicolons, em-dashes, …); fall
        // back to whitespace if there is no internal punctuation.
        var head: String? = nil
        var tail: String? = nil

        let punctPieces = splitOn(firstSentence, where: { punctuationSplitChars.contains($0) })
        if punctPieces.count >= 2 {
            var picked: [String] = []
            var pickedCodes = 0
            var i = 0
            while i < punctPieces.count {
                let pe = estimate(punctPieces[i])
                if !picked.isEmpty && pickedCodes + pe > firstChunkCap { break }
                picked.append(punctPieces[i])
                pickedCodes += pe
                i += 1
            }
            if i < punctPieces.count && !picked.isEmpty {
                head = picked.joined(separator: " ")
                tail = punctPieces[i...].joined(separator: " ")
            }
        }

        if head == nil {
            let words = firstSentence.split(whereSeparator: { $0.isWhitespace })
            if words.count >= 2 {
                var picked: [String] = []
                var pickedCodes = 0
                var i = 0
                while i < words.count {
                    let wc = Int((Double(words[i].count + 1) * ratio).rounded(.up))
                    if !picked.isEmpty && pickedCodes + wc > firstChunkCap { break }
                    picked.append(String(words[i]))
                    pickedCodes += wc
                    i += 1
                }
                if i < words.count && !picked.isEmpty {
                    head = picked.joined(separator: " ")
                    tail = words[i...].joined(separator: " ")
                }
            }
        }

        guard let h = head, let t = tail, !h.isEmpty, !t.isEmpty else {
            // Couldn't split — fall back to normal chunking.
            return chunk(text: trimmed, language: language)
        }

        let headChunk = MagpieTextChunk(
            text: h,
            estimatedCodes: estimate(h),
            // Clause-level pause after the head: it's almost always cut at a
            // comma or mid-clause whitespace, not at sentence end.
            pauseAfterMs: pauseClauseMs)

        // Re-chunk the rest of sentence 1 + every following sentence using the
        // normal pipeline. This keeps merging logic + per-sentence pause
        // assignment intact for everything after the streaming head.
        let remainder = ([t] + sentences.dropFirst()).joined(separator: " ")
        let tailChunks = chunk(text: remainder, language: language)
        return [headChunk] + tailChunks
    }

    /// Split `text` into chunks, each estimated to produce ≤ `maxCodesPerChunk`
    /// codec frames. Order is preserved.
    public static func chunk(
        text: String,
        language: MagpieLanguage
    ) -> [MagpieTextChunk] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        // 1. Sentence-tokenize.
        let sentences = splitSentences(trimmed, language: language)
        guard !sentences.isEmpty else { return [] }

        // 2. Estimate + merge short adjacent sentences.
        let merged = mergeShortSentences(sentences, language: language)

        // 3. Split too-long sentences on punctuation / connectors / whitespace.
        var output: [MagpieTextChunk] = []
        output.reserveCapacity(merged.count)
        for sentence in merged {
            let pieces = splitIfTooLong(sentence, language: language)
            output.append(contentsOf: pieces)
        }
        return output
    }

    // MARK: - Step 1: NaturalLanguage sentence tokenization

    private static func splitSentences(_ text: String, language: MagpieLanguage) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        if let nl = nlLanguage(for: language) {
            tokenizer.setLanguage(nl)
        }
        tokenizer.string = text

        var out: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let s = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty { out.append(s) }
            return true
        }
        return out
    }

    private static func nlLanguage(for language: MagpieLanguage) -> NLLanguage? {
        switch language {
        case .english: return .english
        case .spanish: return .spanish
        case .german: return .german
        case .french: return .french
        case .italian: return .italian
        case .vietnamese: return .vietnamese
        case .mandarin: return .simplifiedChinese
        case .hindi: return .hindi
        }
    }

    // MARK: - Step 2: Merge short adjacent sentences

    private static func mergeShortSentences(
        _ sentences: [String], language: MagpieLanguage
    ) -> [String] {
        let ratio = codesPerChar(language)
        let mergeTarget = maxCodesPerChunk * 3 / 4  // 165 codes ≈ 7.6 s
        var out: [String] = []
        for s in sentences {
            let est = Int((Double(s.count) * ratio).rounded(.up))
            if let last = out.last {
                let lastEst = Int((Double(last.count) * ratio).rounded(.up))
                let combined = lastEst + est
                // Merge greedily while there's room: either neighbor is short,
                // OR combined fits inside the soft target (avoids pointless
                // single-sentence chunks when several easily fit together).
                let bothFit = combined <= maxCodesPerChunk
                let oneIsShort = lastEst < mergeBelowCodes || est < mergeBelowCodes
                let underSoftTarget = combined <= mergeTarget
                if bothFit && (oneIsShort || underSoftTarget) {
                    out[out.count - 1] = last + " " + s
                    continue
                }
            }
            out.append(s)
        }
        return out
    }

    // MARK: - Step 3: Split too-long sentences

    /// Split delimiters tried in order. Each pass keeps the delimiter attached
    /// to the preceding fragment so prosody hints (commas, dashes) survive.
    private static let punctuationSplitChars: [Character] = [",", ";", ":", "—", "–", "，", "；", "："]
    private static let connectorPhrases: [String] = [
        " and ", " but ", " or ", " because ", " however ", " therefore ",
        " so ", " while ", " although ", " though ",
    ]

    private static func splitIfTooLong(
        _ sentence: String, language: MagpieLanguage
    ) -> [MagpieTextChunk] {
        let ratio = codesPerChar(language)
        let estimate: (String) -> Int = { Int((Double($0.count) * ratio).rounded(.up)) }

        if estimate(sentence) <= maxCodesPerChunk {
            return [makeChunk(sentence, codes: estimate(sentence))]
        }

        // Try punctuation splits.
        var pieces = splitOn(sentence, where: { punctuationSplitChars.contains($0) })
        if allFit(pieces, estimate: estimate) {
            return pieces.map { makeChunk($0, codes: estimate($0)) }
        }

        // Try connector phrases on each piece that's still too long.
        pieces = pieces.flatMap { piece -> [String] in
            if estimate(piece) <= maxCodesPerChunk { return [piece] }
            return splitOnPhrases(piece, phrases: connectorPhrases)
        }
        if allFit(pieces, estimate: estimate) {
            return pieces.map { makeChunk($0, codes: estimate($0)) }
        }

        // Last resort: whitespace split at the codes-budget boundary.
        pieces = pieces.flatMap { piece -> [String] in
            if estimate(piece) <= maxCodesPerChunk { return [piece] }
            return splitOnWhitespace(piece, ratio: ratio)
        }
        return pieces.map { makeChunk($0, codes: estimate($0)) }
    }

    private static func allFit(
        _ pieces: [String], estimate: (String) -> Int
    ) -> Bool {
        pieces.allSatisfy { estimate($0) <= maxCodesPerChunk }
    }

    /// Splits keeping the delimiter attached to the preceding fragment.
    private static func splitOn(
        _ s: String, where isDelimiter: (Character) -> Bool
    ) -> [String] {
        var out: [String] = []
        var current = ""
        for ch in s {
            current.append(ch)
            if isDelimiter(ch) {
                let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { out.append(trimmed) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        return out
    }

    private static func splitOnPhrases(_ s: String, phrases: [String]) -> [String] {
        // Find the phrase nearest the middle that gives the most balanced split.
        let lowered = s.lowercased()
        var best: (start: String.Index, end: String.Index)?
        var bestImbalance = Int.max
        for phrase in phrases {
            var searchStart = lowered.startIndex
            while let range = lowered.range(of: phrase, range: searchStart..<lowered.endIndex) {
                let leftLen = lowered.distance(from: lowered.startIndex, to: range.lowerBound)
                let rightLen = lowered.distance(from: range.upperBound, to: lowered.endIndex)
                let imbalance = abs(leftLen - rightLen)
                if imbalance < bestImbalance {
                    bestImbalance = imbalance
                    best = (range.lowerBound, range.upperBound)
                }
                searchStart = range.upperBound
            }
        }
        guard let split = best else { return [s] }
        // Map indices from `lowered` back into `s` (same UTF-8 length per char,
        // but lowercasing can change byte length, so use distance-based mapping).
        let leftDist = lowered.distance(from: lowered.startIndex, to: split.start)
        let rightDist = lowered.distance(from: lowered.startIndex, to: split.end)
        let leftIdx = s.index(s.startIndex, offsetBy: leftDist)
        let rightIdx = s.index(s.startIndex, offsetBy: rightDist)
        let left = String(s[s.startIndex..<leftIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
        let right = String(s[rightIdx..<s.endIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
        return [left, right].filter { !$0.isEmpty }
    }

    private static func splitOnWhitespace(_ s: String, ratio: Double) -> [String] {
        let words = s.split(whereSeparator: { $0.isWhitespace })
        guard !words.isEmpty else { return [s] }
        var out: [String] = []
        var current = ""
        var currentCodes = 0
        for w in words {
            let wordCodes = Int((Double(w.count + 1) * ratio).rounded(.up))
            if currentCodes + wordCodes > maxCodesPerChunk, !current.isEmpty {
                out.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = ""
                currentCodes = 0
            }
            if !current.isEmpty { current.append(" ") }
            current.append(String(w))
            currentCodes += wordCodes
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        // Avoid orphan tail: if the last piece is tiny, fold it into the
        // previous one as long as the combined fragment still fits.
        if out.count >= 2 {
            let lastCount = out[out.count - 1].count
            let prevCount = out[out.count - 2].count
            let lastEst = Int((Double(lastCount) * ratio).rounded(.up))
            let prevEst = Int((Double(prevCount) * ratio).rounded(.up))
            if lastEst < mergeBelowCodes && lastEst + prevEst <= maxCodesPerChunk {
                out[out.count - 2] = out[out.count - 2] + " " + out[out.count - 1]
                out.removeLast()
            }
        }
        return out
    }

    // MARK: - Pause assignment

    private static func makeChunk(_ text: String, codes: Int) -> MagpieTextChunk {
        MagpieTextChunk(text: text, estimatedCodes: codes, pauseAfterMs: pauseAfterMs(text))
    }

    private static func pauseAfterMs(_ text: String) -> Int {
        guard let last = text.last else { return pauseDefaultMs }
        if last == "\n" { return pauseParagraphMs }
        if "。？！.?!".contains(last) { return pauseSentenceMs }
        if ",;:，；：".contains(last) { return pauseClauseMs }
        return pauseDefaultMs
    }
}
