import Foundation
import OSLog

/// A chunk of text prepared for synthesis.
/// - words: Tokenised words used for phoneme lookup
/// - atoms: Original text segments (words and punctuation) used to rebuild chunk text
/// - phonemes: Flat phoneme sequence with single-space separators between words
/// - totalFrames: Reserved for legacy/frame-aware modes (unused here)
/// - pauseAfterMs: Silence to insert after this chunk (punctuation/paragraph driven)
struct TextChunk {
    let words: [String]
    let atoms: [String]
    let phonemes: [String]
    let totalFrames: Float
    let pauseAfterMs: Int

    var text: String {
        atoms.reduce(into: "") { partialResult, atom in
            partialResult = KokoroChunker.appendSegment(partialResult, with: atom)
        }
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Punctuation-aware chunker that splits paragraphs → sentences → clauses
/// and packs them under the model token budget.
enum KokoroChunker {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroChunker")
    private static let noPrespaceCharacters: Set<Character> = [
        ",", ";", ":", "!", "?", ".", "…", "—", "–", "'", "\"", ")", "]", "}", "”", "’"
    ]
    private static let decimalDigits = CharacterSet.decimalDigits
    private static let spelledOutFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.numberStyle = .spellOut
        formatter.maximumFractionDigits = 0
        formatter.roundingMode = .down
        return formatter
    }()

    static func appendSegment(_ base: String, with next: String) -> String {
        let trimmedNext = next.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedNext.isEmpty else { return base }
        if base.isEmpty {
            return trimmedNext
        }
        if let first = trimmedNext.first, noPrespaceCharacters.contains(first) {
            return base + trimmedNext
        }
        return base + " " + trimmedNext
    }

    private static func spelledOutTokens(for token: String) -> [String]? {
        guard !token.isEmpty else { return nil }
        if token.rangeOfCharacter(from: decimalDigits.inverted) != nil {
            return nil
        }
        guard let value = Int(token) else { return nil }
        guard let spelled = spelledOutFormatter.string(from: NSNumber(value: value)) else { return nil }
        let separators = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "-"))
        let parts = spelled
            .lowercased()
            .components(separatedBy: separators)
            .filter { !$0.isEmpty }
        return parts.isEmpty ? nil : parts
    }
    /// Build chunks under the token budget.
    /// - Parameters:
    ///   - text: Raw input text
    ///   - wordToPhonemes: Mapping of lowercase words to phoneme arrays
    ///   - targetTokens: Model token capacity for `input_ids`
    ///   - hasLanguageToken: Whether to reserve one token for a language marker
    /// - Returns: Array of TextChunk with punctuation-driven `pauseAfterMs`
    static func chunk(
        text: String,
        wordToPhonemes: [String: [String]],
        targetTokens: Int,
        hasLanguageToken: Bool
    ) -> [TextChunk] {
        let baseOverhead = 2 + (hasLanguageToken ? 1 : 0)  // BOS/EOS + optional language token
        // Safety margin to avoid hard edge near model capacity (prevents mid-sentence cut-offs)
        let safety = 12
        let cap = max(1, targetTokens - safety)

        let vocabulary = KokoroVocabulary.getVocabulary()
        let punctuationTokensToStrip: Set<String> = [":", ";", ","]
        let punctuationEndingTokens: Set<String> = [".", "!", "?", "…"]
        // Conjunctions & prepositions we try not to end a chunk on (keep in lowercase)
        let boundaryStopWords: Set<String> = [
            // Coordinating conjunctions
            "and", "but", "or", "nor", "so", "yet", "for",
            // Common prepositions
            "with", "without", "about", "above", "below", "across",
            "after", "before", "into", "onto", "upon", "between",
            "among", "around", "near", "to", "of", "in", "on",
            "at", "by", "from", "over", "under", "through",
            // Subordinating conjunctions / prepositional variants
            "during", "regarding", "concerning", "because", "since",
            "though", "although", "while"
        ]
        let periodTokenAvailable = vocabulary["."] != nil

        // Pause mapping (ms)
        let pauseSentence = 300
        let pauseClause = 150
        let pauseParagraph = 500

        let normalizationSet = CharacterSet.letters.union(.decimalDigits).union(CharacterSet(charactersIn: "'"))

        func normalizeWord(_ s: String) -> String {
            let lowered = s.lowercased()
                .replacingOccurrences(of: "\u{2019}", with: "'")
                .replacingOccurrences(of: "\u{2018}", with: "'")
            return String(lowered.unicodeScalars.filter { normalizationSet.contains($0) })
        }

        func phonemizeWords(_ words: [String]) -> [String] {
            let allowed = Set(vocabulary.keys)

            func isPunct(_ ch: Character) -> Bool {
                return !(ch.isLetter || ch.isNumber || ch.isWhitespace || ch == "'")
            }

            var out: [String] = []
            for (wi, w) in words.enumerated() {
                // Split w into runs: word segments and single punctuation chars
                var seg = ""
                func flushSeg() {
                    guard !seg.isEmpty else { return }
                    let key = normalizeWord(seg)
                    if let arr = wordToPhonemes[key] {
                        out.append(contentsOf: arr)
                    } else {
                        var handled = false
                        if let spelledTokens = spelledOutTokens(for: key) {
                            var spelledPhonemes: [String] = []
                            var success = true
                            for spelled in spelledTokens {
                                if let arr = wordToPhonemes[spelled] {
                                    if !spelledPhonemes.isEmpty { spelledPhonemes.append(" ") }
                                    spelledPhonemes.append(contentsOf: arr)
                                } else {
                                    #if canImport(ESpeakNG) || canImport(CEspeakNG)
                                    if let ipa = EspeakG2P.shared.phonemize(word: spelled) {
                                        let mapped = PhonemeMapper.mapIPA(ipa, allowed: allowed)
                                        if !mapped.isEmpty {
                                            if !spelledPhonemes.isEmpty { spelledPhonemes.append(" ") }
                                            spelledPhonemes.append(contentsOf: mapped)
                                            KokoroChunker.logger.info("EspeakG2P used for spelled-out token: \(spelled)")
                                        } else {
                                            success = false
                                            break
                                        }
                                    } else {
                                        success = false
                                        break
                                    }
                                    #else
                                    success = false
                                    break
                                    #endif
                                }
                            }
                            if success && !spelledPhonemes.isEmpty {
                                out.append(contentsOf: spelledPhonemes)
                                seg.removeAll()
                                handled = true
                            }
                        }
                        if handled { return }
                        // Use C eSpeak NG for OOV phonemization only (if available)
                        #if canImport(ESpeakNG) || canImport(CEspeakNG)
                        if let ipa = EspeakG2P.shared.phonemize(word: key) {
                            let mapped = PhonemeMapper.mapIPA(ipa, allowed: allowed)
                            if !mapped.isEmpty {
                                print(
                                    "[G2P] word=\(key) | ipa=\(ipa.joined(separator: " ")) | map=\(mapped.joined(separator: " "))"
                                )
                                out.append(contentsOf: mapped)
                                KokoroChunker.logger.info("EspeakG2P used for OOV word: \(key)")
                            } else {
                                print("[G2P] word=\(key) | ipa=<none> | map=<empty>")
                                KokoroChunker.logger.warning("OOV word yielded no mappable IPA tokens: \(key)")
                            }
                        } else {
                            print("[G2P] word=\(key) | ipa=<failed> | map=<empty>")
                            KokoroChunker.logger.warning("EspeakG2P failed for OOV word: \(key)")
                        }
                        #else
                        // G2P not available in this build; skip OOV word
                        KokoroChunker.logger.warning("CEspeakNG not available; skipping OOV word: \(key)")
                        #endif
                    }
                    seg.removeAll()
                }
                for ch in w {
                    if isPunct(ch) {
                        flushSeg()
                        let p = String(ch)
                        if allowed.contains(p) { out.append(p) }
                    } else {
                        seg.append(ch)
                    }
                }
                flushSeg()
                if wi != words.count - 1 { out.append(" ") }
            }
            if out.last == " " { out.removeLast() }
            return out
        }

        func appendPeriod(to atoms: inout [String]) {
            guard !atoms.isEmpty else {
                atoms.append(".")
                return
            }
            var updated = atoms[atoms.count - 1]
            while let last = updated.last, last.isWhitespace {
                updated.removeLast()
            }
            while let last = updated.last, punctuationTokensToStrip.contains(String(last)) {
                updated.removeLast()
            }
            if updated.isEmpty {
                atoms[atoms.count - 1] = "."
            } else {
                atoms[atoms.count - 1] = updated + "."
            }
        }

        /// Remove trailing whitespace tokens and suppress terminal colon/semicolon/comma tokens so
        /// punctuation pauses do not produce artifacts in the synthesised audio. Optionally appends
        /// a terminating period when a chunk ends on a conjunction or preposition (if budget allows).
        func sanitizeChunkTokens(
            _ tokens: [String],
            lastWord: String?,
            currentTokenCount: Int
        ) -> ([String], Bool) {
            var cleaned = tokens
            var appendedPeriod = false
            var replacedPunctuation = false
            while let last = cleaned.last, last == " " {
                cleaned.removeLast()
            }
            if let last = cleaned.last, punctuationTokensToStrip.contains(last) {
                if periodTokenAvailable {
                    cleaned[cleaned.count - 1] = "."
                    appendedPeriod = true
                    replacedPunctuation = true
                } else {
                    cleaned.removeLast()
                    while let tail = cleaned.last, tail == " " {
                        cleaned.removeLast()
                    }
                }
            }

            let currentlyTerminated = cleaned.last.map { punctuationEndingTokens.contains($0) || punctuationTokensToStrip.contains($0) } ?? false

            if periodTokenAvailable {
                let normalized = lastWord.map { normalizeWord($0) }
                if let normalized,
                    !normalized.isEmpty,
                    boundaryStopWords.contains(normalized)
                {
                    let alreadyTerminated = currentlyTerminated
                    let additionalCost = appendedPeriod ? 0 : 1
                    if alreadyTerminated == false && currentTokenCount + additionalCost <= targetTokens {
                        cleaned.append(".")
                        appendedPeriod = true
                    }
                }
            } else if !currentlyTerminated, !replacedPunctuation {
                // If we removed punctuation and cannot emit '.', the chunk will end without added token.
            }

            return (cleaned, appendedPeriod)
        }

        func splitSentences(_ paragraph: String) -> [(String, Character?)] {
            var units: [(String, Character?)] = []
            var buf = ""
            for ch in paragraph {
                buf.append(ch)
                if ch == "." || ch == "!" || ch == "?" {
                    units.append((buf.trimmingCharacters(in: .whitespaces), ch))
                    buf = ""
                }
            }
            let tail = buf.trimmingCharacters(in: .whitespaces)
            if !tail.isEmpty { units.append((tail, nil)) }
            return units
        }

        func splitClauses(_ sentence: String, endPunct: Character?) -> [(String, Int)] {
            var parts: [(String, Int)] = []
            var buf = ""
            for ch in sentence {
                if ch == "," || ch == ";" || ch == ":" {
                    let trimmed = buf.trimmingCharacters(in: .whitespaces)
                    if !trimmed.isEmpty {
                        let segment = trimmed + String(ch)
                        parts.append((segment, pauseClause))
                        KokoroChunker.logger.info("    Split at '\(ch)': added '\(segment)'")
                    }
                    buf = ""
                } else {
                    buf.append(ch)
                }
            }
            let last = buf.trimmingCharacters(in: .whitespaces)
            if !last.isEmpty {
                parts.append((last, endPunct != nil ? pauseSentence : 0))
                KokoroChunker.logger.info("    Final part: '\(last)'")
            }
            return parts
        }

        let paragraphs = text.components(separatedBy: "\n\n").filter {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }

        var chunks: [TextChunk] = []
        for (pIndex, para) in paragraphs.enumerated() {
            let sentences = splitSentences(para)
            var units: [(text: String, words: [String], displayWords: [String], pause: Int)] = []
            for (sent, endP) in sentences {
                let clauses = splitClauses(sent, endPunct: endP)
                KokoroChunker.logger.info("Sentence: '\(sent)' -> \(clauses.count) clauses")
                for (idx, (cl, pause)) in clauses.enumerated() {
                    let trimmed = cl.trimmingCharacters(in: .whitespaces)
                    guard !trimmed.isEmpty else { continue }
                    let displayWords = trimmed.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
                    let words = displayWords.map { $0.lowercased() }
                    if !words.isEmpty {
                        units.append((text: trimmed, words: words, displayWords: displayWords, pause: pause))
                        KokoroChunker.logger.info(
                            "  Clause \(idx): '\(trimmed)' -> \(words.count) words, pause=\(pause)")
                    }
                }
            }

            var curWords: [String] = []
            var curAtoms: [String] = []
            var curPhon: [String] = []
            var curTokenCount = baseOverhead
            var lastPause = pIndex < paragraphs.count - 1 ? pauseParagraph : 0

            for unit in units {
                let ph = phonemizeWords(unit.words)
                // If the entire unit doesn't fit into an empty chunk, split by words to respect budget.
                let fitsEmpty = (baseOverhead + ph.count) <= cap

                if ph.isEmpty {
                    // Unit has no known phonemes; treat as zero-cost. Skip, but carry over pause.
                    lastPause = unit.pause
                    continue
                }

                if fitsEmpty == false {
                    // First, flush any existing buffer before micro-splitting
                    if !curPhon.isEmpty {
                        let (cleaned, appendedPeriod) = sanitizeChunkTokens(
                            curPhon,
                            lastWord: curWords.last,
                            currentTokenCount: curTokenCount)
                        if appendedPeriod { appendPeriod(to: &curAtoms) }
                        if !cleaned.isEmpty {
                            chunks.append(
                                TextChunk(
                                    words: curWords,
                                    atoms: curAtoms,
                                    phonemes: cleaned,
                                    totalFrames: 0,
                                    pauseAfterMs: lastPause))
                        }
                        curWords.removeAll()
                        curAtoms.removeAll()
                        curPhon.removeAll()
                        curTokenCount = baseOverhead
                    }

                    // Micro-split: accumulate word-by-word within this unit.
                    var subWords: [String] = []
                    var subAtoms: [String] = []
                    var subPhon: [String] = []
                    var subCount = baseOverhead

                    func flushSub(finalPause: Int) {
                        if !subPhon.isEmpty {
                            let (cleaned, appendedPeriod) = sanitizeChunkTokens(
                                subPhon,
                                lastWord: subWords.last,
                                currentTokenCount: subCount)
                            if appendedPeriod { appendPeriod(to: &subAtoms) }
                            if !cleaned.isEmpty {
                                chunks.append(
                                    TextChunk(
                                        words: subWords,
                                        atoms: subAtoms,
                                        phonemes: cleaned,
                                        totalFrames: 0,
                                        pauseAfterMs: finalPause))
                            }
                            subWords.removeAll(keepingCapacity: true)
                            subAtoms.removeAll(keepingCapacity: true)
                            subPhon.removeAll(keepingCapacity: true)
                            subCount = baseOverhead
                        }
                    }

                    for (i, w) in unit.words.enumerated() {
                        let displayWord = unit.displayWords[i]
                        let wPh = phonemizeWords([w])
                        // Fallback cost for unknown words: treat as small cost so we still split; no phonemes appended.
                        let wCost = (wPh.isEmpty ? 1 : wPh.count) + (subPhon.isEmpty ? 0 : 1)
                        if subCount + wCost <= cap {
                            if !wPh.isEmpty {
                                if !subPhon.isEmpty { subPhon.append(" ") }
                                subPhon.append(contentsOf: wPh)
                            }
                            subWords.append(w)
                            subAtoms.append(displayWord)
                            subCount += wCost
                        } else {
                            // flush with no internal pause between sub-chunks
                            flushSub(finalPause: 0)
                            if !wPh.isEmpty {
                                subPhon.append(contentsOf: wPh)
                            }
                            subWords.append(w)
                            subAtoms.append(displayWord)
                            subCount = baseOverhead + (wPh.isEmpty ? 1 : wPh.count)
                        }
                        // At end of unit, flush with the unit's pause
                        if i == unit.words.count - 1 {
                            flushSub(finalPause: unit.pause)
                            lastPause = unit.pause
                        }
                    }
                    continue
                }

                // Normal packing at unit granularity
                let additional = ph.count + (curPhon.isEmpty ? 0 : 1)  // +1 for join space
                if curTokenCount + additional <= cap {
                    if !curPhon.isEmpty { curPhon.append(" ") }
                    curPhon.append(contentsOf: ph)
                    curWords.append(contentsOf: unit.words)
                    curAtoms.append(unit.text)
                    curTokenCount += additional
                    lastPause = unit.pause
                } else {
                    if !curPhon.isEmpty {
                        let (cleaned, appendedPeriod) = sanitizeChunkTokens(
                            curPhon,
                            lastWord: curWords.last,
                            currentTokenCount: curTokenCount)
                        if appendedPeriod { appendPeriod(to: &curAtoms) }
                        if !cleaned.isEmpty {
                            chunks.append(
                                TextChunk(
                                    words: curWords,
                                    atoms: curAtoms,
                                    phonemes: cleaned,
                                    totalFrames: 0,
                                    pauseAfterMs: lastPause))
                        }
                    }
                    curWords = unit.words
                    curAtoms = [unit.text]
                    curPhon = ph
                    curTokenCount = baseOverhead + ph.count
                    lastPause = unit.pause
                }
            }
            if !curPhon.isEmpty {
                let (cleaned, appendedPeriod) = sanitizeChunkTokens(
                    curPhon,
                    lastWord: curWords.last,
                    currentTokenCount: curTokenCount)
                guard !cleaned.isEmpty else { continue }
                if appendedPeriod { appendPeriod(to: &curAtoms) }
                let paraPause = (pIndex < paragraphs.count - 1) ? pauseParagraph : lastPause
                chunks.append(
                    TextChunk(
                        words: curWords,
                        atoms: curAtoms,
                        phonemes: cleaned,
                        totalFrames: 0,
                        pauseAfterMs: paraPause))
            }
        }
        return chunks
    }
}
