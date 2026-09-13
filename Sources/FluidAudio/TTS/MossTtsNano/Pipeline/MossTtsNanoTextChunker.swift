import Foundation

/// Sentence / clause / token-budget chunker for voice-clone synthesis.
///
/// Port of the upstream `_prepare_text_for_sentence_chunking` +
/// `_split_text_into_best_sentences` flow: text is normalized (line breaks →
/// spaces, capitalized, terminal punctuation ensured), split at sentence-ending
/// punctuation, then clauses, then hard token-budget cuts, and finally re-packed
/// greedily so each chunk stays within `maxTokens` tokenizer tokens.
enum MossTtsNanoTextChunker {

    static let sentenceEnd: Set<Character> = [".", "!", "?", "。", "！", "？", "；", ";"]
    static let clauseSplit: Set<Character> = [",", "，", "、", "；", ";", "：", ":"]
    static let closing: Set<Character> = ["\"", "'", "”", "’", ")", "]", "}", "）", "】", "》", "」", "』"]

    static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { s in
            (0x4E00...0x9FFF).contains(s.value) || (0x3400...0x4DBF).contains(s.value)
                || (0x3040...0x30FF).contains(s.value) || (0xAC00...0xD7AF).contains(s.value)
        }
    }

    /// Upstream `_prepare_text_for_sentence_chunking` (minus its leading-space
    /// padding for short texts, which the tokenizer's whitespace stripping makes a
    /// no-op).
    static func prepare(_ text: String) throws -> String {
        var normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { throw MossTtsNanoError.emptyText }
        normalized = normalized.replacingOccurrences(of: "\n", with: " ").replacingOccurrences(of: "\r", with: " ")
        while normalized.contains("  ") {
            normalized = normalized.replacingOccurrences(of: "  ", with: " ")
        }
        guard let last = normalized.last else { throw MossTtsNanoError.emptyText }
        if containsCJK(normalized) {
            if !sentenceEnd.contains(last) { normalized.append("。") }
            return normalized
        }
        if let first = normalized.first, !first.isUppercase {
            normalized = String(first).uppercased() + normalized.dropFirst()
        }
        if last.isLetter || last.isNumber {
            normalized.append(".")
        }
        return normalized
    }

    /// Upstream `_split_text_by_punctuation`: cut after each punctuation character
    /// (keeping trailing closing quotes/brackets), dropping inter-sentence whitespace.
    static func split(_ text: String, at punctuation: Set<Character>) -> [String] {
        let chars = Array(text)
        var pieces: [String] = []
        var current: [Character] = []
        var index = 0
        while index < chars.count {
            let ch = chars[index]
            current.append(ch)
            if punctuation.contains(ch) {
                var lookahead = index + 1
                while lookahead < chars.count, closing.contains(chars[lookahead]) {
                    current.append(chars[lookahead])
                    lookahead += 1
                }
                let sentence = String(current).trimmingCharacters(in: .whitespaces)
                if !sentence.isEmpty { pieces.append(sentence) }
                current = []
                while lookahead < chars.count, chars[lookahead].isWhitespace {
                    lookahead += 1
                }
                index = lookahead
                continue
            }
            index += 1
        }
        let tail = String(current).trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty { pieces.append(tail) }
        return pieces
    }

    static func join(_ left: String, _ right: String) -> String {
        if left.isEmpty { return right }
        if right.isEmpty { return left }
        if containsCJK(left) || containsCJK(right) { return left + right }
        return left + " " + right
    }

    /// Upstream `_split_text_by_token_budget`: binary-search the longest prefix that
    /// fits, then back off (≤ 24 chars) to a punctuation/space boundary.
    static func splitByBudget(_ text: String, maxTokens: Int, count: (String) -> Int) -> [String] {
        var remaining = text.trimmingCharacters(in: .whitespaces)
        guard !remaining.isEmpty else { return [] }
        let boundary = clauseSplit.union(sentenceEnd).union([" "])
        var pieces: [String] = []
        while !remaining.isEmpty {
            if count(remaining) <= maxTokens {
                pieces.append(remaining)
                break
            }
            let chars = Array(remaining)
            var low = 1
            var high = chars.count
            var bestPrefix = 1
            while low <= high {
                let middle = (low + high) / 2
                let candidate = String(chars[0..<middle]).trimmingCharacters(in: .whitespaces)
                if candidate.isEmpty {
                    low = middle + 1
                    continue
                }
                if count(candidate) <= maxTokens {
                    bestPrefix = middle
                    low = middle + 1
                } else {
                    high = middle - 1
                }
            }
            var cut = bestPrefix
            let scanFloor = max(-1, bestPrefix - 25)
            var scan = bestPrefix - 1
            while scan > scanFloor {
                if boundary.contains(chars[scan]) {
                    cut = scan + 1
                    break
                }
                scan -= 1
            }
            var piece = String(chars[0..<cut]).trimmingCharacters(in: .whitespaces)
            if piece.isEmpty {
                piece = String(chars[0..<bestPrefix]).trimmingCharacters(in: .whitespaces)
                cut = bestPrefix
            }
            pieces.append(piece)
            remaining = String(chars[cut...]).trimmingCharacters(in: .whitespaces)
        }
        return pieces
    }

    /// Upstream `_split_text_into_best_sentences`. `maxTokens <= 0` returns the text as
    /// a single chunk (after `prepare`).
    static func chunk(_ text: String, maxTokens: Int, count: (String) -> Int) throws -> [String] {
        let prepared = try prepare(text)
        if maxTokens <= 0 { return [prepared] }

        var sentences = split(prepared, at: sentenceEnd)
        if sentences.isEmpty { sentences = [prepared] }

        var slices: [(tokens: Int, text: String)] = []
        for sentence in sentences {
            let s = sentence.trimmingCharacters(in: .whitespaces)
            guard !s.isEmpty else { continue }
            let n = count(s)
            if n <= maxTokens {
                slices.append((n, s))
                continue
            }
            var clauses = split(s, at: clauseSplit)
            if clauses.count <= 1 { clauses = [s] }
            for clause in clauses {
                let c = clause.trimmingCharacters(in: .whitespaces)
                guard !c.isEmpty else { continue }
                let cn = count(c)
                if cn <= maxTokens {
                    slices.append((cn, c))
                    continue
                }
                for piece in splitByBudget(c, maxTokens: maxTokens, count: count) {
                    let p = piece.trimmingCharacters(in: .whitespaces)
                    if !p.isEmpty { slices.append((count(p), p)) }
                }
            }
        }

        var chunks: [String] = []
        var current = ""
        var currentTokens = 0
        for (n, s) in slices {
            if current.isEmpty {
                current = s
                currentTokens = n
                continue
            }
            if currentTokens + n > maxTokens {
                chunks.append(current.trimmingCharacters(in: .whitespaces))
                current = s
                currentTokens = n
            } else {
                current = join(current, s)
                currentTokens = count(current)
            }
        }
        if !current.isEmpty { chunks.append(current.trimmingCharacters(in: .whitespaces)) }
        return chunks.isEmpty ? [prepared] : chunks
    }

    /// Upstream `_estimate_voice_clone_inter_chunk_pause_seconds`.
    static func pauseSeconds(after chunk: String) -> Float {
        let words = chunk.split(whereSeparator: { $0.isWhitespace }).count
        return words <= 4 ? MossTtsNanoConstants.interChunkPauseShort : MossTtsNanoConstants.interChunkPauseLong
    }
}
