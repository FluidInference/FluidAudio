import FluidAudio
import Foundation
import NaturalLanguage

/// Text chunking system for Kokoro TTS that segments input text into optimal chunks based on token capacity limits.
/// Handles sentence splitting, phoneme resolution, and ensures each chunk fits within the model's processing constraints.

/// Lightweight chunk representation passed into Kokoro synthesis.
struct TextChunk: Sendable {
    let words: [String]
    let atoms: [String]
    let phonemes: [String]
    let totalFrames: Float
    let pauseAfterMs: Int
    let text: String
}

/// Splits normalized input text into Kokoro-friendly segments: sentence tokenization,
/// punctuation-aware merging, and phoneme lookup ensure each chunk stays within the model’s
/// token capacity before synthesis.
enum KokoroChunker {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroChunker")
    private static let decimalDigits = CharacterSet.decimalDigits
    private static let apostropheCharacters: Set<Character> = ["'", "’", "ʼ", "‛", "‵", "′"]

    private static func isWordCharacter(_ character: Character) -> Bool {
        if character.isLetter || character.isNumber || apostropheCharacters.contains(character) {
            return true
        }

        return character.unicodeScalars.contains { scalar in
            scalar.properties.isEmojiPresentation || scalar.properties.isEmoji
        }
    }
    /// Public entry point used by `KokoroSynthesizer`
    static func chunk(
        text: String,
        wordToPhonemes: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        customLexicon: TtsCustomLexicon?,
        targetTokens: Int,
        hasLanguageToken: Bool,
        allowedPhonemes: Set<String>,
        phoneticOverrides: [TtsPhoneticOverride]
    ) throws -> [TextChunk] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let capacity = computeCapacity(targetTokens: targetTokens, hasLanguageToken: hasLanguageToken)

        let normalized = collapseNewlines(trimmed)

        let (sentences, _) = splitIntoSentences(normalized)
        guard !sentences.isEmpty else { return [] }

        let refinedSentences = sentences.compactMap { sentence in
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        guard !refinedSentences.isEmpty else {
            logger.info("Kokoro chunker produced no segments after refinement")
            return []
        }

        let mergedSentences = try mergeShortSentences(
            refinedSentences,
            lexicon: wordToPhonemes,
            caseSensitiveLexicon: caseSensitiveLexicon,
            customLexicon: customLexicon,
            allowed: allowedPhonemes,
            capacity: capacity
        )

        let segmentsByPeriods = mergedSentences.isEmpty ? refinedSentences : mergedSentences

        var segmentsByPunctuations: [String] = []
        segmentsByPunctuations.reserveCapacity(segmentsByPeriods.count)

        for (periodIndex, segment) in segmentsByPeriods.enumerated() {
            let count = try tokenCountForSegment(
                for: segment,
                lexicon: wordToPhonemes,
                caseSensitiveLexicon: caseSensitiveLexicon,
                customLexicon: customLexicon,
                allowed: allowedPhonemes,
                capacity: capacity
            )

            if count > capacity {
                let fragments = splitByPunctuation(segment)
                let reassembled = try reassembleFragments(
                    fragments,
                    lexicon: wordToPhonemes,
                    caseSensitiveLexicon: caseSensitiveLexicon,
                    customLexicon: customLexicon,
                    allowed: allowedPhonemes,
                    capacity: capacity
                )
                if !reassembled.isEmpty {
                    segmentsByPunctuations.append(contentsOf: reassembled)
                    continue
                }
                logger.warning(
                    "segmentsByPeriodsSplit[\(periodIndex)]: no punctuation-based split within capacity; deferring to chunk builder"
                )
            }

            segmentsByPunctuations.append(segment)
        }

        let sortedOverrides =
            phoneticOverrides
            .enumerated()
            .sorted { lhs, rhs in
                if lhs.element.wordIndex == rhs.element.wordIndex {
                    return lhs.offset < rhs.offset
                }
                return lhs.element.wordIndex < rhs.element.wordIndex
            }
            .map { $0.element }
        var overrideIndex = 0

        var globalWordIndex = 0
        var chunks: [TextChunk] = []
        chunks.reserveCapacity(segmentsByPunctuations.count)

        for chunkText in segmentsByPunctuations {
            let built = try buildChunks(
                from: chunkText,
                lexicon: wordToPhonemes,
                caseSensitiveLexicon: caseSensitiveLexicon,
                customLexicon: customLexicon,
                allowed: allowedPhonemes,
                capacity: capacity,
                wordIndex: &globalWordIndex,
                overrides: sortedOverrides,
                overrideIndex: &overrideIndex
            )
            chunks.append(contentsOf: built)
        }

        if overrideIndex < sortedOverrides.count {
            let remaining = sortedOverrides[overrideIndex...]
            let sample = remaining.prefix(5).map { $0.word }
            logger.warning("Unused phonetic overrides for words: \(sample.joined(separator: ", "))")
        }

        return chunks
    }

    private static func computeCapacity(targetTokens: Int, hasLanguageToken: Bool) -> Int {
        // Kokoro inputs prepend BOS, EOS, and optionally a language token, so reserve space for them.
        // A small safety margin keeps us under the model limit after merging and punctuation splits.
        let baseOverhead = 2 + (hasLanguageToken ? 1 : 0)
        let safety = 12
        return max(1, targetTokens - baseOverhead - safety)
    }

    // MARK: - Sentence Processing

    private static func splitIntoSentences(_ text: String) -> ([String], NLLanguage?) {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        let dominant = recognizer.dominantLanguage ?? .english

        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.setLanguage(dominant)

        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let candidate = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !candidate.isEmpty {
                sentences.append(candidate)
            }
            return true
        }
        if sentences.isEmpty {
            return ([text], dominant)
        }
        return (sentences, dominant)
    }

    private static func mergeShortSentences(
        _ sentences: [String],
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        customLexicon: TtsCustomLexicon?,
        allowed: Set<String>,
        capacity: Int
    ) throws -> [String] {
        guard !sentences.isEmpty else { return [] }

        let threshold = max(1, min(capacity, TtsConstants.shortSentenceMergeTokenThreshold))
        var merged: [String] = []
        var buffer: String = ""
        var bufferTokens = 0
        var didMerge = false

        func flushBuffer() {
            let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                merged.append(trimmed)
            }
            buffer.removeAll(keepingCapacity: false)
            bufferTokens = 0
        }

        for sentence in sentences {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            let sentenceTokens = try tokenCountForSegment(
                for: trimmed,
                lexicon: lexicon,
                caseSensitiveLexicon: caseSensitiveLexicon,
                customLexicon: customLexicon,
                allowed: allowed,
                capacity: capacity
            )

            if sentenceTokens > threshold {
                flushBuffer()
                merged.append(trimmed)
                continue
            }

            if buffer.isEmpty {
                buffer = trimmed
                bufferTokens = sentenceTokens
                continue
            }

            if bufferTokens > threshold {
                flushBuffer()
                buffer = trimmed
                bufferTokens = sentenceTokens
                continue
            }

            let candidate = appendSegment(buffer, with: trimmed)
            let candidateTokens = try tokenCountForSegment(
                for: candidate,
                lexicon: lexicon,
                caseSensitiveLexicon: caseSensitiveLexicon,
                customLexicon: customLexicon,
                allowed: allowed,
                capacity: capacity
            )

            if candidateTokens <= threshold {
                buffer = candidate
                bufferTokens = candidateTokens
                didMerge = true
            } else {
                flushBuffer()
                buffer = trimmed
                bufferTokens = sentenceTokens
            }
        }

        flushBuffer()

        if didMerge {
            logger.debug("Merged short sentences into \(merged.count) segments (threshold=\(threshold) tokens)")
        }

        return merged
    }

    // MARK: - Chunk Construction

    private static func buildChunks(
        from text: String,
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        customLexicon: TtsCustomLexicon?,
        allowed: Set<String>,
        capacity: Int,
        wordIndex: inout Int,
        overrides: [TtsPhoneticOverride],
        overrideIndex: inout Int
    ) throws -> [TextChunk] {
        let atoms = tokenizeAtoms(text)
        guard !atoms.isEmpty else { return [] }

        var chunks: [TextChunk] = []
        var chunkWords: [String] = []
        var chunkAtoms: [String] = []
        var chunkPhonemes: [String] = []
        var chunkTokenCount = 0
        var needsWordSeparator = false
        var missing: Set<String> = []
        var invalidCustomLexiconWords: Set<String> = []

        func flushChunk() {
            guard !chunkPhonemes.isEmpty else { return }
            if chunkPhonemes.last == " " {
                chunkPhonemes.removeLast()
                chunkTokenCount -= 1
            }
            let textValue = chunkAtoms.reduce(into: "") { partial, atom in
                partial = appendSegment(partial, with: atom)
            }.trimmingCharacters(in: .whitespacesAndNewlines)
            chunks.append(
                TextChunk(
                    words: chunkWords,
                    atoms: chunkAtoms,
                    phonemes: chunkPhonemes,
                    totalFrames: 0,
                    pauseAfterMs: 0,
                    text: textValue
                )
            )
            chunkWords.removeAll(keepingCapacity: true)
            chunkAtoms.removeAll(keepingCapacity: true)
            chunkPhonemes.removeAll(keepingCapacity: true)
            chunkTokenCount = 0
            needsWordSeparator = false
        }

        for atom in atoms {
            switch atom.kind {
            case .word(let original):
                let normalized = normalize(original)
                if normalized.isEmpty {
                    wordIndex += 1
                    continue
                }

                var resolved: [String]? = nil
                if overrideIndex < overrides.count {
                    while overrideIndex < overrides.count {
                        let candidate = overrides[overrideIndex]
                        if candidate.wordIndex < wordIndex {
                            logger.warning(
                                "Skipping stale phonetic override for word: \(candidate.word) (index \(candidate.wordIndex))"
                            )
                            overrideIndex += 1
                            continue
                        }
                        if candidate.wordIndex == wordIndex {
                            let overrideTokens = resolveOverride(candidate, allowed: allowed)
                            if overrideTokens.isEmpty {
                                logger.warning(
                                    "Phonetic override for word index \(wordIndex) (word: \(candidate.word)) produced no valid tokens; falling back to lexicon"
                                )
                            } else {
                                resolved = overrideTokens
                            }
                            overrideIndex += 1
                        }
                        break
                    }
                }

                // Check custom lexicon (highest priority after per-word overrides)
                if resolved == nil,
                    let customLexicon,
                    let customPhonemes = customLexicon.phonemes(for: original)
                {
                    let filtered = customPhonemes.filter { allowed.contains($0) }
                    if filtered.isEmpty {
                        if invalidCustomLexiconWords.insert(original).inserted {
                            let invalid = customPhonemes.filter { !allowed.contains($0) }
                            let preview = invalid.prefix(12).joined(separator: " ")
                            if preview.isEmpty {
                                logger.warning(
                                    "Custom lexicon entry for '\(original)' produced no usable phoneme tokens."
                                )
                            } else {
                                logger.warning(
                                    "Custom lexicon entry for '\(original)' has no tokens in Kokoro vocabulary "
                                        + "(first invalid: \(preview)). "
                                        + "Ensure your custom phonemes match the Kokoro vocabulary."
                                )
                            }
                        }
                    } else {
                        resolved = filtered
                    }
                }

                if resolved == nil {
                    guard
                        let fallback = try resolvePhonemes(
                            for: original,
                            normalized: normalized,
                            lexicon: lexicon,
                            caseSensitiveLexicon: caseSensitiveLexicon,
                            allowed: allowed,
                            missing: &missing
                        )
                    else {
                        wordIndex += 1
                        continue
                    }
                    resolved = fallback
                }

                guard let resolvedPhonemes = resolved, !resolvedPhonemes.isEmpty else {
                    wordIndex += 1
                    continue
                }

                var tokenCost = resolvedPhonemes.count
                if needsWordSeparator {
                    tokenCost += 1
                }

                if chunkTokenCount + tokenCost > capacity && !chunkPhonemes.isEmpty {
                    flushChunk()
                }

                if needsWordSeparator {
                    chunkPhonemes.append(" ")
                    chunkTokenCount += 1
                }

                chunkPhonemes.append(contentsOf: resolvedPhonemes)
                chunkTokenCount += resolvedPhonemes.count
                chunkWords.append(original)
                chunkAtoms.append(original)
                needsWordSeparator = true
                wordIndex += 1

            case .punctuation(let symbol):
                guard allowed.contains(symbol) else { continue }
                if chunkTokenCount + 1 > capacity && !chunkPhonemes.isEmpty {
                    flushChunk()
                }
                chunkPhonemes.append(symbol)
                chunkTokenCount += 1
                chunkAtoms.append(symbol)
                needsWordSeparator = false
            }
        }

        flushChunk()

        if !missing.isEmpty {
            logger.warning("Missing phoneme entries for: \(missing.sorted().joined(separator: ", "))")
        }

        return chunks
    }

    private enum AtomKind {
        case word(String)
        case punctuation(String)
    }

    private struct AtomToken {
        let text: String
        let kind: AtomKind
    }

    private static func tokenizeAtoms(_ text: String) -> [AtomToken] {
        var atoms: [AtomToken] = []
        var currentWord = ""

        func flushWord() {
            guard !currentWord.isEmpty else { return }
            let word = currentWord
            atoms.append(AtomToken(text: word, kind: .word(word)))
            currentWord.removeAll(keepingCapacity: true)
        }

        for ch in text {
            if ch.isWhitespace {
                flushWord()
                continue
            }

            if isWordCharacter(ch) {
                if apostropheCharacters.contains(ch) {
                    currentWord.append("'")
                } else {
                    currentWord.append(ch)
                }
                continue
            }

            flushWord()
            atoms.append(AtomToken(text: String(ch), kind: .punctuation(String(ch))))
        }

        flushWord()
        return atoms
    }

    private static func resolveOverride(
        _ override: TtsPhoneticOverride,
        allowed: Set<String>
    ) -> [String] {
        let tokens = override.tokens.filter { allowed.contains($0) }
        if !tokens.isEmpty {
            return tokens
        }

        let mappedFromTokens = PhonemeMapper.mapIPA(override.tokens, allowed: allowed)
        if !mappedFromTokens.isEmpty {
            return mappedFromTokens
        }

        if !override.scalarTokens.isEmpty {
            let mappedScalars = PhonemeMapper.mapIPA(override.scalarTokens, allowed: allowed)
            if !mappedScalars.isEmpty {
                return mappedScalars
            }
        }

        return []
    }

    private static func resolvePhonemes(
        for original: String,
        normalized: String,
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        allowed: Set<String>,
        missing: inout Set<String>
    ) throws -> [String]? {
        var phonemes = caseSensitiveLexicon[original]

        if phonemes == nil, let exactNormalized = caseSensitiveLexicon[normalized] {
            phonemes = exactNormalized
        }

        if phonemes == nil {
            phonemes = lexicon[normalized]
        }

        if phonemes == nil, let ipa = try EspeakG2P.shared.phonemize(word: normalized) {
            let mapped = PhonemeMapper.mapIPA(ipa, allowed: allowed)
            if !mapped.isEmpty {
                phonemes = mapped
            }
        }

        if phonemes == nil,
            let spelledTokens = spelledOutTokens(for: normalized),
            !spelledTokens.isEmpty
        {
            var spelledPhonemes: [String] = []
            var success = true
            var firstSegment = true
            for spelled in spelledTokens {
                var segment = lexicon[spelled]

                if segment == nil, let ipa = try EspeakG2P.shared.phonemize(word: spelled) {
                    let mapped = PhonemeMapper.mapIPA(ipa, allowed: allowed)
                    if !mapped.isEmpty {
                        segment = mapped
                    }
                }

                if segment == nil, let fallback = letterPronunciations[spelled] {
                    let filtered = fallback.filter { allowed.contains($0) }
                    if !filtered.isEmpty {
                        segment = filtered
                    }
                }

                guard var resolvedSegment = segment, !resolvedSegment.isEmpty else {
                    success = false
                    break
                }

                resolvedSegment = resolvedSegment.filter { allowed.contains($0) }
                if resolvedSegment.isEmpty {
                    success = false
                    break
                }

                if !firstSegment {
                    spelledPhonemes.append(" ")
                }
                spelledPhonemes.append(contentsOf: resolvedSegment)
                firstSegment = false
            }

            if success, !spelledPhonemes.isEmpty {
                phonemes = spelledPhonemes
            }
        }

        if phonemes == nil, let fallback = letterPronunciations[normalized] {
            let filtered = fallback.filter { allowed.contains($0) }
            if !filtered.isEmpty {
                phonemes = filtered
            }
        }

        guard var resolved = phonemes, !resolved.isEmpty else {
            missing.insert(normalized)
            return nil
        }

        resolved = resolved.filter { allowed.contains($0) }
        guard !resolved.isEmpty else {
            missing.insert(normalized)
            return nil
        }

        return resolved
    }

    private static func tokenCountForSegment(
        for text: String,
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        customLexicon: TtsCustomLexicon?,
        allowed: Set<String>,
        capacity: Int
    ) throws -> Int {
        let atoms = tokenizeAtoms(text)
        guard !atoms.isEmpty else { return 0 }

        var dummyMissing: Set<String> = []

        var tokenCount = 0
        var needsWordSeparator = false

        for atom in atoms {
            switch atom.kind {
            case .word(let original):
                let normalized = normalize(original)
                guard !normalized.isEmpty else { continue }

                // Check custom lexicon first
                var phonemes: [String]?
                if let customLexicon = customLexicon,
                    let customPhonemes = customLexicon.phonemes(for: original)
                {
                    let filtered = customPhonemes.filter { allowed.contains($0) }
                    if !filtered.isEmpty {
                        phonemes = filtered
                    }
                }

                if phonemes == nil {
                    phonemes = try resolvePhonemes(
                        for: original,
                        normalized: normalized,
                        lexicon: lexicon,
                        caseSensitiveLexicon: caseSensitiveLexicon,
                        allowed: allowed,
                        missing: &dummyMissing
                    )
                }

                guard let resolvedPhonemes = phonemes else {
                    continue
                }

                tokenCount += resolvedPhonemes.count
                if needsWordSeparator {
                    tokenCount += 1
                }
                needsWordSeparator = true
            case .punctuation(let symbol):
                guard allowed.contains(symbol) else { continue }
                tokenCount += 1
                needsWordSeparator = false
            }

            if tokenCount > capacity {
                return tokenCount
            }
        }

        return tokenCount
    }

    private static func reassembleFragments(
        _ fragments: [String],
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        customLexicon: TtsCustomLexicon?,
        allowed: Set<String>,
        capacity: Int
    ) throws -> [String] {
        guard !fragments.isEmpty else { return [] }

        var assembled: [String] = []
        var current = ""

        func flushCurrent() {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                assembled.append(trimmed)
            }
            current.removeAll(keepingCapacity: false)
        }

        for fragment in fragments {
            let trimmedFragment = fragment.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedFragment.isEmpty else { continue }

            let candidate =
                current.isEmpty
                ? trimmedFragment
                : appendSegment(current, with: trimmedFragment)
            let candidateTokens = try tokenCountForSegment(
                for: candidate,
                lexicon: lexicon,
                caseSensitiveLexicon: caseSensitiveLexicon,
                customLexicon: customLexicon,
                allowed: allowed,
                capacity: capacity
            )

            if candidateTokens <= capacity || current.isEmpty {
                current = candidate
            } else {
                flushCurrent()
                current = trimmedFragment
                let fragmentTokens = try tokenCountForSegment(
                    for: current,
                    lexicon: lexicon,
                    caseSensitiveLexicon: caseSensitiveLexicon,
                    customLexicon: customLexicon,
                    allowed: allowed,
                    capacity: capacity
                )
                if fragmentTokens > capacity {
                    // Fall back to returning empty so caller can handle via chunk builder.
                    return []
                }
            }
        }

        flushCurrent()
        return assembled
    }

    private static func splitByPunctuation(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }

        var segments: [String] = []
        var currentStart = text.startIndex
        let breakCharacters = CharacterSet(charactersIn: ",;:")
        let separatorTokens = [": ", "; ", ", "]

        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = text
        tagger.enumerateTags(
            in: text.startIndex..<text.endIndex,
            unit: .word,
            scheme: .lexicalClass,
            options: []
        ) { tag, range in
            guard tag == .punctuation else { return true }
            let token = text[range]
            if token.unicodeScalars.contains(where: { breakCharacters.contains($0) }) {
                var endIndex = range.upperBound
                for separator in separatorTokens where text[endIndex...].hasPrefix(separator) {
                    endIndex = text.index(endIndex, offsetBy: separator.count)
                    break
                }
                let segment = text[currentStart..<endIndex]
                let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    segments.append(trimmed)
                }
                currentStart = endIndex
            }
            return true
        }

        if currentStart < text.endIndex {
            let tail = text[currentStart..<text.endIndex]
            let trimmedTail = tail.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedTail.isEmpty {
                segments.append(trimmedTail)
            }
        }

        return segments.isEmpty ? [text] : segments
    }

    private static func normalize(_ word: String) -> String {
        let lowered = word.lowercased()
        let allowedSet = CharacterSet.letters.union(.decimalDigits).union(CharacterSet(charactersIn: "'"))
        let filteredScalars = lowered.unicodeScalars.filter { allowedSet.contains($0) }
        return String(String.UnicodeScalarView(filteredScalars))
    }

    private static func collapseNewlines(_ text: String) -> String {
        guard text.contains(where: { $0.isNewline }) else { return text }
        let segments = text.split(whereSeparator: { $0.isNewline })
        return segments.map(String.init).joined(separator: " ")
    }

    private static func appendSegment(_ base: String, with next: String) -> String {
        let trimmedNext = next.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedNext.isEmpty else { return base }
        if base.isEmpty { return trimmedNext }
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
        guard let spelled = FluidAudioTTS.spellOutFormatter.string(from: NSNumber(value: value)) else { return nil }
        let separators = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "-"))
        let components =
            spelled
            .lowercased()
            .components(separatedBy: separators)
            .filter { !$0.isEmpty }
        return components.isEmpty ? nil : components
    }

    private static let noPrespaceCharacters: Set<Character> = [
        ",", ";", ":", "!", "?", ".", "…", "—", "–", "'", "\"", ")", "]", "}", "”", "’",
    ]

    private static let letterPronunciations: [String: [String]] = [
        "a": ["e", "ɪ"],
        "b": ["b", "i"],
        "c": ["s", "i"],
        "d": ["d", "i"],
        "e": ["i"],
        "f": ["ɛ", "f"],
        "g": ["ʤ", "i"],
        "h": ["e", "ɪ", "ʧ"],
        "i": ["a", "ɪ"],
        "j": ["ʤ", "e"],
        "k": ["k", "e"],
        "l": ["ɛ", "l"],
        "m": ["ɛ", "m"],
        "n": ["ɛ", "n"],
        "o": ["o"],
        "p": ["p", "i"],
        "q": ["k", "j", "u"],
        "r": ["ɑ", "r"],
        "s": ["ɛ", "s"],
        "t": ["t", "i"],
        "u": ["j", "u"],
        "v": ["v", "i"],
        "w": ["d", "ʌ", "b", "əl", "j", "u"],
        "x": ["ɛ", "k", "s"],
        "y": ["w", "a", "ɪ"],
        "z": ["z", "i"],
    ]
}
