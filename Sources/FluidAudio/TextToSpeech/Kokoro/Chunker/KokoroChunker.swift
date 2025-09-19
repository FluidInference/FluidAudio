import Foundation
import OSLog

/// Punctuation-aware chunker that splits paragraphs → sentences → clauses
/// and packs them under the model token budget.
enum KokoroChunker {
    static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroChunker")

    enum LanguageStrategy {
        case latinLike
        case preserveNonLatin

        static func from(languageCode: String?) -> LanguageStrategy {
            guard let code = languageCode?.lowercased(), !code.isEmpty else { return .latinLike }
            let nonLatinPrefixes: [String] = ["zh", "ja", "ko", "cmn", "yue"]
            for prefix in nonLatinPrefixes {
                if code.hasPrefix(prefix) {
                    return .preserveNonLatin
                }
            }
            return .latinLike
        }
    }

    /// Build chunks under the token budget.
    /// - Parameters:
    ///   - text: Raw input text
    ///   - wordToPhonemes: Mapping of lowercase words to phoneme arrays
    ///   - targetTokens: Model token capacity for `input_ids`
    ///   - hasLanguageToken: Whether to reserve one token for a language marker
    ///   - languageCode: BCP-47 language code associated with the selected voice
    ///   - allowedPhonemeTokens: Optional override for allowable phoneme tokens (used in tests)
    /// - Returns: Array of `TextChunk` with punctuation-driven `pauseAfterMs`
    static func chunk(
        text: String,
        wordToPhonemes: [String: [String]],
        targetTokens: Int,
        hasLanguageToken: Bool,
        languageCode: String? = nil,
        voiceIdentifier: String? = nil,
        allowedPhonemeTokens overrideAllowedTokens: Set<String>? = nil
    ) -> [TextChunk] {
        let normalizedLanguageCode = languageCode?.lowercased()
        let strategy = LanguageStrategy.from(languageCode: normalizedLanguageCode)
        return chunkInternal(
            text: text,
            wordToPhonemes: wordToPhonemes,
            targetTokens: targetTokens,
            hasLanguageToken: hasLanguageToken,
            languageCode: normalizedLanguageCode,
            voiceIdentifier: voiceIdentifier,
            languageStrategy: strategy,
            allowedPhonemeTokens: overrideAllowedTokens
        )
    }

    static func chunkInternal(
        text: String,
        wordToPhonemes: [String: [String]],
        targetTokens: Int,
        hasLanguageToken: Bool,
        languageCode: String?,
        voiceIdentifier: String?,
        languageStrategy: LanguageStrategy,
        allowedPhonemeTokens overrideAllowedTokens: Set<String>? = nil
    ) -> [TextChunk] {
        let baseOverhead = 2 + (hasLanguageToken ? 1 : 0)
        let safety = 12
        let cap = max(1, targetTokens - safety)

        let pauseSentence = 300
        let pauseClause = 150
        let pauseParagraph = 500
        let pauseLineBreak = 250

        let allowedTokens: Set<String>
        if let overrideAllowedTokens {
            allowedTokens = overrideAllowedTokens
        } else {
            let vocabulary = KokoroVocabulary.getVocabulary()
            allowedTokens = Set(vocabulary.keys)
        }

        let normalizedText = ChunkPreprocessor.process(text)
        let paragraphs = ChunkTokenizer.paragraphs(from: normalizedText)
        var unitsPerParagraph: [[ChunkTokenizer.ClauseUnit]] = []
        unitsPerParagraph.reserveCapacity(paragraphs.count)
        for paragraph in paragraphs {
            let units = ChunkTokenizer.clauseUnits(
                for: paragraph,
                pauseSentence: pauseSentence,
                pauseClause: pauseClause,
                pauseLineBreak: pauseLineBreak
            )
            unitsPerParagraph.append(units)
        }
        let phonemizer = WordPhonemizer(
            wordToPhonemes: wordToPhonemes,
            allowedTokens: allowedTokens,
            languageCode: languageCode,
            voiceIdentifier: voiceIdentifier,
            languageStrategy: languageStrategy
        )

        var chunks: [TextChunk] = []

        for (pIndex, units) in unitsPerParagraph.enumerated() {
            var currentWords: [String] = []
            var currentPhonemes: [String] = []
            var currentTokenCount = baseOverhead
            var lastPause = pIndex < paragraphs.count - 1 ? pauseParagraph : 0
            var currentTrailingPunctuation: String? = nil
            var currentSentenceBoundary = false

            for unit in units {
                let phonemes = phonemizer.phonemize(words: unit.words)
                let fitsEmpty = baseOverhead + phonemes.count <= cap

                if phonemes.isEmpty {
                    lastPause = unit.pause
                    continue
                }

                if !fitsEmpty && !currentPhonemes.isEmpty {
                    if currentPhonemes.last == " " {
                        currentPhonemes.removeLast()
                    }
                    chunks.append(
                        TextChunk(
                            words: currentWords,
                            phonemes: currentPhonemes,
                            totalFrames: 0,
                            pauseAfterMs: lastPause,
                            isForcedSplit: false,
                            trailingPunctuation: currentTrailingPunctuation,
                            isSentenceBoundary: currentSentenceBoundary
                        )
                    )
                    currentWords.removeAll()
                    currentPhonemes.removeAll()
                    currentTokenCount = baseOverhead
                    currentTrailingPunctuation = nil
                    currentSentenceBoundary = false
                }

                if !fitsEmpty {
                    var subWords: [String] = []
                    var subPhonemes: [String] = []
                    var subCount = baseOverhead

                    func flushSub(finalPause: Int, punctuation: String? = nil, isSentenceBoundary: Bool = false) {
                        guard !subPhonemes.isEmpty else { return }
                        if subPhonemes.last == " " {
                            subPhonemes.removeLast()
                        }
                        chunks.append(
                            TextChunk(
                                words: subWords,
                                phonemes: subPhonemes,
                                totalFrames: 0,
                                pauseAfterMs: finalPause,
                                isForcedSplit: false,
                                trailingPunctuation: punctuation,
                                isSentenceBoundary: isSentenceBoundary
                            )
                        )
                        subWords.removeAll(keepingCapacity: true)
                        subPhonemes.removeAll(keepingCapacity: true)
                        subCount = baseOverhead
                    }

                    for (index, word) in unit.words.enumerated() {
                        let perWordPhonemes = phonemizer.phonemize(words: [word])
                        let hasPhonemes = !perWordPhonemes.isEmpty
                        let cost = (hasPhonemes ? perWordPhonemes.count : 1) + (subPhonemes.isEmpty ? 0 : 1)

                        if subCount + cost <= cap {
                            if hasPhonemes {
                                if !subPhonemes.isEmpty {
                                    subPhonemes.append(" ")
                                }
                                subPhonemes.append(contentsOf: perWordPhonemes)
                            }
                            subWords.append(word)
                            subCount += cost
                        } else {
                            flushSub(finalPause: 0)
                            if hasPhonemes {
                                subPhonemes.append(contentsOf: perWordPhonemes)
                            }
                            subWords.append(word)
                            subCount = baseOverhead + (hasPhonemes ? perWordPhonemes.count : 1)
                        }

                        if index == unit.words.count - 1 {
                            flushSub(finalPause: unit.pause, punctuation: unit.trailingPunctuation, isSentenceBoundary: unit.isSentenceBoundary)
                            lastPause = unit.pause
                        }
                    }

                    currentTrailingPunctuation = nil
                    currentSentenceBoundary = false
                    continue
                }

                let additional = phonemes.count + (currentPhonemes.isEmpty ? 0 : 1)
                if currentTokenCount + additional <= cap {
                    if !currentPhonemes.isEmpty {
                        currentPhonemes.append(" ")
                    }
                    currentPhonemes.append(contentsOf: phonemes)
                    currentWords.append(contentsOf: unit.words)
                    currentTokenCount += additional
                    lastPause = unit.pause
                } else {
                    if !currentPhonemes.isEmpty {
                        if currentPhonemes.last == " " {
                            currentPhonemes.removeLast()
                        }
                        chunks.append(
                            TextChunk(
                                words: currentWords,
                                phonemes: currentPhonemes,
                                totalFrames: 0,
                                pauseAfterMs: lastPause,
                                isForcedSplit: false,
                                trailingPunctuation: currentTrailingPunctuation,
                                isSentenceBoundary: currentSentenceBoundary
                            )
                        )
                        currentTrailingPunctuation = nil
                        currentSentenceBoundary = false
                    }
                    currentWords = unit.words
                    currentPhonemes = phonemes
                    currentTokenCount = baseOverhead + phonemes.count
                    lastPause = unit.pause
                }
                currentTrailingPunctuation = unit.trailingPunctuation
                currentSentenceBoundary = unit.isSentenceBoundary
            }

            if !currentPhonemes.isEmpty {
                if currentPhonemes.last == " " {
                    currentPhonemes.removeLast()
                }
                let finalPause = (pIndex < paragraphs.count - 1) ? pauseParagraph : lastPause
                chunks.append(
                    TextChunk(
                        words: currentWords,
                        phonemes: currentPhonemes,
                        totalFrames: 0,
                        pauseAfterMs: finalPause,
                        isForcedSplit: false,
                        trailingPunctuation: currentTrailingPunctuation,
                        isSentenceBoundary: currentSentenceBoundary
                    )
                )
            }
        }

        return chunks
    }
}

private struct WordPhonemizer {
    let wordToPhonemes: [String: [String]]
    let allowedTokens: Set<String>
    let languageCode: String?
    let voiceIdentifier: String?
    let languageStrategy: KokoroChunker.LanguageStrategy

    func phonemize(words: [String]) -> [String] {
        var out: [String] = []

        func isPunct(_ character: Character) -> Bool {
            !(character.isLetter || character.isNumber || character.isWhitespace || character == "'")
        }

        func normalize(_ input: String) -> String {
            let lowered = input.lowercased()
                .replacingOccurrences(of: "\u{2019}", with: "'")
                .replacingOccurrences(of: "\u{2018}", with: "'")
                .replacingOccurrences(of: "\u{201B}", with: "'")

            switch languageStrategy {
            case .latinLike:
                let allowedSet = CharacterSet.letters
                    .union(.decimalDigits)
                    .union(CharacterSet(charactersIn: "'"))
                return String(lowered.unicodeScalars.filter { allowedSet.contains($0) })
            case .preserveNonLatin:
                return lowered
            }
        }

        for (index, word) in words.enumerated() {
            var segment = ""

            func flushSegment() {
                guard !segment.isEmpty else { return }
                let key = normalize(segment)
                if let arr = wordToPhonemes[key] {
                    out.append(contentsOf: arr)
                } else {
                    phonemizeOOV(key: key, output: &out)
                }
                segment.removeAll()
            }

            for character in word {
                if isPunct(character) {
                    flushSegment()
                    let punct = String(character)
                    if allowedTokens.contains(punct) {
                        out.append(punct)
                    }
                } else {
                    segment.append(character)
                }
            }

            flushSegment()
            if index != words.count - 1 {
                out.append(" ")
            }
        }

        if out.last == " " {
            out.removeLast()
        }

        return out
    }

    private func phonemizeOOV(key: String, output: inout [String]) {
        guard !key.isEmpty else { return }
        #if canImport(ESpeakNG)
        if let ipa = EspeakG2P.shared.phonemize(
            word: key,
            voiceIdentifier: voiceIdentifier,
            languageCode: languageCode
        ) {
            let mapped = PhonemeMapper.mapIPA(ipa, allowed: allowedTokens)
            if !mapped.isEmpty {
                output.append(contentsOf: mapped)
                KokoroChunker.logger.info("EspeakG2P used for OOV word: \(key)")
            } else {
                KokoroChunker.logger.warning("OOV word yielded no mappable IPA tokens: \(key)")
            }
        } else {
            KokoroChunker.logger.warning("EspeakG2P failed for OOV word: \(key)")
        }
        #else
        KokoroChunker.logger.warning("ESpeakNG not available; skipping OOV word: \(key)")
        #endif
    }
}
