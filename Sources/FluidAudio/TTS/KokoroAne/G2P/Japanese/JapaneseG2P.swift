import Foundation

/// Text frontend for the Kokoro ANE Japanese variant: MeCab tokenization over
/// the trimmed `unidic-lite` dictionary and Misaki's Cutlet rules, entirely in
/// process. Assets live under `<repoDir>/g2p/` and are downloaded lazily by
/// ``KokoroAneResourceDownloader/ensureJapaneseG2P(repoDirectory:)``.
actor JapaneseG2P {
    private let tokenizer: JapaneseTokenizer
    private let dictionaryWords: Set<String>

    init(directory: URL) throws {
        let dictionary = try JapaneseMecabDictionary(directory: directory)
        tokenizer = JapaneseTokenizer(dictionary: dictionary)
        let wordsURL = directory.appendingPathComponent(KokoroAneConstants.japaneseWordListFile)
        let contents = try String(contentsOf: wordsURL, encoding: .utf8)
        dictionaryWords = Set(
            contents.split(whereSeparator: \.isNewline).map { String($0).trimmingCharacters(in: .whitespaces) })
    }

    func phonemize(_ text: String) throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("Japanese G2P received empty text.")
        }
        let result = JapaneseCutlet.phonemize(trimmed, tokenizer: tokenizer, words: dictionaryWords)
        guard !result.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("Japanese G2P produced no phonemes for '\(text)'.")
        }
        return result
    }

    /// Tokens as MeCab/fugashi would produce them, for tests and diagnostics.
    func tokens(_ text: String) -> [JapaneseTokenizer.Word] {
        tokenizer.tokenize(JapaneseCutlet.normalize(text))
    }
}
