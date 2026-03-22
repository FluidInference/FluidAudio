import Testing

@testable import FluidAudio

@Suite("KittenTTS Tokenizer Tests")
struct KittenTtsTokenizerTests {

    @Test("Vocab scalars has 178 entries")
    func vocabScalarsLength() {
        #expect(KittenTtsConstants.vocabScalars.count == KittenTtsConstants.vocabSize)
    }

    @Test("First scalar is the padding token $")
    func padToken() {
        let first = KittenTtsConstants.vocabScalars.first
        #expect(first == "$")
    }

    @Test("Empty input produces BOS + EOS only")
    func emptyInput() {
        let result = KittenTtsSynthesizer.tokenize([])
        #expect(result == [0, 0])
    }

    @Test("Single IPA character is tokenized correctly")
    func singleCharacter() {
        // 'a' should be in the vocab at a known position (after $ + punctuation + uppercase + lowercase)
        let result = KittenTtsSynthesizer.tokenize(["a"])
        #expect(result.count == 3)  // BOS + 'a' + EOS
        #expect(result.first == 0)  // BOS
        #expect(result.last == 0)  // EOS
        #expect(result[1] > 0)  // 'a' has a non-zero ID
    }

    @Test("Multiple phonemes are tokenized with BOS/EOS")
    func multiplePhonemes() {
        let result = KittenTtsSynthesizer.tokenize(["h", "ə", "l", "o"])
        #expect(result.first == 0)
        #expect(result.last == 0)
        // Should have BOS + at least some valid tokens + EOS
        #expect(result.count >= 3)
    }

    @Test("Unknown characters are dropped")
    func unknownCharactersDropped() {
        // Use characters unlikely to be in the 178-char IPA vocab
        let result = KittenTtsSynthesizer.tokenize(["🎵"])
        #expect(result == [0, 0])  // Only BOS + EOS, emoji dropped
    }

    @Test("Multi-character phoneme strings are split into individual scalars")
    func multiCharPhoneme() {
        // A phoneme like "aɪ" should be split into 'a' and 'ɪ' individually
        let result = KittenTtsSynthesizer.tokenize(["aɪ"])
        #expect(result.count == 4)  // BOS + 'a' + 'ɪ' + EOS
    }

    @Test("Pad token is not added from input")
    func padTokenNotFromInput() {
        // '$' is the pad token (index 0) and should not be added as a real token
        let result = KittenTtsSynthesizer.tokenize(["$"])
        #expect(result == [0, 0])  // Only BOS + EOS, '$' mapped to 0 but filtered
    }

    @Test("Known IPA characters map to expected indices")
    func knownCharacterMapping() {
        let vocabScalars = KittenTtsConstants.vocabScalars

        // Check that 'A' maps to its position in the vocab
        if let aIndex = vocabScalars.firstIndex(of: "A") {
            let result = KittenTtsSynthesizer.tokenize(["A"])
            #expect(result[1] == Int32(aIndex))
        }

        // Check that 'ɑ' (IPA open back unrounded vowel) maps correctly
        if let ipaIndex = vocabScalars.firstIndex(of: "\u{0251}") {
            let result = KittenTtsSynthesizer.tokenize(["ɑ"])
            #expect(result[1] == Int32(ipaIndex))
        }
    }

    @Test("Punctuation characters are tokenized")
    func punctuationTokenized() {
        let result = KittenTtsSynthesizer.tokenize(["!", ",", "."])
        // BOS + 3 punctuation chars + EOS = 5
        #expect(result.count == 5)
        // All punctuation should have valid IDs (>0)
        for id in result[1..<4] {
            #expect(id > 0)
        }
    }
}
