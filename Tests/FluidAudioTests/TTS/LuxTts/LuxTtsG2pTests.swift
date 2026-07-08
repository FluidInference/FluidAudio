import XCTest

@testable import FluidAudio

/// Espeak-parity G2P tests. Expected strings are the espeak-ng oracle
/// output (piper_phonemize en-us via EmiliaTokenizer, i.e. the exact
/// frontend LuxTTS was trained on), generated with
/// `mobius/models/tts/zipvoice/coreml/g2p/validate.py dump-oracle`.
///
/// Corpus-level gates (1,000 sentences: 99.6% sentence exact match,
/// 0.01% token edit rate) are enforced by the reproducible harness in
/// the mobius worktree; these tests pin representative behaviors.
final class LuxTtsG2pTests: XCTestCase {

    private static let g2p: LuxTtsG2p = {
        do {
            return try LuxTtsG2p()
        } catch {
            fatalError("cannot load bundled G2P resources: \(error)")
        }
    }()

    private func assertPhonemes(
        _ text: String, _ expected: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(
            Self.g2p.phonemize(text: text), expected, file: file, line: line)
    }

    // MARK: - Phase-1 fixture sentences

    func testFixtureSentence1() {
        assertPhonemes(
            "The quick brown fox jumps over the lazy dog, and honestly, it felt great.",
            "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ, ænd ˈɑːnɪstli, ɪt fˈɛlt ɡɹˈeɪt.")
    }

    func testFixtureSentence2CamelCase() {
        assertPhonemes(
            "FluidAudio runs speech models locally on Apple silicon, no cloud required.",
            "flˈuːɪd ˈɔːdɪˌoʊ ɹˈʌnz spˈiːtʃ mˈɑːdəlz lˈoʊkəli ˌɔn ˈæpəl sˈɪlɪkən, nˈoʊ klˈaʊd ɹᵻkwˈaɪɚd.")
    }

    func testFixtureSentence3PromptTranscript() {
        assertPhonemes(
            "Quick brown fox jumps over the lazy dog and honestly it felt great.",
            "kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ ænd ˈɑːnɪstli ɪt fˈɛlt ɡɹˈeɪt.")
    }

    // MARK: - Numbers / normalization (upstream ZipVoice normalizer parity)

    func testNumbersTimeOrdinalCurrency() {
        assertPhonemes(
            "The meeting starts at 3:45 PM on March 21st, and it costs $12.50.",
            "ðə mˈiːɾɪŋ stˈɑːɹts æt θɹˈiː: fˈɔːɹɾifˈaɪv pˌiːˈɛm ˌɔn mˈɑːɹtʃ twˈɛntifˈɜːst, "
                + "ænd ɪt kˈɔsts twˈɛlv dˈɑːlɚz, fˈɪfti sˈɛnts.")
    }

    func testAbbreviationAndHomographs() {
        // Dr. -> doctor; "read" defaults to present, stays present after
        // pronoun "I" ($verbf window); "the record" selects the noun form.
        assertPhonemes(
            "Dr. Smith read the record; I read it yesterday.",
            "dˈɑːktɚ.smˈɪθ ɹˈiːd ðə ɹˈɛkɚd; aɪ ɹˈiːd ɪt jˈɛstɚdˌeɪ.")
    }

    func testNormalizerYearAndCardinal() {
        XCTAssertEqual(
            LuxTtsEnglishNormalizer.normalize("born in 1855, moved in 2007"),
            "born in  eighteen fifty-five , moved in  two thousand seven ")
        XCTAssertEqual(LuxTtsEnglishNormalizer.cardinalWords(123), "one hundred twenty-three")
        XCTAssertEqual(LuxTtsEnglishNormalizer.ordinalWords("42nd"), "forty-second")
    }

    // MARK: - espeak clause behaviors

    func testWeakFormsAndMerges() {
        // "in the" merges without a space; "the" -> ðɪ before vowels;
        // capital I is the unstressed pronoun (aɪ), lowercase i the letter.
        assertPhonemes("in the house", "ɪnðə hˈaʊs")
        assertPhonemes("the apple", "ðɪ ˈæpəl")
        assertPhonemes("I want to eat", "aɪ wˈɔnt tʊ ˈiːt")
        assertPhonemes("I want to go", "aɪ wˈɔnt tə ɡˈoʊ")
    }

    func testStrendStressResolution() {
        // $strend2: "over" is fully stressed only when followed by
        // unstressed words (with linking-r before a vowel).
        assertPhonemes("jump over it", "dʒˈʌmp ˈoʊvɚɹ ɪt")
        assertPhonemes("jumps over the lazy dog", "dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ")
    }

    func testAllCapsSpellOut() {
        assertPhonemes("the FBI called", "ðə ˌɛfbˌiːˈaɪ kˈɔːld")
    }

    func testPossessiveFallback() {
        // "John's" resolves via the possessive rule when absent from the
        // lexicon row set (voiced final -> z).
        let phonemes = Self.g2p.phonemize(text: "John's dog")
        XCTAssertTrue(phonemes.hasPrefix("dʒˈɑːnz"), "got \(phonemes)")
    }

    // MARK: - Token id mapping (fixture ids from the espeak oracle)

    func testFixtureSentence1TokenIds() throws {
        let expected = [
            41, 59, 3, 23, 35, 120, 74, 23, 3, 15, 88, 120, 14, 100, 26, 3,
            19, 120, 51, 122, 23, 31, 3, 17, 108, 120, 102, 25, 28, 31, 3,
            121, 27, 100, 34, 60, 3, 41, 59, 3, 24, 120, 18, 74, 38, 21, 3,
            17, 120, 51, 122, 66, 8, 3, 39, 26, 17, 3, 120, 51, 122, 26, 74,
            31, 32, 24, 21, 8, 3, 74, 32, 3, 19, 120, 61, 24, 32, 3, 66, 88,
            120, 18, 74, 32, 10,
        ]
        let tokensURL = try LuxTtsFixtures.resourceURL("tokens.txt")
        let tokenizer = try LuxTtsTokenizer(tokensFileURL: tokensURL)
        let phonemes = Self.g2p.phonemize(
            text: "The quick brown fox jumps over the lazy dog, and honestly, it felt great.")
        XCTAssertEqual(tokenizer.tokenIds(phonemes: phonemes), expected)
    }
}
