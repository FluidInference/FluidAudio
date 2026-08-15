import Foundation
import XCTest

@testable import FluidAudio

/// Tests for the English KokoroAne text frontend (issue #691): Misaki
/// lexicon weak forms beat the BART G2P citation forms, punctuation is
/// preserved as prosody tokens, and custom-lexicon overrides win.
final class KokoroAneEnglishPhonemizerTests: XCTestCase {

    /// Misaki-style lexicon stand-in. `to` is the issue #691 word: the
    /// lexicon carries the unstressed weak form while BART G2P returns
    /// the stressed citation form `tˈO`.
    private let lexicon: [String: [String]] = [
        "'em": ["ə", "m"],
        "'til": ["t", "ˈ", "I", "l"],
        "'twas": ["t", "w", "ˈ", "ɑ", "z"],
        "to": ["t", "u"],
        "i": ["ˈ", "I"],
        "want": ["w", "ˈ", "ɑ", "n", "t"],
        "go": ["ɡ", "ˈ", "O"],
        "hello": ["h", "ə", "l", "ˈ", "O"],
        "there's": ["ð", "ɛ", "ɹ", "z"],
        "world": ["w", "ˈ", "ɜ", "ɹ", "l", "d"],
        // Lowercase pronoun must stay the weak `ʌs` shape (issue #710).
        "us": ["ˌ", "ʌ", "s"],
    ]

    /// Mirrors the real `us_lexicon_cache.json`: the blended `AI`/`US`
    /// shapes the #710 overrides bypass, the per-letter names the spell-out
    /// reads, and known acronyms that must stay lexicon-backed.
    private let caseSensitive: [String: [String]] = [
        "AI": ["ˈ", "A", "ˌ", "I"],
        "US": ["ˌ", "ʌ", "s"],
        "A": ["ˈ", "A"],
        "I": ["ˈ", "I"],
        "U": ["j", "ˈ", "u"],
        "S": ["ˈ", "ɛ", "s"],
        "F": ["ˈ", "ɛ", "f"],
        "B": ["b", "ˈ", "i"],
        "T": ["t", "ˈ", "i"],
        "P": ["p", "ˈ", "i"],
        "NASA": ["n", "ˈ", "æ", "s", "ə"],
        "OK": ["ˌ", "O", "k", "ˈ", "A"],
    ]

    /// Punctuation present in the real `ANE/vocab.json`.
    private let punctuation: Set<Character> = [",", ".", "!", "?", ";", ":", "…"]

    private func makePhonemizer(
        custom: [String: String] = [:]
    ) -> KokoroAneEnglishPhonemizer {
        KokoroAneEnglishPhonemizer(
            wordToPhonemes: lexicon,
            caseSensitiveWordToPhonemes: caseSensitive,
            customLexicon: custom,
            allowedPunctuation: punctuation
        )
    }

    /// G2P stand-in that returns the stressed citation form for "to" the
    /// way the BART model does, and records which words reached it.
    private actor FallbackRecorder {
        var words: [String] = []
        func g2p(_ word: String) -> [String]? {
            words.append(word)
            if word == "to" { return ["t", "ˈ", "O"] }
            return ["<g2p:\(word)>"]
        }
    }

    // MARK: - Weak forms (the issue #691 symptom)

    func testFunctionWordToUsesLexiconWeakFormNotG2P() async throws {
        let recorder = FallbackRecorder()
        let result = try await makePhonemizer().phonemize("I want to go") { await recorder.g2p($0) }

        XCTAssertEqual(result, "ˈI wˈɑnt tu ɡˈO")
        XCTAssertFalse(result.contains("tˈO"), "'to' must not get the stressed citation form")
        let recordedEmpty = await recorder.words.isEmpty
        XCTAssertTrue(recordedEmpty, "all words should resolve from the lexicon")
    }

    func testUppercaseToStillResolvesWeakForm() async throws {
        // "TO" has no case-sensitive entry; it must hit the lower-cased
        // lexicon, not fall through to G2P.
        let recorder = FallbackRecorder()
        let result = try await makePhonemizer().phonemize("TO") { await recorder.g2p($0) }
        XCTAssertEqual(result, "tu")
        let recorded = await recorder.words
        XCTAssertTrue(recorded.isEmpty)
    }

    // MARK: - Resolution order

    func testCaseSensitiveLexiconWinsForProperNouns() async throws {
        // `NASA` is a lexicon-backed acronym, not spelled out.
        let result = try await makePhonemizer().phonemize("NASA") { _ in nil }
        XCTAssertEqual(result, "nˈæsə")
    }

    // MARK: - Letter-name initialisms (issue #710)

    func testAIOverrideSpellsLetterNamesNotBlendedShape() async throws {
        // `AI` bypasses the blended `ˈAˌI` lexicon entry and reads `A I`.
        let result = try await makePhonemizer().phonemize("AI") { _ in nil }
        XCTAssertEqual(result, "ˈA ˈI")
    }

    func testUSOverrideSpellsLetterNamesNotPronoun() async throws {
        // Uppercase `US` reads `U S`, not the lowercase pronoun `ʌs`.
        let result = try await makePhonemizer().phonemize("US") { _ in nil }
        XCTAssertEqual(result, "jˈu ˈɛs")
    }

    func testLowercaseUsStaysPronoun() async throws {
        // The override only matches the exact uppercase spelling.
        let result = try await makePhonemizer().phonemize("us") { _ in nil }
        XCTAssertEqual(result, "ˌʌs")
    }

    func testUnknownAllCapsInitialismSpelledAsLetterNames() async throws {
        // `FBI`/`ATP` miss the lexicon and spell out instead of reaching G2P.
        let recorder = FallbackRecorder()
        let fbi = try await makePhonemizer().phonemize("FBI") { await recorder.g2p($0) }
        XCTAssertEqual(fbi, "ˈɛf bˈi ˈI")
        let atp = try await makePhonemizer().phonemize("ATP") { await recorder.g2p($0) }
        XCTAssertEqual(atp, "ˈA tˈi pˈi")
        let recorded = await recorder.words
        XCTAssertTrue(recorded.isEmpty, "initialisms must not reach BART G2P")
    }

    func testKnownAcronymStaysLexiconBackedNotSpelled() async throws {
        // `OK` is a lexicon hit (2-5 all-caps) — it keeps its bundled shape
        // rather than spelling `O K`.
        let result = try await makePhonemizer().phonemize("OK") { _ in nil }
        XCTAssertEqual(result, "ˌOkˈA")
    }

    func testInitialismSpellOutFallsThroughToG2PWithoutLetterEntries() async throws {
        // G2P-only degraded path: no per-letter lexicon entries, so the
        // all-caps token must reach the fallback rather than emit a partial.
        let phonemizer = KokoroAneEnglishPhonemizer(allowedPunctuation: punctuation)
        let recorder = FallbackRecorder()
        let result = try await phonemizer.phonemize("FBI") { await recorder.g2p($0) }
        XCTAssertEqual(result, "<g2p:fbi>")
        let recorded = await recorder.words
        XCTAssertEqual(recorded, ["fbi"])
    }

    func testOverrideFallsBackToLexiconWhenLettersMissing() async throws {
        // Degraded lexicon: `US` is present but the per-letter entries are
        // not, so the override can't spell it and falls through to the
        // bundled shape (logged, never silently dropped or sent to G2P).
        let phonemizer = KokoroAneEnglishPhonemizer(
            caseSensitiveWordToPhonemes: ["US": ["ˌ", "ʌ", "s"]],
            allowedPunctuation: punctuation
        )
        let recorder = FallbackRecorder()
        let result = try await phonemizer.phonemize("US") { await recorder.g2p($0) }
        XCTAssertEqual(result, "ˌʌs")
        let recorded = await recorder.words
        XCTAssertTrue(recorded.isEmpty, "override fall-through must use the lexicon, not G2P")
    }

    func testLongAllCapsWordIsNotSpelledButReachesG2P() async throws {
        // Outside the 2-5 length range → not an initialism; reaches G2P
        // instead of being spelled letter by letter. (Candidate boundaries
        // are unit-tested in EnglishInitialismsTests.)
        let recorder = FallbackRecorder()
        let result = try await makePhonemizer().phonemize("ABCDEF") { await recorder.g2p($0) }
        XCTAssertEqual(result, "<g2p:abcdef>")
        let recorded = await recorder.words
        XCTAssertEqual(recorded, ["abcdef"])
    }

    func testOOVWordFallsBackToG2PWithNormalizedSpelling() async throws {
        let recorder = FallbackRecorder()
        let result = try await makePhonemizer().phonemize("I want Zorblax") { await recorder.g2p($0) }
        XCTAssertEqual(result, "ˈI wˈɑnt <g2p:zorblax>")
        let recordedWords = await recorder.words
        XCTAssertEqual(recordedWords, ["zorblax"])
    }

    func testCustomLexiconOverridesEverything() async throws {
        let phonemizer = makePhonemizer(custom: ["to": "tə"])
        let result = try await phonemizer.phonemize("I want to go") { _ in nil }
        XCTAssertEqual(result, "ˈI wˈɑnt tə ɡˈO")
    }

    func testCustomLexiconExactSpellingBeatsLowercased() async throws {
        let phonemizer = makePhonemizer(custom: ["to": "tə", "TO": "tˈu"])
        let emphatic = try await phonemizer.phonemize("TO") { _ in nil }
        XCTAssertEqual(emphatic, "tˈu")
        let weak = try await phonemizer.phonemize("to") { _ in nil }
        XCTAssertEqual(weak, "tə")
    }

    // MARK: - Punctuation and quote delimiters

    func testSupportedPunctuationAttachesToPrecedingWord() async throws {
        let result = try await makePhonemizer().phonemize("Hello, world!") { _ in nil }
        XCTAssertEqual(result, "həlˈO, wˈɜɹld!")
    }

    func testUnsupportedPunctuationIsDropped() async throws {
        // '#' is not in the chain vocab → dropped, no stray space.
        let result = try await makePhonemizer().phonemize("hello # world") { _ in nil }
        XCTAssertEqual(result, "həlˈO wˈɜɹld")
    }

    func testApostropheWordsStayIntactForLexiconLookup() async throws {
        let phonemizer = KokoroAneEnglishPhonemizer(
            wordToPhonemes: ["don't": ["d", "ˈ", "O", "n", "t"]],
            allowedPunctuation: punctuation
        )
        let result = try await phonemizer.phonemize("don't") { _ in nil }
        XCTAssertEqual(result, "dˈOnt")
    }

    func testSingleQuotesAreDelimitersNotPartOfLexiconKey() async throws {
        let recorder = FallbackRecorder()
        let result = try await makePhonemizer().phonemize("'to'") { await recorder.g2p($0) }
        XCTAssertEqual(result, "tu")
        let recorded = await recorder.words
        XCTAssertTrue(recorded.isEmpty)
    }

    func testQuotedSentenceKeepsContractionsIntact() async throws {
        let recorder = FallbackRecorder()
        let result = try await makePhonemizer().phonemize("'there's to'") {
            await recorder.g2p($0)
        }
        XCTAssertEqual(result, "ðɛɹz tu")
        let recorded = await recorder.words
        XCTAssertTrue(recorded.isEmpty)
    }

    func testKnownLeadingApostropheWordsStayIntactForLexiconLookup() async throws {
        let recorder = FallbackRecorder()
        let result = try await makePhonemizer().phonemize("'twas 'em 'til 'to'") {
            await recorder.g2p($0)
        }

        XCTAssertEqual(result, "twˈɑz əm tˈIl tu")
        let recorded = await recorder.words
        XCTAssertTrue(recorded.isEmpty)
    }

    // MARK: - Degraded paths

    func testG2PNilSkipsWordButKeepsRest() async throws {
        let result = try await makePhonemizer().phonemize("want zzz go") { word in
            word == "zzz" ? nil : ["x"]
        }
        XCTAssertEqual(result, "wˈɑnt ɡˈO")
    }

    func testG2PErrorPropagates() async {
        struct Boom: Error {}
        do {
            _ = try await makePhonemizer().phonemize("Zorblax") { _ in throw Boom() }
            XCTFail("expected error to propagate")
        } catch {
            XCTAssertTrue(error is Boom)
        }
    }

    func testEmptyInputThrows() async {
        do {
            _ = try await makePhonemizer().phonemize("   ") { _ in nil }
            XCTFail("expected inputProcessingFailed")
        } catch let error as KokoroAneError {
            guard case .inputProcessingFailed = error else {
                return XCTFail("unexpected error: \(error)")
            }
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testNothingResolvedThrows() async {
        do {
            _ = try await makePhonemizer().phonemize("zzz") { _ in nil }
            XCTFail("expected inputProcessingFailed")
        } catch let error as KokoroAneError {
            guard case .inputProcessingFailed = error else {
                return XCTFail("unexpected error: \(error)")
            }
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: - Smart apostrophes (issue #774)

    func testSmartApostropheContractionStaysIntact() async throws {
        let recorder = FallbackRecorder()
        // U+2019 curly apostrophe must fold to ASCII so `we’re` hits the
        // lexicon intact instead of splitting into `we` + `re`.
        let phonemizer = KokoroAneEnglishPhonemizer(
            wordToPhonemes: ["we're": ["w", "ɪ", "ɹ"]],
            allowedPunctuation: punctuation
        )
        let result = try await phonemizer.phonemize("we\u{2019}re here") { await recorder.g2p($0) }
        XCTAssertEqual(result, "wɪɹ <g2p:here>")
        let recorded = await recorder.words
        XCTAssertFalse(recorded.contains("re"), "contraction must not split into a standalone `re`")
    }

    func testNormalizeApostrophesFoldsTypographicForms() {
        XCTAssertEqual(KokoroAneEnglishPhonemizer.normalizeApostrophes("we\u{2019}re"), "we're")
        XCTAssertEqual(KokoroAneEnglishPhonemizer.normalizeApostrophes("\u{2018}tis"), "'tis")
        XCTAssertEqual(KokoroAneEnglishPhonemizer.normalizeApostrophes("we\u{02BC}re"), "we're")
        // No smart apostrophe → unchanged.
        XCTAssertEqual(KokoroAneEnglishPhonemizer.normalizeApostrophes("plain"), "plain")
    }

    // MARK: - Hyphenated words (issue #775)

    func testHyphenatedLexiconKeyResolvesWithHyphenIntact() async throws {
        let recorder = FallbackRecorder()
        // The Misaki cache stores `twenty-one` WITH the hyphen; the lookup must
        // try the raw lowercased token before `normalizeKey` strips it.
        let phonemizer = KokoroAneEnglishPhonemizer(
            wordToPhonemes: ["twenty-one": ["t", "w", "ˈ", "ɛ", "n", "t", "i", "w", "ˈ", "ʌ", "n"]],
            allowedPunctuation: punctuation
        )
        let result = try await phonemizer.phonemize("twenty-one") { await recorder.g2p($0) }
        XCTAssertEqual(result, "twˈɛntiwˈʌn")
        let recorded = await recorder.words
        XCTAssertTrue(recorded.isEmpty, "should hit the lexicon, not G2P")
    }

    func testHyphenatedCompoundMissSplitsIntoParts() async throws {
        let recorder = FallbackRecorder()
        // `tales-to-amaze` isn't a lexicon entry; each part resolves separately
        // (`to` via lexicon, `tales`/`amaze` via G2P) instead of gluing into
        // `talestoamaze`.
        let result = try await makePhonemizer().phonemize("tales-to-amaze") { await recorder.g2p($0) }
        XCTAssertEqual(result, "<g2p:tales> tu <g2p:amaze>")
        let recorded = await recorder.words
        XCTAssertEqual(recorded, ["tales", "amaze"])
    }

    func testHyphenatedCompoundFallsBackToWholeWordWhenPartUnresolved() async throws {
        // If any part can't be resolved, the compound aborts and the whole
        // glued token goes to G2P — no partial output.
        let result = try await makePhonemizer().phonemize("go-zzz") { word in
            word == "zzz" ? nil : ["<g2p:\(word)>"]
        }
        XCTAssertEqual(result, "<g2p:gozzz>")
    }

    // MARK: - Without lexicon (pre-#691 behavior preserved)

    func testEmptyLexiconFallsBackToG2PForEveryWord() async throws {
        let phonemizer = KokoroAneEnglishPhonemizer(allowedPunctuation: punctuation)
        let recorder = FallbackRecorder()
        let result = try await phonemizer.phonemize("I want to go") { await recorder.g2p($0) }
        let recordedAll = await recorder.words
        XCTAssertEqual(recordedAll, ["i", "want", "to", "go"])
        XCTAssertTrue(result.contains("tˈO"), "G2P-only path keeps the old citation form")
    }
}
