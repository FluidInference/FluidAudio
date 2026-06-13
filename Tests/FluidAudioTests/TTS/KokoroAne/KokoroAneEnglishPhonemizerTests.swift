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
    ]

    private let caseSensitive: [String: [String]] = [
        "AI": ["ˈ", "A", "ˈ", "I"]
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

    func testCaseSensitiveLexiconWinsForAbbreviations() async throws {
        let result = try await makePhonemizer().phonemize("AI") { _ in nil }
        XCTAssertEqual(result, "ˈAˈI")
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
