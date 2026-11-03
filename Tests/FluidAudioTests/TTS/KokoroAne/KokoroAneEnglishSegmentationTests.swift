import XCTest

@testable import FluidAudio

/// The phonemize walk records which characters produced which phoneme scalars.
/// A skipped token drops only that token.
final class KokoroAneEnglishSegmentationTests: XCTestCase {
    private let punctuation: Set<Character> = [",", ".", "!", "?", ";", ":", "…"]

    private func makePhonemizer(
        words: [String: [String]] = [
            "hello": ["h", "ə", "l", "ˈ", "O"],
            "world": ["w", "ˈ", "ɜ", "ɹ", "l", "d"],
            "to": ["t", "u"],
            "want": ["w", "ˈ", "ɑ", "n", "t"],
            "go": ["ɡ", "ˈ", "O"],
            "metal": ["m", "ˈ", "ɛ", "ɾ", "l"],
        ],
        caseSensitive: [String: [String]] = [
            "AI": ["ˈ", "A", "ˌ", "I"],
            "A": ["ˈ", "A"],
            "I": ["ˈ", "I"],
            "F": ["ˈ", "ɛ", "f"],
            "B": ["b", "ˈ", "i"],
            "NASA": ["n", "ˈ", "æ", "s", "ə"],
        ]
    ) -> KokoroAneEnglishPhonemizer {
        KokoroAneEnglishPhonemizer(
            wordToPhonemes: words,
            caseSensitiveWordToPhonemes: caseSensitive,
            allowedPunctuation: punctuation
        )
    }

    func testPhonemizeReturnsTheSegmentedString() async throws {
        let phonemizer = makePhonemizer()
        let phonemes = try await phonemizer.phonemize("Hello, world!") { _ in nil }
        let segmentation = try await phonemizer.segment("Hello, world!") { _ in nil }
        XCTAssertEqual(phonemes, segmentation.phonemes)
        XCTAssertEqual(segmentation.phonemes, "həlˈO, wˈɜɹld!")
    }

    func testPunctuationExtendsThePreviousSegmentAndTheSpaceStaysOnIt() async throws {
        let segmentation = try await makePhonemizer().segment("Hello, world!") { _ in nil }
        XCTAssertEqual(segmentation.segments.count, 2)
        XCTAssertEqual(segmentation.segments[0].surface, 0..<6)
        XCTAssertEqual(segmentation.segments[0].phonemes, "həlˈO,")
        XCTAssertEqual(segmentation.segments[1].surface, 7..<13)
        XCTAssertEqual(segmentation.segments[1].phonemes, "wˈɜɹld!")
        assertScalarsCover(segmentation)
        let scalars = Array(segmentation.phonemes.unicodeScalars)
        let first = try XCTUnwrap(segmentation.segments[0].scalars)
        XCTAssertEqual(scalars[first.upperBound - 1], " ")
    }

    func testADroppedMarkAndASkippedWordLeaveTheirNeighbors() async throws {
        let dropped = try await makePhonemizer().segment("hello # world") { _ in nil }
        XCTAssertEqual(dropped.segments.map(\.phonemes), ["həlˈO", "", "wˈɜɹld"])
        XCTAssertEqual(dropped.segments.map(\.surface), [0..<5, 6..<7, 8..<13])
        XCTAssertNil(dropped.segments[1].scalars)
        XCTAssertEqual(dropped.phonemes, "həlˈO wˈɜɹld")
        assertScalarsCover(dropped)

        let skipped = try await makePhonemizer().segment("want zzz go") { word in
            word == "zzz" ? nil : ["x"]
        }
        XCTAssertEqual(skipped.segments.map(\.phonemes), ["wˈɑnt", "", "ɡˈO"])
        XCTAssertEqual(skipped.segments.map(\.surface), [0..<4, 5..<8, 9..<11])
        XCTAssertNil(skipped.segments[1].scalars)
        XCTAssertEqual(skipped.phonemes, "wˈɑnt ɡˈO")
        assertScalarsCover(skipped)
    }

    func testLeadingTrimAndASmartApostropheKeepCharacterOffsets() async throws {
        let phonemizer = makePhonemizer(words: ["we're": ["w", "ɪ", "ɹ"], "hello": ["h", "ə", "l", "ˈ", "O"]])
        let padded = try await phonemizer.segment("  Hello") { _ in nil }
        XCTAssertEqual(padded.segments.map(\.surface), [2..<7])
        let curly = try await phonemizer.segment("  we\u{2019}re") { _ in nil }
        XCTAssertEqual(curly.segments.map(\.surface), [2..<7])
        XCTAssertEqual(curly.phonemes, "wɪɹ")
        assertScalarsCover(curly)
    }

    func testLetterNamesAndAHyphenSplitStayInsideOneSegment() async throws {
        let phonemizer = makePhonemizer()
        let ai = try await phonemizer.segment("AI") { _ in nil }
        XCTAssertEqual(ai.segments.count, 1)
        XCTAssertEqual(ai.segments[0].phonemes, "ˈA ˈI")
        XCTAssertEqual(ai.segments[0].surface, 0..<2)
        assertScalarsCover(ai)

        let compound = try await phonemizer.segment("tales-to-amaze") { word in
            ["<g2p:\(word)>"]
        }
        XCTAssertEqual(compound.segments.count, 1)
        XCTAssertEqual(compound.segments[0].phonemes, "<g2p:tales> tu <g2p:amaze>")
        XCTAssertEqual(compound.segments[0].surface, 0..<14)
        assertScalarsCover(compound)
    }

    func testAPossessiveCliticStaysOnTheStemSegment() async throws {
        let phonemizer = makePhonemizer()
        let nasa = try await phonemizer.segment("NASA's") { _ in nil }
        XCTAssertEqual(nasa.segments.count, 1)
        XCTAssertEqual(nasa.segments[0].phonemes, "nˈæsəz")
        XCTAssertFalse(nasa.segments[0].phonemes.contains(" "))

        let fbi = try await phonemizer.segment("FBI's") { _ in nil }
        XCTAssertEqual(fbi.segments.count, 1)
        XCTAssertEqual(fbi.segments[0].phonemes, "ˈɛf bˈi ˈIz")
        XCTAssertEqual(fbi.segments[0].surface, 0..<5)
        assertScalarsCover(fbi)
    }

    func testRepeatedWordsGetDistinctSurfaces() async throws {
        let segmentation = try await makePhonemizer().segment("go go") { _ in nil }
        XCTAssertEqual(segmentation.segments.map(\.surface), [0..<2, 3..<5])
        XCTAssertEqual(segmentation.segments.map(\.phonemes), ["ɡˈO", "ɡˈO"])
        assertScalarsCover(segmentation)
    }

    func testFlapRewriteKeepsTheScalarCount() async throws {
        let segmentation = try await makePhonemizer().segment("metal") { _ in nil }
        XCTAssertEqual(segmentation.phonemes, "mˈɛɾl")
        let rewritten = segmentation.segments.map { KokoroAneManager.misakiOutputForm($0.phonemes) }
        XCTAssertEqual(
            rewritten.map(\.unicodeScalars.count),
            segmentation.segments.map { $0.phonemes.unicodeScalars.count }
        )
        XCTAssertEqual(KokoroAneManager.misakiOutputForm(segmentation.phonemes), rewritten.joined(separator: " "))
        XCTAssertEqual(rewritten, ["mˈɛTl"])
    }

    /// Kept scalar ranges tile the phoneme string. Empty segments stay out of it.
    private func assertScalarsCover(
        _ segmentation: KokoroAneEnglishSegmentation, file: StaticString = #filePath, line: UInt = #line
    ) {
        let joined = segmentation.segments.filter { !$0.phonemes.isEmpty }.map(\.phonemes).joined(separator: " ")
        XCTAssertEqual(joined, segmentation.phonemes, file: file, line: line)
        var covered = 0
        for segment in segmentation.segments {
            guard let scalars = segment.scalars else {
                XCTAssertEqual(segment.phonemes, "", file: file, line: line)
                continue
            }
            XCTAssertEqual(scalars.lowerBound, covered, file: file, line: line)
            XCTAssertLessThan(scalars.lowerBound, scalars.upperBound, file: file, line: line)
            covered = scalars.upperBound
        }
        XCTAssertEqual(covered, segmentation.phonemes.unicodeScalars.count, file: file, line: line)
    }
}
