import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsStreamingTests: XCTestCase {

    // MARK: - AudioFrame Tests

    func testAudioFrameProperties() {
        let samples: [Float] = Array(repeating: 0.5, count: PocketTtsConstants.samplesPerFrame)
        let frame = PocketTtsSynthesizer.AudioFrame(
            samples: samples,
            frameIndex: 3,
            chunkIndex: 1,
            chunkCount: 4,
            utteranceIndex: nil
        )

        XCTAssertEqual(frame.samples.count, PocketTtsConstants.samplesPerFrame)
        XCTAssertEqual(frame.frameIndex, 3)
        XCTAssertEqual(frame.chunkIndex, 1)
        XCTAssertEqual(frame.chunkCount, 4)
        XCTAssertNil(frame.utteranceIndex)
    }

    func testAudioFrameIsSendable() {
        // Verify AudioFrame can be sent across concurrency boundaries
        let frame = PocketTtsSynthesizer.AudioFrame(
            samples: [1.0, 2.0, 3.0],
            frameIndex: 0,
            chunkIndex: 0,
            chunkCount: 1,
            utteranceIndex: nil
        )

        let expectation = expectation(description: "Frame sent across tasks")
        Task {
            let _: PocketTtsSynthesizer.AudioFrame = frame
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
    }

    // MARK: - PocketTtsManager Guard Tests

    func testSynthesizeStreamingFailsWithoutInitialization() async {
        let manager = PocketTtsManager()

        do {
            _ = try await manager.synthesizeStreaming(text: "Hello")
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testSynthesizeStreamingWithVoiceDataFailsWithoutInitialization() async {
        let manager = PocketTtsManager()
        let fakeVoiceData = PocketTtsVoiceData(audioPrompt: [], promptLength: 0)

        do {
            _ = try await manager.synthesizeStreaming(text: "Hello", voiceData: fakeVoiceData)
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - Text Normalization (used by streaming pipeline)

    func testNormalizeTextAddsTerminalPunctuation() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("Hello world")
        XCTAssertTrue(text.hasSuffix("."), "Should add period when no terminal punctuation")
    }

    func testNormalizeTextPreservesExistingPunctuation() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("Hello world!")
        XCTAssertTrue(text.hasSuffix("!"), "Should preserve existing punctuation")
        XCTAssertFalse(text.hasSuffix("!."), "Should not add extra period")
    }

    func testNormalizeTextCapitalizesFirstLetter() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("hello")
        XCTAssertTrue(text.contains("H"), "Should capitalize first letter")
    }

    func testNormalizeTextShortTextPadding() {
        // Short text (< 5 words) gets padding
        let (text, frames) = PocketTtsSynthesizer.normalizeText("Hi")
        XCTAssertTrue(text.hasPrefix(" "), "Short text should be padded")
        XCTAssertEqual(frames, PocketTtsConstants.shortTextPadFrames)
    }

    func testNormalizeTextLongTextNoExtraPadding() {
        let (_, frames) = PocketTtsSynthesizer.normalizeText(
            "This is a longer sentence with more than five words in it")
        XCTAssertEqual(frames, PocketTtsConstants.longTextExtraFrames)
    }

    // MARK: - Smart Quote Normalization (issue #584)

    func testNormalizeSmartQuotesReplacesU2019() {
        // U+2019 RIGHT SINGLE QUOTATION MARK is the default auto-corrected
        // apostrophe on most modern keyboards. It must be normalized to ASCII
        // so SentencePiece doesn't tokenize French contractions wastefully.
        let input = "Avant d\u{2019}aboutir, c\u{2019}est fini."
        let output = PocketTtsSynthesizer.normalizeSmartQuotes(input)
        XCTAssertEqual(output, "Avant d'aboutir, c'est fini.")
    }

    func testNormalizeSmartQuotesReplacesAllQuoteVariants() {
        let input = "\u{2018}hello\u{2019} and \u{201C}world\u{201D}"
        let output = PocketTtsSynthesizer.normalizeSmartQuotes(input)
        XCTAssertEqual(output, "'hello' and \"world\"")
    }

    func testNormalizeTextNormalizesSmartQuotesInline() {
        let (text, _) = PocketTtsSynthesizer.normalizeText(
            "Il n\u{2019}a pas pu d\u{2019}aboutir.")
        XCTAssertFalse(
            text.contains("\u{2019}"),
            "normalizeText should strip smart apostrophes (issue #584)")
        XCTAssertTrue(text.contains("n'a"))
        XCTAssertTrue(text.contains("d'aboutir"))
    }

    // MARK: - Mid-Sentence Chunk Normalization (issue #584)

    func testNormalizeTextMidSentencePreservesCase() {
        // Mid-sentence chunks must keep their original (lowercase) leading
        // letter; otherwise the synthesizer treats every clause as a new
        // sentence and the prosody arc breaks.
        let (text, _) = PocketTtsSynthesizer.normalizeText(
            "combustibles, carburants et chauffage", isMidSentence: true)
        XCTAssertTrue(
            text.trimmingCharacters(in: .whitespaces).hasPrefix("c"),
            "Mid-sentence chunks should not be re-capitalized; got: \(text)")
    }

    func testNormalizeTextMidSentenceDoesNotAppendPeriod() {
        // A mid-sentence chunk ending in a comma must keep the comma — adding
        // a period would render it as a standalone sentence.
        let (text, _) = PocketTtsSynthesizer.normalizeText(
            "combustibles, carburants,", isMidSentence: true)
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        XCTAssertTrue(
            trimmed.hasSuffix(","),
            "Mid-sentence chunk should keep trailing comma; got: \(text)")
        XCTAssertFalse(trimmed.hasSuffix("."))
    }

    func testNormalizeTextMidSentencePreservesPreposition() {
        // Orphaned prepositions ("de") at chunk end must not be capitalized
        // or terminated with a period.
        let (text, _) = PocketTtsSynthesizer.normalizeText(
            "stations-service de", isMidSentence: true)
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        XCTAssertEqual(trimmed, "stations-service de")
    }

    func testNormalizeTextSentenceEndStillCapitalizes() {
        // Default behavior unchanged for sentence-boundary chunks.
        let (text, _) = PocketTtsSynthesizer.normalizeText("hello world")
        XCTAssertTrue(text.contains("Hello"))
        XCTAssertTrue(text.hasSuffix("."))
    }

    // MARK: - Sentence Splitting (issue #584)

    func testSplitSentencesDoesNotSplitOnSmartApostrophe() {
        // After normalizeSmartQuotes runs, sentences containing French
        // contractions should remain a single sentence.
        let input = PocketTtsSynthesizer.normalizeSmartQuotes(
            "Avant d\u{2019}aboutir nous devons l\u{2019}essayer.")
        let sentences = PocketTtsSynthesizer.splitSentences(input)
        XCTAssertEqual(sentences.count, 1, "Expected single sentence, got: \(sentences)")
    }

    func testSplitSentencesDoesNotSplitOnRawU2019() {
        // Even without normalization, U+2019 must never be treated as a
        // sentence terminator (only `.!?` are sentence terminators).
        let sentences = PocketTtsSynthesizer.splitSentences(
            "Avant d\u{2019}aboutir nous devons l\u{2019}essayer")
        XCTAssertEqual(sentences.count, 1)
    }

    func testSplitSentencesSplitsAtPeriods() {
        let sentences = PocketTtsSynthesizer.splitSentences("Hello world. How are you?")
        XCTAssertEqual(sentences.count, 2)
        XCTAssertTrue(sentences[0].hasSuffix("."))
        XCTAssertTrue(sentences[1].hasSuffix("?"))
    }

    func testSplitSentencesHandlesAbbreviations() {
        // "Dr." should not terminate a sentence.
        let sentences = PocketTtsSynthesizer.splitSentences("Dr. Smith arrived.")
        XCTAssertEqual(sentences.count, 1)
    }

    // MARK: - Clause Boundary Splitting

    func testSplitAtClauseBoundariesAtCommas() {
        let parts = PocketTtsSynthesizer.splitAtClauseBoundaries(
            "combustibles, carburants et chauffage")
        XCTAssertEqual(parts.count, 2)
    }

    func testSplitAtClauseBoundariesPreservesNumbers() {
        // "3,500" should not be split.
        let parts = PocketTtsSynthesizer.splitAtClauseBoundaries("about 3,500 units")
        XCTAssertEqual(parts.count, 1)
    }

    // MARK: - TextChunk metadata (issue #584)

    func testTextChunkConstruction() {
        let chunk = PocketTtsSynthesizer.TextChunk(
            text: "hello", isMidSentence: true)
        XCTAssertEqual(chunk.text, "hello")
        XCTAssertTrue(chunk.isMidSentence)
    }

    func testTextChunkEquatable() {
        let a = PocketTtsSynthesizer.TextChunk(text: "x", isMidSentence: false)
        let b = PocketTtsSynthesizer.TextChunk(text: "x", isMidSentence: false)
        let c = PocketTtsSynthesizer.TextChunk(text: "x", isMidSentence: true)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    // MARK: - Configurable maxTokensPerChunk (issue #584)

    func testMaxTokensPerChunkConstantIsExposed() {
        // Verify the default constant is reachable, since it's the default
        // for the new configurable parameter.
        XCTAssertGreaterThan(PocketTtsConstants.maxTokensPerChunk, 0)
    }

    // MARK: - Language Separation (English default vs French)

    func testNormalizeForLanguageEnglishIsNoOp() {
        // English path must not touch guillemets or NBSP — those characters
        // are rare in English text and not part of the SentencePiece vocab
        // gap we're working around for French.
        let input = "He said \u{00AB}hello\u{00BB} to me.\u{00A0}OK?"
        let output = PocketTtsSynthesizer.normalizeForLanguage(
            input, language: .english)
        XCTAssertEqual(output, input)
    }

    func testNormalizeForLanguageFrenchReplacesGuillemets() {
        let input = "Il a dit \u{00AB}bonjour\u{00BB}."
        let output = PocketTtsSynthesizer.normalizeForLanguage(
            input, language: .french24L)
        XCTAssertEqual(output, "Il a dit \"bonjour\".")
    }

    func testNormalizeForLanguageFrenchReplacesNBSP() {
        // French typography puts NBSP (U+00A0) before `! ? : ;`. The
        // tokenizer has no NBSP piece, so we normalize to ASCII space.
        let input = "Vraiment\u{00A0}?"
        let output = PocketTtsSynthesizer.normalizeForLanguage(
            input, language: .french24L)
        XCTAssertEqual(output, "Vraiment ?")
    }

    func testNormalizeForLanguageFrenchReplacesNarrowNBSP() {
        // U+202F (narrow NBSP) is the modern French typography preference;
        // should be normalized identically to regular NBSP.
        let input = "Vraiment\u{202F}?"
        let output = PocketTtsSynthesizer.normalizeForLanguage(
            input, language: .french24L)
        XCTAssertEqual(output, "Vraiment ?")
    }

    func testNormalizeTextDefaultsToEnglishLanguage() {
        // Calling normalizeText without a language must behave exactly like
        // the English path (the default) — ensures backward compatibility.
        let (defaulted, _) = PocketTtsSynthesizer.normalizeText("Hello world")
        let (english, _) = PocketTtsSynthesizer.normalizeText(
            "Hello world", language: .english)
        XCTAssertEqual(defaulted, english)
    }

    func testNormalizeTextFrenchNormalizesGuillemets() {
        let (text, _) = PocketTtsSynthesizer.normalizeText(
            "Il a dit \u{00AB}bonjour\u{00BB}", language: .french24L)
        XCTAssertFalse(text.contains("\u{00AB}"))
        XCTAssertFalse(text.contains("\u{00BB}"))
        XCTAssertTrue(text.contains("\"bonjour\""))
    }

    func testFrenchAbbreviationsTableContainsCivilities() {
        // Spot-check that the French abbreviation set actually has the
        // common civility titles. Used by splitSentences for the French path.
        let abbr = PocketTtsSynthesizer.abbreviations(for: .french24L)
        XCTAssertTrue(abbr.contains("mme"))
        XCTAssertTrue(abbr.contains("mlle"))
        XCTAssertTrue(abbr.contains("dr"))
        XCTAssertTrue(abbr.contains("st"))
    }

    func testAbbreviationsForLanguageEnglishIsDefaultTable() {
        // English language must resolve to the existing default abbreviation
        // table (no regression for existing English callers).
        let english = PocketTtsSynthesizer.abbreviations(for: .english)
        XCTAssertEqual(english, PocketTtsSynthesizer.abbreviations)
    }

    func testSplitSentencesDefaultsToEnglish() {
        // The default-argument overload must match the explicit English call.
        let defaultResult = PocketTtsSynthesizer.splitSentences(
            "Dr. Smith arrived.")
        let englishResult = PocketTtsSynthesizer.splitSentences(
            "Dr. Smith arrived.", language: .english)
        XCTAssertEqual(defaultResult, englishResult)
        XCTAssertEqual(defaultResult.count, 1)
    }

    func testSplitSentencesFrenchHandlesCivilityAbbreviations() {
        // "Mme Dupont" must not be split after "Mme." even if a period
        // appears (e.g. "Mme. Dupont est arrivée."). The English table
        // doesn't include "mme" so this only works on the French path.
        let sentences = PocketTtsSynthesizer.splitSentences(
            "Mme. Dupont est arrivée.", language: .french24L)
        XCTAssertEqual(sentences.count, 1)
    }

    func testSplitSentencesFrenchHandlesReferenceAbbreviations() {
        // "cf." and "p." should not terminate sentences in French.
        let sentences = PocketTtsSynthesizer.splitSentences(
            "Voir cf. p. 42 pour plus de détails.", language: .french24L)
        XCTAssertEqual(sentences.count, 1)
    }

    // MARK: - Issue #584 Reproductions (verbatim reporter samples)
    //
    // These exercise the text-only pieces of the chunker against the exact
    // strings from the issue. Full end-to-end reproduction (tokenize →
    // chunk count → audio) requires loading the French_24l model pack and
    // is covered by CLI benchmark runs, not unit tests.

    func testIssue584Sample1SmartApostropheNoLongerSplitsSentence() {
        // Reporter's sample 1, with U+2019:
        //   "…une proposition susceptible d'aboutir à une trêve, lancée à la suite…"
        // Pre-fix behavior: smart apostrophe inflates token count, sentence
        // gets split into 3 fragments with mid-clause capitalization.
        // Post-fix: smart quote normalized to ASCII, splitSentences sees a
        // single sentence (no `.!?` inside).
        let input =
            "Sa déclaration intervient après des propos récents de Téhéran "
            + "évoquant une proposition susceptible d\u{2019}aboutir à une "
            + "trêve, lancée à la suite des bombardements américains et "
            + "israéliens du 28 février."
        let normalized = PocketTtsSynthesizer.normalizeSmartQuotes(input)
        let sentences = PocketTtsSynthesizer.splitSentences(
            normalized, language: .french24L)
        XCTAssertEqual(
            sentences.count, 1,
            "Issue #584 sample 1 must be one sentence after smart-quote "
                + "normalization; got \(sentences.count): \(sentences)")
        XCTAssertTrue(normalized.contains("d'aboutir"))
        XCTAssertFalse(normalized.contains("\u{2019}"))
    }

    func testIssue584Sample1MidSentenceFragmentNotRecapitalized() {
        // Even if the chunker later splits this sentence at the comma (a
        // clause boundary), the second clause must NOT come out as
        // "D'aboutir à une trêve." — that's exactly the broken-prosody
        // output the reporter showed.
        let clause = "d'aboutir à une trêve"
        let (text, _) = PocketTtsSynthesizer.normalizeText(
            clause, isMidSentence: true, language: .french24L)
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        XCTAssertTrue(
            trimmed.hasPrefix("d"),
            "Mid-sentence clause must not be re-capitalized; got: \(text)")
        XCTAssertFalse(
            trimmed.hasSuffix("."),
            "Mid-sentence clause must not gain a terminal period; got: \(text)")
    }

    func testIssue584Sample1ClauseSplitProducesMidSentencePieces() {
        // The chunker may still split the long sentence at the comma — but
        // the resulting two pieces should be treated as clause boundaries,
        // not fresh sentences. We verify via splitAtClauseBoundaries (the
        // building block used by splitOversizedSentence).
        let input =
            "Sa déclaration intervient après des propos récents de Téhéran "
            + "évoquant une proposition susceptible d'aboutir à une trêve, "
            + "lancée à la suite des bombardements américains et israéliens "
            + "du 28 février."
        let parts = PocketTtsSynthesizer.splitAtClauseBoundaries(input)
        // Two clauses on either side of the comma (the trailing period
        // doesn't introduce a new clause part).
        XCTAssertEqual(
            parts.count, 2,
            "Expected exactly two clause parts at the trêve comma; got: \(parts)")
    }

    func testIssue584Sample2OrphanedPrepositionNotProduced() {
        // Reporter's sample 2 mentioned dangling fragments like
        // "Carburants et chauffage (FF3C)." Word-boundary splitting must
        // donate a word back rather than orphan a single short word at the
        // tail. Using a synthetic tokenizer-free check via splitAtWordBoundaries
        // would require a tokenizer; instead we exercise the equivalent
        // orphan-tail invariant via the public chunk metadata path with the
        // normalizeText preservation rule.
        let (text, _) = PocketTtsSynthesizer.normalizeText(
            "stations-service de", isMidSentence: true, language: .french24L)
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        XCTAssertEqual(
            trimmed, "stations-service de",
            "Orphaned preposition tail must not be capitalized or "
                + "terminated; got: \(text)")
    }

    func testIssue584Sample2MidSentenceCommaTailPreserved() {
        // From the reporter's sample 2 chunking output:
        // "TotalEnergies, qu'elle juge déloyal." — the comma should be
        // preserved when this is a mid-sentence chunk (no extra period,
        // no recapitalization of "qu'elle").
        let (text, _) = PocketTtsSynthesizer.normalizeText(
            "totalenergies, qu'elle juge déloyal",
            isMidSentence: true, language: .french24L)
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        XCTAssertTrue(
            trimmed.hasPrefix("t"),
            "Mid-sentence chunk must keep its leading lowercase; got: \(text)")
        XCTAssertFalse(
            trimmed.hasSuffix("."),
            "Mid-sentence chunk must not append a period; got: \(text)")
    }

    // MARK: - Mid-sentence short-chunk prosody (issue #584 follow-up)

    func testNormalizeTextMidSentenceShortChunkSkipsLeadingPadding() {
        // Mid-sentence continuations under the short-text threshold (e.g. an
        // orphan-tail "stations-service de" at 2 words, or a clause split
        // like "d'aboutir à une trêve" at 4 words) must NOT receive the
        // 8-space leading pad. Otherwise the synthesizer emits silence at
        // the seam, re-creating the prosody break that #584 fixes.
        let (orphan, _) = PocketTtsSynthesizer.normalizeText(
            "stations-service de", isMidSentence: true, language: .french24L)
        XCTAssertFalse(
            orphan.hasPrefix(" "),
            "Mid-sentence short chunk must not be left-padded; got: '\(orphan)'")

        let (clause, _) = PocketTtsSynthesizer.normalizeText(
            "d'aboutir à une trêve", isMidSentence: true, language: .french24L)
        XCTAssertFalse(
            clause.hasPrefix(" "),
            "Mid-sentence short chunk must not be left-padded; got: '\(clause)'")
    }

    func testNormalizeTextMidSentenceShortChunkUsesLongTextExtraFrames() {
        // Mid-sentence short chunks must use the long-text trailing frame
        // budget; the short-text pad value adds extra silence after EOS that
        // shows up as a gap between continuation chunks.
        let (_, orphanFrames) = PocketTtsSynthesizer.normalizeText(
            "stations-service de", isMidSentence: true, language: .french24L)
        XCTAssertEqual(
            orphanFrames, PocketTtsConstants.longTextExtraFrames,
            "Mid-sentence short chunk must not use shortTextPadFrames")

        let (_, clauseFrames) = PocketTtsSynthesizer.normalizeText(
            "d'aboutir à une trêve", isMidSentence: true, language: .french24L)
        XCTAssertEqual(
            clauseFrames, PocketTtsConstants.longTextExtraFrames,
            "Mid-sentence short clause must not use shortTextPadFrames")
    }

    func testNormalizeTextFullSentenceShortChunkStillPads() {
        // The padding behaviour for legitimate short sentences (the original
        // prosody-stabilisation case) must remain unchanged.
        let (text, frames) = PocketTtsSynthesizer.normalizeText(
            "Hi there", isMidSentence: false, language: .english)
        XCTAssertTrue(
            text.hasPrefix(" "),
            "Full short sentence should still be left-padded; got: '\(text)'")
        XCTAssertEqual(
            frames, PocketTtsConstants.shortTextPadFrames,
            "Full short sentence should still use shortTextPadFrames")
    }

    func testNormalizeTextMidSentenceLongChunkUnchanged() {
        // Sanity check: mid-sentence chunks at or above the threshold behave
        // identically to the pre-fix code path (no leading pad, longTextExtraFrames).
        let (text, frames) = PocketTtsSynthesizer.normalizeText(
            "qu'elle juge déloyal en raison de la concurrence",
            isMidSentence: true, language: .french24L)
        XCTAssertFalse(text.hasPrefix(" "))
        XCTAssertEqual(frames, PocketTtsConstants.longTextExtraFrames)
    }
}
