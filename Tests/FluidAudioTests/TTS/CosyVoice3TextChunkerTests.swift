import XCTest

@testable import FluidAudio

final class CosyVoice3TextChunkerTests: XCTestCase {

    // MARK: - estimateSpeechTokens

    func testEstimateSpeechTokensCJK() {
        // 4 CJK chars × 7.5 = 30 tokens
        XCTAssertEqual(CosyVoice3TextChunker.estimateSpeechTokens("你好世界"), 30)
    }

    func testEstimateSpeechTokensASCII() {
        // 5 ASCII chars × 1.5 = 7.5 → rounds to 8
        XCTAssertEqual(CosyVoice3TextChunker.estimateSpeechTokens("hello"), 8)
    }

    func testEstimateSpeechTokensEmpty() {
        XCTAssertEqual(CosyVoice3TextChunker.estimateSpeechTokens(""), 0)
    }

    // MARK: - chunk: short input fast path

    func testChunkEmptyReturnsEmpty() {
        XCTAssertEqual(CosyVoice3TextChunker.chunk(""), [])
        XCTAssertEqual(CosyVoice3TextChunker.chunk("   "), [])
        XCTAssertEqual(CosyVoice3TextChunker.chunk("\n\n"), [])
    }

    func testChunkShortReturnsSingle() {
        // 5 chars (4 CJK + 「。」) ≈ 33 tokens, well under default 110
        XCTAssertEqual(
            CosyVoice3TextChunker.chunk("你好世界。"),
            ["你好世界。"])
    }

    func testChunkShortTrimsWhitespace() {
        XCTAssertEqual(
            CosyVoice3TextChunker.chunk("  hello world.  "),
            ["hello world."])
    }

    // MARK: - chunk: hard sentence enders

    func testChunkSplitsOnHardEnders() {
        // 25 CJK chars × 7.5 = 187.5 tokens > 110 default → must split
        let text = "今天天气很好。我们去公园散步。明天可能会下雨。下周打算去看电影。"
        let chunks = CosyVoice3TextChunker.chunk(text)
        XCTAssertGreaterThan(chunks.count, 1)
        // No chunk should exceed budget by more than the soft margin
        for chunk in chunks {
            let est = CosyVoice3TextChunker.estimateSpeechTokens(chunk)
            XCTAssertLessThanOrEqual(est, 110 + 30 + 8, "chunk over force-split margin: \(chunk)")
        }
        // Concatenating chunks back should reconstruct the input modulo
        // whitespace trimming.
        XCTAssertEqual(chunks.joined(), text)
    }

    func testChunkSplitsOnEnglishSentenceEnders() {
        // Each sentence ≈ 25–30 tokens; with maxSpeechTokens=80 every
        // sentence fits individually so the chunker should commit on the
        // first hard ender it sees rather than packing greedily across
        // sentences and hitting force-split.
        let text = "Hello world. This is a test. Pack my box with five jugs. Quick brown fox jumps."
        let chunks = CosyVoice3TextChunker.chunk(text, maxSpeechTokens: 80)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertTrue(
                chunk.hasSuffix(".") || chunk.hasSuffix("!") || chunk.hasSuffix("?"),
                "chunk does not end at hard boundary: \(chunk)")
        }
    }

    // MARK: - chunk: soft enders fall-through

    func testChunkFallsBackToSoftEnders() {
        // One huge sentence with commas, no periods. Should split on 「，」.
        let text = "一个非常非常长的句子，里面有很多分句，每个分句都不是很长，但是加在一起就会超过预算限制"
        let chunks = CosyVoice3TextChunker.chunk(text, maxSpeechTokens: 50)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            let est = CosyVoice3TextChunker.estimateSpeechTokens(chunk)
            // Force-split allows one CJK char of overshoot past the +30 margin
            // because the budget check runs AFTER appending the current char.
            XCTAssertLessThanOrEqual(est, 50 + 30 + 8)
        }
    }

    // MARK: - chunk: force-split fallback

    func testChunkForceSplitsOnContinuousCJKWithoutPunctuation() {
        // 30 CJK chars, no punctuation: ≈ 225 tokens, must force-split
        // somewhere even without natural boundaries.
        let text = "今天天气很好我们去公园散步明天可能会下雨下周打算看电影然后回家"
        let chunks = CosyVoice3TextChunker.chunk(text, maxSpeechTokens: 50)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            let est = CosyVoice3TextChunker.estimateSpeechTokens(chunk)
            // Force-split has a 30-token overshoot allowance + one CJK char (7.5)
            XCTAssertLessThanOrEqual(est, 50 + 30 + 8, "chunk overflow on force-split: \(chunk)")
        }
        // No content lost
        XCTAssertEqual(chunks.joined(), text)
    }

    func testChunkForceSplitsOnEnglishSpacesWhenNoPunctuation() {
        // Long English with no terminal punctuation; should split on spaces
        // when the running estimate exceeds budget.
        let text = "the quick brown fox jumps over the lazy dog and then runs back home very fast"
        let chunks = CosyVoice3TextChunker.chunk(text, maxSpeechTokens: 20)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            // No leading/trailing whitespace expected on returned chunks
            XCTAssertEqual(chunk, chunk.trimmingCharacters(in: .whitespaces))
        }
    }

    // MARK: - concatWithCrossfade

    func testConcatEmptyReturnsEmpty() {
        let out = CosyVoice3TtsManager.concatWithCrossfade(
            [], sampleRate: 24_000, fadeMs: 8)
        XCTAssertEqual(out, [])
    }

    func testConcatSingleChunkPassthrough() {
        let chunk: [Float] = [0.1, 0.2, 0.3, 0.4]
        let out = CosyVoice3TtsManager.concatWithCrossfade(
            [chunk], sampleRate: 24_000, fadeMs: 8)
        XCTAssertEqual(out, chunk)
    }

    func testConcatZeroFadeIsSimpleAppend() {
        let a: [Float] = [0.1, 0.2, 0.3]
        let b: [Float] = [0.4, 0.5, 0.6]
        let out = CosyVoice3TtsManager.concatWithCrossfade(
            [a, b], sampleRate: 24_000, fadeMs: 0)
        XCTAssertEqual(out, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    }

    func testConcatCrossfadeShrinksGracefullyForShortChunks() {
        // 4-sample chunks; nominal fade at 24 kHz × 8 ms = 192 samples,
        // gets clamped to min(out.count/2, next.count/2) = 2.
        let a: [Float] = [1.0, 1.0, 1.0, 1.0]
        let b: [Float] = [0.0, 0.0, 0.0, 0.0]
        let out = CosyVoice3TtsManager.concatWithCrossfade(
            [a, b], sampleRate: 24_000, fadeMs: 8)
        // Output length: 4 (a) - 2 (fade) + 4 (b) = 6; first 2 of a remain
        // pristine, then a 2-sample crossfade region, then last 2 of b
        XCTAssertEqual(out.count, 6)
        XCTAssertEqual(out[0], 1.0)
        XCTAssertEqual(out[1], 1.0)
        // Crossfade region: a's 1.0 fades to 0; b's 0.0 fades from 0.
        // At j=0: down=1, up=0 → 1.0 * 1 + 0.0 * 0 = 1.0
        // At j=1: down=0.5, up=0.5 → 1.0*0.5 + 0.0*0.5 = 0.5
        XCTAssertEqual(out[2], 1.0, accuracy: 1e-5)
        XCTAssertEqual(out[3], 0.5, accuracy: 1e-5)
        XCTAssertEqual(out[4], 0.0, accuracy: 1e-5)
        XCTAssertEqual(out[5], 0.0, accuracy: 1e-5)
    }

    func testConcatCrossfadePreservesPrefixAndSuffix() {
        // Long enough chunks for a full fade window
        let sampleRate = 24_000
        let fadeMs = 4.0  // 96 samples
        let a = [Float](repeating: 1.0, count: 480)
        let b = [Float](repeating: 0.0, count: 480)
        let out = CosyVoice3TtsManager.concatWithCrossfade(
            [a, b], sampleRate: sampleRate, fadeMs: fadeMs)
        let fade = Int((Double(sampleRate) * fadeMs / 1000).rounded())
        // Output length: a.count - fade + b.count
        XCTAssertEqual(out.count, a.count - fade + b.count)
        // Prefix of `a` (before crossfade region) untouched
        for j in 0..<(a.count - fade) {
            XCTAssertEqual(out[j], 1.0)
        }
        // Suffix of `b` (after crossfade region) untouched
        for j in (a.count..<out.count) {
            XCTAssertEqual(out[j - 0], 0.0)
        }
    }
}
