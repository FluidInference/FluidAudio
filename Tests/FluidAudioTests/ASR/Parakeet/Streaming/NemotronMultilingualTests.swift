import Foundation
import XCTest

@testable import FluidAudio

final class NemotronMultilingualTests: XCTestCase {

    // MARK: - Config

    func testDefaultConfigShape() {
        let config = NemotronMultilingualStreamingConfig()
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.melFeatures, 128)
        XCTAssertEqual(config.chunkMelFrames, 112)
        XCTAssertEqual(config.chunkMs, 1120)
        XCTAssertEqual(config.preEncodeCache, 9)
        XCTAssertEqual(config.totalMelFrames, 121)
        XCTAssertEqual(config.vocabSize, 13087)
        XCTAssertEqual(config.blankIdx, 13087)
        XCTAssertEqual(config.cacheChannelShape, [1, 24, 56, 1024])
        XCTAssertEqual(config.cacheTimeShape, [1, 24, 1024, 8])
        XCTAssertEqual(config.defaultPromptId, 101)
        XCTAssertEqual(config.chunkSamples, 112 * 160)
    }

    func testConfigLoadFromMetadata() throws {
        // Stand-in metadata.json matching the multilingual build format.
        let json: [String: Any] = [
            "sample_rate": 16000,
            "mel_features": 128,
            "chunk_mel_frames": 112,
            "chunk_ms": 1120,
            "pre_encode_cache": 9,
            "total_mel_frames": 121,
            "vocab_size": 13087,
            "blank_idx": 13087,
            "encoder_dim": 1024,
            "decoder_hidden": 640,
            "decoder_layers": 2,
            "cache_channel_shape": [1, 24, 56, 1024],
            "cache_time_shape": [1, 24, 1024, 8],
            "num_prompts": 128,
            "default_prompt_id": 101,
            "prompt_dictionary": [
                "en-US": 0,
                "zh-CN": 4,
                "ja-JP": 10,
                "fr-FR": 12,
                "auto": 101,
            ],
            "lang_tag_token_ids": [1, 256, 397],
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_metadata_test_\(UUID().uuidString).json")
        try data.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let config = try NemotronMultilingualStreamingConfig(from: tmpURL)
        XCTAssertEqual(config.numPrompts, 128)
        XCTAssertEqual(config.defaultPromptId, 101)
        XCTAssertEqual(config.promptDictionary["en-US"], 0)
        XCTAssertEqual(config.promptDictionary["zh-CN"], 4)
        XCTAssertEqual(config.promptDictionary["auto"], 101)
        XCTAssertEqual(config.langTagTokenIds, Set([1, 256, 397]))
    }

    // MARK: - promptId(forLanguage:)

    func testPromptIdDirectLookup() throws {
        let config = try makeConfig(
            promptDictionary: ["en-US": 0, "zh-CN": 4, "ja-JP": 10, "auto": 101]
        )
        XCTAssertEqual(config.promptId(forLanguage: "en-US"), 0)
        XCTAssertEqual(config.promptId(forLanguage: "zh-CN"), 4)
        XCTAssertEqual(config.promptId(forLanguage: "ja-JP"), 10)
    }

    func testPromptIdNilFallsBackToDefault() throws {
        let config = try makeConfig(promptDictionary: ["en-US": 0, "auto": 101])
        XCTAssertEqual(config.promptId(forLanguage: nil), 101)
        XCTAssertEqual(config.promptId(forLanguage: ""), 101)
    }

    func testPromptIdUnderscoreNormalization() throws {
        let config = try makeConfig(promptDictionary: ["en-US": 0, "auto": 101])
        // "en_us" should normalize to "en-US"
        XCTAssertEqual(config.promptId(forLanguage: "en_us"), 0)
        XCTAssertEqual(config.promptId(forLanguage: "EN-us"), 0)
    }

    func testPromptIdBareLanguageFallback() throws {
        let config = try makeConfig(promptDictionary: ["en": 7, "auto": 101])
        // "en-XX" should fall back to bare "en"
        XCTAssertEqual(config.promptId(forLanguage: "en-XX"), 7)
    }

    func testPromptIdUnknownLanguageReturnsDefault() throws {
        let config = try makeConfig(promptDictionary: ["en-US": 0, "auto": 101])
        XCTAssertEqual(config.promptId(forLanguage: "xx-YY"), 101)
    }

    // MARK: - Tokenizer

    func testTokenizerStripAngleBrackets() {
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets("<en-US>"), "en-US")
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets("<zh-CN>"), "zh-CN")
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets("no-brackets"), "no-brackets")
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets("<>"), "")
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets(""), "")
    }

    func testTokenizerFiltersLangTagsAndSurfacesDetectedLanguage() throws {
        // Synthesize a minimal vocab JSON: {"id": "piece"}
        // Token 1 is `<en-US>` (lang tag), 2 is `▁hello`, 3 is `▁world`.
        let vocab: [String: String] = [
            "0": "<unk>",
            "1": "<en-US>",
            "2": "\u{2581}hello",
            "3": "\u{2581}world",
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_vocab_test_\(UUID().uuidString).json")
        try vocabData.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tmpURL,
            langTagTokenIds: Set([1])
        )
        let decoded = tokenizer.decode(ids: [1, 2, 3])
        XCTAssertEqual(decoded.text, "hello world")
        XCTAssertEqual(decoded.detectedLanguage, "en-US")
    }

    func testTokenizerWithNoLangTag() throws {
        let vocab: [String: String] = [
            "0": "<unk>",
            "1": "<en-US>",
            "2": "\u{2581}hi",
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_vocab_test_\(UUID().uuidString).json")
        try vocabData.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tmpURL,
            langTagTokenIds: Set([1])
        )
        let decoded = tokenizer.decode(ids: [2])
        XCTAssertEqual(decoded.text, "hi")
        XCTAssertNil(decoded.detectedLanguage)
    }

    func testRawTokenPreservesWordBoundaryMarker() throws {
        // rawToken must return the UNMODIFIED SentencePiece vocab piece, with the
        // `▁` word-boundary marker intact, so callers can group per-token timings
        // into words. decode()/the visible transcript strip `▁`; rawToken must not,
        // otherwise word starts can't be located and word-level timing breaks.
        let vocab: [String: String] = [
            "0": "<unk>",
            "1": "\u{2581}hello",  // word-start piece (has ▁)
            "2": "ing",  // mid-word continuation (no ▁)
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_vocab_test_\(UUID().uuidString).json")
        try vocabData.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tmpURL,
            langTagTokenIds: Set<Int>()
        )

        // Word-start piece keeps the `▁` marker...
        XCTAssertEqual(tokenizer.rawToken(for: 1), "\u{2581}hello")
        // ...continuation piece has no marker...
        XCTAssertEqual(tokenizer.rawToken(for: 2), "ing")
        // ...the visible transcript strips the marker (why callers need rawToken)...
        XCTAssertFalse(tokenizer.decode(ids: [1]).text.contains("\u{2581}"))
        // ...and an out-of-vocab id returns nil so the caller skips its timing.
        XCTAssertNil(tokenizer.rawToken(for: 999))
    }

    // MARK: - ModelNames

    func testNemotronMultilingualModelNames() {
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.preprocessorFile.hasSuffix(".mlmodelc"))
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.encoderFile.hasSuffix(".mlmodelc"))
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.decoderFile.hasSuffix(".mlmodelc"))
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.jointFile.hasSuffix(".mlmodelc"))
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.preprocessorPackage.hasSuffix(".mlpackage"))
        XCTAssertEqual(ModelNames.NemotronMultilingualStreaming.tokenizer, "tokenizer.json")
        XCTAssertEqual(ModelNames.NemotronMultilingualStreaming.metadata, "metadata.json")
    }

    // MARK: - Helpers

    private func makeConfig(
        promptDictionary: [String: Int],
        defaultPromptId: Int = 101
    ) throws -> NemotronMultilingualStreamingConfig {
        let json: [String: Any] = [
            "prompt_dictionary": promptDictionary,
            "default_prompt_id": defaultPromptId,
            "num_prompts": 128,
            "lang_tag_token_ids": [Int](),
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_cfg_\(UUID().uuidString).json")
        try data.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        return try NemotronMultilingualStreamingConfig(from: tmpURL)
    }
}
