import Foundation
import XCTest

@testable import FluidAudio

final class NemotronStreamingConfigTests: XCTestCase {

    // MARK: - P0: Default Initialization

    func testDefaultInitialization() {
        let config = NemotronStreamingConfig()

        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.melFeatures, 128)
        XCTAssertEqual(config.chunkMelFrames, 112)
        XCTAssertEqual(config.chunkMs, 1120)
        XCTAssertEqual(config.preEncodeCache, 9)
        XCTAssertEqual(config.totalMelFrames, 121)
        XCTAssertEqual(config.vocabSize, 1024)
        XCTAssertEqual(config.blankIdx, 1024)
        XCTAssertEqual(config.encoderDim, 1024)
        XCTAssertEqual(config.decoderHidden, 640)
        XCTAssertEqual(config.decoderLayers, 2)
        XCTAssertEqual(config.cacheChannelShape, [1, 24, 70, 1024])
        XCTAssertEqual(config.cacheTimeShape, [1, 24, 1024, 8])
    }

    func testChunkSamplesComputation() {
        let config = NemotronStreamingConfig()
        // chunkSamples = chunkMelFrames * 160
        XCTAssertEqual(config.chunkSamples, 112 * 160)
        XCTAssertEqual(config.chunkSamples, 17920)
    }

    // MARK: - P0: JSON Loading - Valid Cases

    func testLoadValidMetadataJson1120ms() throws {
        let metadata: [String: Any] = [
            "sample_rate": 16000,
            "mel_features": 128,
            "chunk_mel_frames": 112,
            "chunk_ms": 1120,
            "pre_encode_cache": 9,
            "total_mel_frames": 121,
            "vocab_size": 1024,
            "blank_idx": 1024,
            "encoder_dim": 1024,
            "decoder_hidden": 640,
            "decoder_layers": 2,
            "cache_channel_shape": [1, 24, 70, 1024],
            "cache_time_shape": [1, 24, 1024, 8],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.melFeatures, 128)
        XCTAssertEqual(config.chunkMelFrames, 112)
        XCTAssertEqual(config.chunkMs, 1120)
        XCTAssertEqual(config.preEncodeCache, 9)
        XCTAssertEqual(config.totalMelFrames, 121)
        XCTAssertEqual(config.vocabSize, 1024)
        XCTAssertEqual(config.blankIdx, 1024)
        XCTAssertEqual(config.encoderDim, 1024)
        XCTAssertEqual(config.decoderHidden, 640)
        XCTAssertEqual(config.decoderLayers, 2)
        XCTAssertEqual(config.cacheChannelShape, [1, 24, 70, 1024])
        XCTAssertEqual(config.cacheTimeShape, [1, 24, 1024, 8])
    }

    func testLoadValidMetadataJson80ms() throws {
        let metadata: [String: Any] = [
            "sample_rate": 16000,
            "mel_features": 128,
            "chunk_mel_frames": 8,
            "chunk_ms": 80,
            "pre_encode_cache": 4,
            "total_mel_frames": 12,
            "vocab_size": 1024,
            "blank_idx": 1024,
            "encoder_dim": 1024,
            "decoder_hidden": 640,
            "decoder_layers": 2,
            "cache_channel_shape": [1, 24, 120, 1024],  // Larger cache for tiny chunks
            "cache_time_shape": [1, 24, 1024, 10],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.chunkMelFrames, 8)
        XCTAssertEqual(config.chunkMs, 80)
        XCTAssertEqual(config.preEncodeCache, 4)
        XCTAssertEqual(config.totalMelFrames, 12)
        XCTAssertEqual(config.chunkSamples, 8 * 160)  // 1280 samples
        XCTAssertEqual(config.cacheChannelShape, [1, 24, 120, 1024])
    }

    func testLoadValidMetadataJson560ms() throws {
        let metadata: [String: Any] = [
            "chunk_mel_frames": 56,
            "chunk_ms": 560,
            "pre_encode_cache": 7,
            "total_mel_frames": 63,
            "cache_channel_shape": [1, 24, 85, 1024],
            "cache_time_shape": [1, 24, 1024, 9],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.chunkMelFrames, 56)
        XCTAssertEqual(config.chunkMs, 560)
        XCTAssertEqual(config.preEncodeCache, 7)
        XCTAssertEqual(config.totalMelFrames, 63)
        XCTAssertEqual(config.chunkSamples, 56 * 160)  // 8960 samples
    }

    // MARK: - P0: JSON Loading - Fallback Defaults

    func testLoadPartialJsonUsesDefaults() throws {
        let metadata: [String: Any] = [
            "chunk_mel_frames": 56  // Only this field present
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        // Custom value
        XCTAssertEqual(config.chunkMelFrames, 56)

        // Defaults
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.melFeatures, 128)
        XCTAssertEqual(config.vocabSize, 1024)
        XCTAssertEqual(config.blankIdx, 1024)
    }

    func testLoadEmptyJsonUsesAllDefaults() throws {
        let metadata: [String: Any] = [:]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        // All defaults (same as default init)
        XCTAssertEqual(config.chunkMelFrames, 112)
        XCTAssertEqual(config.chunkMs, 1120)
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.vocabSize, 1024)
    }

    // MARK: - P0: JSON Loading - Error Cases

    func testLoadInvalidJsonFormatThrows() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let file = tempDir.appendingPathComponent("invalid_\(UUID().uuidString).json")

        try "not json at all".write(to: file, atomically: true, encoding: .utf8)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: file)
        }

        XCTAssertThrowsError(try NemotronStreamingConfig(from: file)) { error in
            // Should throw JSON parsing error
            XCTAssertTrue(error is NSError || error is DecodingError)
        }
    }

    func testLoadJsonArrayInsteadOfDictionaryThrows() throws {
        let jsonArray = try JSONSerialization.data(withJSONObject: [1, 2, 3], options: [])
        let file = try createTempFile(jsonArray)

        XCTAssertThrowsError(try NemotronStreamingConfig(from: file)) { error in
            guard case ASRError.processingFailed(let message) = error else {
                XCTFail("Expected ASRError.processingFailed, got \(error)")
                return
            }
            XCTAssertEqual(message, "Invalid metadata.json format")
        }
    }

    func testLoadNonExistentFileThrows() {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(
            "nonexistent_\(UUID().uuidString).json")

        XCTAssertThrowsError(try NemotronStreamingConfig(from: file))
    }

    // MARK: - P0: Type Coercion

    func testLoadJsonWithWrongTypesUsesDefaults() throws {
        let metadata: [String: Any] = [
            "chunk_mel_frames": "not a number",  // Wrong type
            "sample_rate": 16000,  // Correct type
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        // Wrong type → uses default
        XCTAssertEqual(config.chunkMelFrames, 112)  // Default value

        // Correct type → uses provided
        XCTAssertEqual(config.sampleRate, 16000)
    }

    func testLoadJsonWithArrayTypesCorrectly() throws {
        let metadata: [String: Any] = [
            "cache_channel_shape": [1, 24, 100, 2048],
            "cache_time_shape": [1, 24, 2048, 16],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.cacheChannelShape, [1, 24, 100, 2048])
        XCTAssertEqual(config.cacheTimeShape, [1, 24, 2048, 16])
    }

    // MARK: - Helpers

    private func createTempJsonFile(_ dict: [String: Any]) throws -> URL {
        let data = try JSONSerialization.data(withJSONObject: dict, options: [])
        return try createTempFile(data)
    }

    private func createTempFile(_ data: Data) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let file = tempDir.appendingPathComponent("test_metadata_\(UUID().uuidString).json")
        try data.write(to: file)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: file)
        }
        return file
    }
}
