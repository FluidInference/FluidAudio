import XCTest

@testable import FluidAudio

final class MossTtsNanoPromptBuilderTests: XCTestCase {

    /// Decodes a `config.json` matching the published layout (template ids abbreviated).
    private func makeConfig() throws -> MossTtsNanoConfig {
        let json = """
            {
              "model": {"n_vq": 16, "row_width": 17, "hidden_size": 768, "audio_codebook_size": 1024,
                        "sample_rate": 48000, "channels": 2, "samples_per_frame_per_channel": 3840},
              "coreml": {"prefill_rows": 512, "max_len": 1024},
              "tokens": {"pad": 3, "im_start": 4, "im_end": 5, "audio_start": 6, "audio_end": 7,
                         "audio_user_slot": 8, "audio_assistant_slot": 9, "audio_pad": 1024},
              "prompt": {"voice_clone_prefix": [4, 600, 6], "voice_clone_after_reference": [7, 10356],
                         "assistant_suffix": [5, 4, 6], "plain_prefix": [4]},
              "sampling_defaults": {"text_temperature": 1.5, "text_top_k": 50, "text_top_p": 1.0,
                                    "audio_temperature": 1.7, "audio_top_k": 25, "audio_top_p": 0.8,
                                    "audio_repetition_penalty": 1.0, "max_new_frames": 375},
              "text": {"voice_clone_max_text_tokens": 50}
            }
            """
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(MossTtsNanoConfig.self, from: Data(json.utf8))
    }

    func testVoiceCloneRowLayout() throws {
        let config = try makeConfig()
        let builder = MossTtsNanoPromptBuilder(config: config)
        let codes: [[Int32]] = [Array(0..<16), Array(100..<116)]
        let voice = MossTtsNanoVoice(name: "t", codes: codes)
        let rows = builder.voiceCloneRows(textIds: [42, 43], voice: voice)

        XCTAssertEqual(rows.count, 3 + 2 + 2 + 2 + 3)
        XCTAssertTrue(rows.allSatisfy { $0.count == 17 })
        // Text rows: token in column 0, audio pad elsewhere.
        XCTAssertEqual(rows[0], [4] + [Int32](repeating: 1024, count: 16))
        XCTAssertEqual(rows[2][0], 6)
        // Reference rows: user slot + the clip's codes.
        XCTAssertEqual(rows[3], [8] + Array(0..<16))
        XCTAssertEqual(rows[4], [8] + Array(100..<116))
        // After-reference, text, suffix.
        XCTAssertEqual(rows[5][0], 7)
        XCTAssertEqual(rows[7][0], 42)
        XCTAssertEqual(rows[8][0], 43)
        XCTAssertEqual(rows.last?[0], 6)
        XCTAssertEqual(builder.voiceCloneOverheadRows(voice: voice), 3 + 2 + 2 + 3)
    }

    func testGenerationRowUsesAssistantSlot() throws {
        let builder = MossTtsNanoPromptBuilder(config: try makeConfig())
        let row = builder.generationRow(codes: Array(200..<216))
        XCTAssertEqual(row[0], 9)
        XCTAssertEqual(Array(row[1...]), Array(200..<216))
    }

    func testVoiceValidation() {
        XCTAssertNoThrow(try MossTtsNanoVoice(name: "ok", codes: [Array(0..<16)]).validate())
        XCTAssertThrowsError(try MossTtsNanoVoice(name: "short", codes: [Array(0..<15)]).validate())
        XCTAssertThrowsError(
            try MossTtsNanoVoice(name: "range", codes: [[Int32](repeating: 1024, count: 16)]).validate())
        XCTAssertThrowsError(try MossTtsNanoVoice(name: "empty", codes: []).validate())
    }

    func testBuiltInVoiceNames() {
        XCTAssertEqual(MossTtsNanoBuiltInVoice(name: "en-2"), .en2)
        XCTAssertEqual(MossTtsNanoBuiltInVoice(name: "ZH_1"), .zh1)
        XCTAssertNil(MossTtsNanoBuiltInVoice(name: "af_heart"))
        XCTAssertEqual(MossTtsNanoBuiltInVoice.en2.fileName, "voices/en_2.json")
    }
}
