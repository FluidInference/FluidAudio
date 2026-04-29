import Foundation
import XCTest

@testable import FluidAudio

/// Pure-logic unit tests for the StyleTTS2 `config.json` loader + validator.
///
/// Uses inline JSON fixtures so the suite runs without HuggingFace
/// downloads or any CoreML models.
final class StyleTTS2BundleConfigTests: XCTestCase {

    /// Canonical fixture mirroring the live `config.json` shipped on
    /// `FluidInference/StyleTTS-2-coreml`. Trimmed to the must-have fields
    /// the loader actually reads — extra keys in the on-disk file are
    /// ignored by `Codable` decoding.
    private let goodJSON: String = #"""
        {
          "model_type": "styletts2",
          "audio": {
            "sample_rate": 24000,
            "n_fft": 2048,
            "win_length": 1200,
            "hop_length": 300,
            "n_mels": 80
          },
          "tokenizer": {
            "type": "espeak-ng-ipa",
            "vocab_file": "constants/text_cleaner_vocab.json",
            "n_tokens": 178,
            "pad_token": "$",
            "pad_id": 0
          },
          "model": {
            "style_dim": 128,
            "hidden_dim": 512,
            "n_layer": 3,
            "max_dur": 50,
            "ref_s_dim": 256
          },
          "sampler": {
            "type": "ADPM2",
            "schedule": "karras",
            "num_steps": 5,
            "classifier_free_guidance": true,
            "cfg_scale_default": 1.0
          }
        }
        """#

    private func writeFixture(_ json: String, name: String = "config.json") throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("styletts2-config-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try json.data(using: .utf8)!.write(to: url)
        return url
    }

    /// Replace a single integer field inside the canonical fixture. We use
    /// a string-replace rather than re-encoding via `JSONSerialization` so
    /// the diff in test output is obvious.
    private func goodJSONReplacing(_ field: String, with value: String) -> String {
        let pattern = "\"\(field)\": "
        guard let range = goodJSON.range(of: pattern) else {
            XCTFail("Test bug: field \"\(field)\" not in fixture")
            return goodJSON
        }
        // Replace from end-of-pattern to the next "," or "\n"
        let tail = goodJSON[range.upperBound...]
        guard let term = tail.firstIndex(where: { $0 == "," || $0 == "\n" }) else {
            return goodJSON
        }
        return goodJSON.replacingCharacters(
            in: range.upperBound..<term, with: value)
    }

    // MARK: - load

    func testLoadParsesCanonicalConfig() throws {
        let url = try writeFixture(goodJSON)
        let config = try StyleTTS2BundleConfig.load(from: url)

        XCTAssertEqual(config.modelType, "styletts2")
        XCTAssertEqual(config.audio.sampleRate, 24_000)
        XCTAssertEqual(config.audio.nFFT, 2_048)
        XCTAssertEqual(config.audio.winLength, 1_200)
        XCTAssertEqual(config.audio.hopLength, 300)
        XCTAssertEqual(config.audio.nMels, 80)

        XCTAssertEqual(config.tokenizer.type, "espeak-ng-ipa")
        XCTAssertEqual(config.tokenizer.vocabFile, "constants/text_cleaner_vocab.json")
        XCTAssertEqual(config.tokenizer.nTokens, 178)
        XCTAssertEqual(config.tokenizer.padToken, "$")
        XCTAssertEqual(config.tokenizer.padId, 0)

        XCTAssertEqual(config.model.styleDim, 128)
        XCTAssertEqual(config.model.hiddenDim, 512)
        XCTAssertEqual(config.model.nLayer, 3)
        XCTAssertEqual(config.model.maxDur, 50)
        XCTAssertEqual(config.model.refStyleDim, 256)

        XCTAssertEqual(config.sampler.type, "ADPM2")
        XCTAssertEqual(config.sampler.schedule, "karras")
        XCTAssertEqual(config.sampler.numSteps, 5)
        XCTAssertTrue(config.sampler.classifierFreeGuidance)
        XCTAssertEqual(config.sampler.cfgScaleDefault, 1.0, accuracy: 1e-6)
    }

    func testLoadFailsForMissingFile() {
        let bogusURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).json")
        XCTAssertThrowsError(try StyleTTS2BundleConfig.load(from: bogusURL)) { error in
            guard case StyleTTS2Error.modelNotFound = error else {
                XCTFail("expected modelNotFound, got \(error)")
                return
            }
        }
    }

    func testLoadFailsForMalformedJSON() throws {
        let url = try writeFixture("not actually json", name: "bad.json")
        XCTAssertThrowsError(try StyleTTS2BundleConfig.load(from: url)) { error in
            guard case StyleTTS2Error.invalidConfiguration = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testLoadFailsWhenRequiredFieldMissing() throws {
        // Drop the audio block entirely — Codable should reject.
        let bad = #"""
            {
              "model_type": "styletts2",
              "tokenizer": {
                "type": "espeak-ng-ipa",
                "vocab_file": "constants/text_cleaner_vocab.json",
                "n_tokens": 178,
                "pad_token": "$",
                "pad_id": 0
              },
              "model": {
                "style_dim": 128, "hidden_dim": 512, "n_layer": 3,
                "max_dur": 50, "ref_s_dim": 256
              },
              "sampler": {
                "type": "ADPM2", "schedule": "karras",
                "num_steps": 5, "classifier_free_guidance": true,
                "cfg_scale_default": 1.0
              }
            }
            """#
        let url = try writeFixture(bad, name: "missing-audio.json")
        XCTAssertThrowsError(try StyleTTS2BundleConfig.load(from: url)) { error in
            guard case StyleTTS2Error.invalidConfiguration = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    // MARK: - validate

    func testValidatePassesForCanonicalConfig() throws {
        let url = try writeFixture(goodJSON)
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertNoThrow(try config.validate())
    }

    func testValidateFailsOnWrongModelType() throws {
        let bad = goodJSON.replacingOccurrences(
            of: "\"model_type\": \"styletts2\"",
            with: "\"model_type\": \"kokoro\"")
        let url = try writeFixture(bad, name: "wrong-model-type.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("model_type"))
        }
    }

    func testValidateFailsOnSampleRateMismatch() throws {
        let bad = goodJSONReplacing("sample_rate", with: "16000")
        let url = try writeFixture(bad, name: "wrong-sr.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("audio.sample_rate"))
        }
    }

    func testValidateFailsOnHopLengthMismatch() throws {
        let bad = goodJSONReplacing("hop_length", with: "256")
        let url = try writeFixture(bad, name: "wrong-hop.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("audio.hop_length"))
        }
    }

    func testValidateFailsOnNMelsMismatch() throws {
        let bad = goodJSONReplacing("n_mels", with: "128")
        let url = try writeFixture(bad, name: "wrong-mels.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("audio.n_mels"))
        }
    }

    func testValidateFailsOnNFFTMismatch() throws {
        let bad = goodJSONReplacing("n_fft", with: "1024")
        let url = try writeFixture(bad, name: "wrong-nfft.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("audio.n_fft"))
        }
    }

    func testValidateFailsOnWinLengthMismatch() throws {
        let bad = goodJSONReplacing("win_length", with: "800")
        let url = try writeFixture(bad, name: "wrong-win.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("audio.win_length"))
        }
    }

    func testValidateFailsOnNTokensMismatch() throws {
        let bad = goodJSONReplacing("n_tokens", with: "256")
        let url = try writeFixture(bad, name: "wrong-ntokens.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("tokenizer.n_tokens"))
        }
    }

    func testValidateFailsOnPadIdMismatch() throws {
        let bad = goodJSONReplacing("pad_id", with: "1")
        let url = try writeFixture(bad, name: "wrong-padid.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("tokenizer.pad_id"))
        }
    }

    func testValidateFailsOnStyleDimMismatch() throws {
        let bad = goodJSONReplacing("style_dim", with: "64")
        let url = try writeFixture(bad, name: "wrong-style.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("model.style_dim"))
        }
    }

    func testValidateFailsOnHiddenDimMismatch() throws {
        let bad = goodJSONReplacing("hidden_dim", with: "256")
        let url = try writeFixture(bad, name: "wrong-hidden.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("model.hidden_dim"))
        }
    }

    func testValidateFailsOnRefSDimMismatch() throws {
        let bad = goodJSONReplacing("ref_s_dim", with: "128")
        let url = try writeFixture(bad, name: "wrong-refs.json")
        let config = try StyleTTS2BundleConfig.load(from: url)
        XCTAssertThrowsError(try config.validate()) { error in
            guard case StyleTTS2Error.invalidConfiguration(let msg) = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("model.ref_s_dim"))
        }
    }
}
