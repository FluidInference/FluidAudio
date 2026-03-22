import Foundation
import XCTest

@testable import FluidAudio

final class Qwen3TtsConstantsTests: XCTestCase {

    // MARK: - Constants Validation

    func testAudioSampleRate() {
        XCTAssertEqual(Qwen3TtsConstants.audioSampleRate, 24_000)
    }

    func testMaxCodecTokens() {
        XCTAssertGreaterThan(Qwen3TtsConstants.maxCodecTokens, 0)
    }

    func testMaxTextLength() {
        XCTAssertGreaterThan(Qwen3TtsConstants.maxTextLength, 0)
    }

    func testCodecEosTokenId() {
        XCTAssertGreaterThan(Qwen3TtsConstants.codecEosTokenId, 0)
    }

    func testSamplingParameters() {
        XCTAssertGreaterThan(Qwen3TtsConstants.cpTemperature, 0)
        XCTAssertGreaterThan(Qwen3TtsConstants.cpTopK, 0)
        XCTAssertGreaterThan(Qwen3TtsConstants.repetitionPenalty, 1.0)
    }

    func testMinNewTokensIsReasonable() {
        XCTAssertGreaterThanOrEqual(Qwen3TtsConstants.minNewTokens, 0)
        XCTAssertLessThan(Qwen3TtsConstants.minNewTokens, Qwen3TtsConstants.maxCodecTokens)
    }

    // MARK: - Model Names

    func testQwen3TtsRequiredModelsNonEmpty() {
        XCTAssertFalse(ModelNames.Qwen3TTS.requiredModels.isEmpty)
    }

    func testQwen3TtsRequiredModelsContainCoreModels() {
        let required = ModelNames.Qwen3TTS.requiredModels
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.lmPrefillFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.lmDecodeFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.cpPrefillFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.cpDecodeFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.audioDecoderFile))
    }

    func testQwen3TtsModelFilesHaveExtensions() {
        let validExtensions: Set<String> = [".mlmodelc", ".npy"]
        for model in ModelNames.Qwen3TTS.requiredModels {
            let hasValid = validExtensions.contains(where: { model.hasSuffix($0) })
            XCTAssertTrue(hasValid, "Model '\(model)' should have a valid extension")
        }
    }

    func testQwen3TtsContainsEmbeddingFiles() {
        let required = ModelNames.Qwen3TTS.requiredModels
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.speakerEmbeddingFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.cpEmbeddingsFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.ttsBosEmbedFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.ttsPadEmbedFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.ttsEosEmbedFile))
    }

    // MARK: - Repo

    func testQwen3TtsRepoName() {
        XCTAssertEqual(Repo.qwen3Tts.name, "qwen3-tts-coreml")
    }

    func testQwen3TtsRepoRemotePath() {
        XCTAssertTrue(Repo.qwen3Tts.remotePath.contains("qwen3-tts-coreml"))
    }

    func testQwen3TtsRepoFolderName() {
        XCTAssertFalse(Repo.qwen3Tts.folderName.isEmpty)
    }

    // MARK: - Manager

    func testQwen3TtsManagerInitialState() async {
        let manager = Qwen3TtsManager()
        let available = await manager.isAvailable
        XCTAssertFalse(available, "Manager should not be available before loading models")
    }
}
