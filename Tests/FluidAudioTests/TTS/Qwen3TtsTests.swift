import Foundation
import XCTest

@testable import FluidAudio

final class Qwen3TtsConstantsTests: XCTestCase {

    // MARK: - Constants Validation

    func testAudioSampleRate() {
        XCTAssertEqual(Qwen3TtsConstants.audioSampleRate, 24_000)
    }

    func testSamplesPerFrame() {
        XCTAssertEqual(Qwen3TtsConstants.samplesPerFrame, 1_920)
    }

    func testMaxCodecTokens() {
        XCTAssertGreaterThan(Qwen3TtsConstants.maxCodecTokens, 0)
        XCTAssertEqual(Qwen3TtsConstants.maxCodecTokens, 125)
    }

    func testCodecEosId() {
        XCTAssertEqual(Qwen3TtsConstants.codecEosId, 2150)
    }

    func testCodecSpecialTokens() {
        XCTAssertEqual(Qwen3TtsConstants.codecPadId, 2148)
        XCTAssertEqual(Qwen3TtsConstants.codecBosId, 2149)
        XCTAssertEqual(Qwen3TtsConstants.codecEosId, 2150)
        XCTAssertEqual(Qwen3TtsConstants.codecThinkId, 2154)
        XCTAssertEqual(Qwen3TtsConstants.codecNoThinkId, 2155)
        XCTAssertEqual(Qwen3TtsConstants.codecThinkBosId, 2156)
        XCTAssertEqual(Qwen3TtsConstants.codecThinkEosId, 2157)
    }

    func testLanguageIds() {
        XCTAssertEqual(Qwen3TtsConstants.languageIds["english"], 2050)
        XCTAssertEqual(Qwen3TtsConstants.languageIds["chinese"], 2055)
        XCTAssertEqual(Qwen3TtsConstants.languageIds.count, 10)
    }

    func testKvCacheDimensions() {
        // CodeDecoder
        XCTAssertEqual(Qwen3TtsConstants.cdKvLen, 256)
        XCTAssertEqual(Qwen3TtsConstants.cdKvDim, 28_672)
        // MultiCodeDecoder
        XCTAssertEqual(Qwen3TtsConstants.mcdKvLen, 16)
        XCTAssertEqual(Qwen3TtsConstants.mcdKvDim, 5_120)
    }

    func testModelDimensions() {
        XCTAssertEqual(Qwen3TtsConstants.hiddenSize, 1024)
        XCTAssertEqual(Qwen3TtsConstants.numCodebooks, 16)
        XCTAssertEqual(Qwen3TtsConstants.codecVocabSize, 2048)
    }

    func testSamplingParameters() {
        XCTAssertEqual(Qwen3TtsConstants.temperature, 0.9)
        XCTAssertEqual(Qwen3TtsConstants.topK, 50)
        XCTAssertGreaterThan(Qwen3TtsConstants.repetitionPenalty, 1.0)
    }

    func testMinNewTokensIsReasonable() {
        XCTAssertGreaterThanOrEqual(Qwen3TtsConstants.minNewTokens, 0)
        XCTAssertLessThan(Qwen3TtsConstants.minNewTokens, Qwen3TtsConstants.maxCodecTokens)
    }

    func testSpeechDecoderFrames() {
        XCTAssertEqual(Qwen3TtsConstants.speechDecoderFrames, 125)
    }

    // MARK: - Model Names

    func testQwen3TtsRequiredModelsNonEmpty() {
        XCTAssertFalse(ModelNames.Qwen3TTS.requiredModels.isEmpty)
    }

    func testQwen3TtsRequiredModelsContainCoreModels() {
        let required = ModelNames.Qwen3TTS.requiredModels
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.textProjectorFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.codeEmbedderFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.multiCodeEmbedderFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.codeDecoderFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.multiCodeDecoderFile))
        XCTAssertTrue(required.contains(ModelNames.Qwen3TTS.speechDecoderFile))
    }

    func testQwen3TtsRequiredModelsCount() {
        XCTAssertEqual(ModelNames.Qwen3TTS.requiredModels.count, 6)
    }

    func testQwen3TtsModelFilesHaveExtensions() {
        for model in ModelNames.Qwen3TTS.requiredModels {
            XCTAssertTrue(model.hasSuffix(".mlmodelc"), "Model '\(model)' should have .mlmodelc extension")
        }
    }

    func testQwen3TtsSpeakerEmbeddingNotInRequired() {
        let required = ModelNames.Qwen3TTS.requiredModels
        XCTAssertFalse(required.contains(ModelNames.Qwen3TTS.speakerEmbeddingFile))
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
