#if arch(arm64)
import Foundation
import XCTest

@testable import FluidAudio

final class LFM3AudioTests: XCTestCase {

    // MARK: - Registry

    func testRequiredFilesSplitByBackendAndSpeech() {
        let names = ModelNames.LFM3Audio.self
        XCTAssertEqual(ModelNames.getRequiredModelNames(for: .lfm3Audio, variant: nil), names.requiredModels)
        XCTAssertTrue(names.recognitionModels.isSuperset(of: names.languageModels))
        XCTAssertTrue(names.recognitionModels.isSuperset(of: names.sharedRecognitionModels))
        XCTAssertTrue(names.languageModels.isDisjoint(with: names.sharedRecognitionModels))
        XCTAssertTrue(names.speechModels.isDisjoint(with: names.recognitionModels))
        XCTAssertTrue(names.sharedRecognitionModels.contains("coreml/AudioEncoder-fp16.mlmodelc"))
        XCTAssertEqual(Repo.lfm3Audio.remotePath, "FluidInference/lfm3-audio-2b-coreml")
        XCTAssertEqual(Repo.lfm3AudioCoreAI.remotePath, "FluidInference/lfm3-audio-2b-coreai")
        XCTAssertNotEqual(Repo.lfm3Audio.folderName, Repo.lfm3AudioCoreAI.folderName)
    }

    func testDownloadSetsPerBackend() throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip("LFM3-Audio needs macOS 15 / iOS 18") }
        let names = ModelNames.LFM3Audio.self
        func files(_ backend: LFM3AudioBackend, _ speech: Bool) -> Set<String> {
            ModelNames.getRequiredModelNames(
                for: .lfm3Audio, variant: LFM3AudioModels.downloadVariant(backend: backend, speech: speech))
        }
        XCTAssertEqual(files(.coreML, true), names.requiredModels)
        XCTAssertEqual(files(.coreML, false), names.recognitionModels)
        // The Core AI backend never downloads the Core ML LM or the Core ML depth decoder.
        XCTAssertTrue(files(.coreAI, true).isDisjoint(with: names.languageModels))
        XCTAssertEqual(files(.coreAI, true), names.sharedRecognitionModels.union(names.speechCodecModels))
        XCTAssertFalse(files(.coreAI, true).contains(names.compiledModelFile(names.depthDecoder)))
        XCTAssertEqual(files(.coreAI, false), names.sharedRecognitionModels)
        // Core AI repo: LM only for recognition, LM + fused depth decoder with speech.
        let lm = names.coreAIBundleFiles(names.coreAILanguageModel)
        XCTAssertEqual(ModelNames.getRequiredModelNames(for: .lfm3AudioCoreAI, variant: "recognition"), lm)
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .lfm3AudioCoreAI, variant: nil),
            lm.union(names.coreAIBundleFiles(names.coreAIDepthDecoder)))
        XCTAssertTrue(lm.allSatisfy { $0.hasPrefix("coreai/LFM3-Audio-u8b32-f16L20-23.aimodel/") })
    }

    // MARK: - Prompts

    func testPromptsMatchTheChatTemplate() {
        let asr = LFM3AudioConfig.asrPrompt(language: "French")
        XCTAssertEqual(
            asr.prefix, "<|startoftext|><|im_start|>system\nPerform ASR in French.<|im_end|>\n<|im_start|>user\n")
        XCTAssertEqual(asr.suffix, "<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertTrue(LFM3AudioConfig.asrPrompt(language: nil).prefix.contains("system\nPerform ASR.<|im_end|>"))
        let tts = LFM3AudioConfig.ttsPrompt(text: "Hi.", language: "British English", voice: "male")
        XCTAssertTrue(tts.contains("Perform TTS in British English using a male voice.<|im_end|>"))
        XCTAssertTrue(tts.hasSuffix("<|im_start|>user\nHi.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>"))
    }

    // MARK: - Backbone host logic

    func testPrefillPlanPadsTheFirstChunkOnAFreshState() {
        // 300 rows -> 44 rows padded into a q64 call (20 pads), then one q256 call
        let plan = LFM3AudioPrefill.plan(n: 300, buckets: [256, 64, 16, 13], padFirst: true)
        XCTAssertEqual(
            plan,
            [
                LFM3AudioChunk(slotStart: 0, q: 64, nPad: 20),
                LFM3AudioChunk(slotStart: 64, q: 256, nPad: 0),
            ])
        XCTAssertEqual(plan.map { $0.q - $0.nPad }.reduce(0, +), 300)
    }

    func testPrefillPlanLaddersAfterTheStart() {
        let plan = LFM3AudioPrefill.plan(n: 33, buckets: [256, 64, 16, 13], padFirst: true, startSlot: 100)
        XCTAssertEqual(
            plan,
            [
                LFM3AudioChunk(slotStart: 100, q: 16, nPad: 0),
                LFM3AudioChunk(slotStart: 116, q: 16, nPad: 0),
                LFM3AudioChunk(slotStart: 132, q: 1, nPad: 0),
            ])
        XCTAssertEqual(LFM3AudioPrefill.plan(n: 1, buckets: [256, 13], padFirst: false, startSlot: 7).first?.q, 1)
        XCTAssertTrue(LFM3AudioPrefill.plan(n: 0, buckets: [256], padFirst: true).isEmpty)
    }

    func testLiveChunksUseADeferredPrefillBucket() {
        // A LiveASR chunk feeds `<|text_end|>` + 12 encoder rows mid-context: one q13 call, loaded on first use.
        let rows = 1 + LFM3AudioConfig.liveEncoderTokens
        let plan = LFM3AudioPrefill.plan(n: rows, buckets: [256, 64, 16, 13], padFirst: true, startSlot: 200)
        XCTAssertEqual(plan, [LFM3AudioChunk(slotStart: 200, q: 13, nPad: 0)])
        XCTAssertTrue(LFM3AudioConfig.deferredPrefillRows.isSuperset(of: plan.map(\.q)))
        // Decode and the large prefill buckets load with the pack.
        XCTAssertTrue(LFM3AudioConfig.deferredPrefillRows.isDisjoint(with: [1, 64, 256]))
    }

    func testLowDiskLoadFailureKeepsTheModelCache() {
        let lowDisk = LFM3AudioError.lowDiskSpace(freeBytes: 12_000_000_000, underlying: "load failed")
        XCTAssertTrue(lowDisk.preservesModelCache)
        XCTAssertTrue(lowDisk.localizedDescription.contains("12 GB"), lowDisk.localizedDescription)
        XCTAssertFalse(LFM3AudioError.invalidModel("embedding shape").preservesModelCache)
        XCTAssertFalse(LFM3AudioError.modelNotFound("coreml/AudioEncoder-fp16.mlmodelc").preservesModelCache)
        XCTAssertNotNil(LFM3AudioConfig.freeDiskBytes())
    }

    func testFullMaskIsCausalAndIsolatesPads() {
        let span = 8
        var mask = [Float16](repeating: 1, count: 3 * span)
        mask.withUnsafeMutableBufferPointer {
            LFM3AudioMasks.full($0.baseAddress!, slotStart: 2, q: 3, span: span, nPad: 2)
        }
        for r in 0..<3 {
            let query = 2 + r
            for key in 0..<span {
                let allowed = key >= 2 && key <= query
                XCTAssertEqual(mask[r * span + key], allowed ? 0 : LFM3AudioMasks.negative, "row \(r) key \(key)")
            }
        }
    }

    func testSlidingMaskKeepsTheWindow() {
        let fifo = 4
        let q = 2
        var mask = [Float16](repeating: 1, count: q * (fifo + q))
        mask.withUnsafeMutableBufferPointer {
            LFM3AudioMasks.sliding($0.baseAddress!, slotStart: 10, q: q, window: 3, fifo: fifo, nPad: 0)
        }
        // row 0 = slot 10: fifo holds slots 6...9; window 3 -> keys 8, 9, 10
        let row0 = Array(mask[0..<(fifo + q)]).map { $0 == 0 }
        XCTAssertEqual(row0, [false, false, true, true, true, false])
        let row1 = Array(mask[(fifo + q)...]).map { $0 == 0 }
        XCTAssertEqual(row1, [false, false, false, true, true, true])
    }

    func testRopeZeroTailPairsDoNotRotate() {
        let rope = LFM3AudioRope(theta: 1_000_000, pairs: 2, headDim: 8)
        XCTAssertEqual(rope.invFreq.count, 4)
        XCTAssertEqual(rope.invFreq[0], 1)
        XCTAssertEqual(rope.invFreq[2], 0)
        XCTAssertEqual(rope.invFreq[3], 0)
    }

    // MARK: - Audio front end

    func testMelAndEncoderTokenCounts() {
        // 1 s at 16 kHz -> padded to 16080 samples -> 99 mel frames (n_fft 400, hop 160) -> 13 encoder tokens
        XCTAssertEqual(LFM3AudioLogMel.frameCount(samples: 16_000), 99)
        XCTAssertEqual(LFM3AudioLogMel.tokenCount(frames: 99), 13)
        XCTAssertEqual(LFM3AudioLogMel.frameCount(samples: 100), 1)  // shorter than one window: padded to 400
    }

    func testEncoderPlannerEmitsCompleteWindowsThenTheFlush() {
        var planner = LFM3AudioEncoderPlanner(tokensPerCall: 12)
        let chunk = LFM3AudioMelBuffer(data: [Float](repeating: 0, count: 128 * 96), frames: 96)
        let first = planner.push(chunk)
        XCTAssertEqual(first.map(\.outputRows), [12])
        XCTAssertEqual(first.first?.mel.count, 128 * (8 * 12 + 8))
        let tail = LFM3AudioMelBuffer(data: [Float](repeating: 0, count: 128 * 50), frames: 50)
        let last = planner.push(tail, isFinal: true)
        XCTAssertEqual(last.map(\.outputRows), [7])  // (50 + 7) / 8
        XCTAssertEqual(planner.tokens, LFM3AudioLogMel.tokenCount(frames: 146))
        XCTAssertTrue(planner.push(chunk).isEmpty)  // finished
    }

    func testFifoMaskWindow() {
        let mask = LFM3AudioEncoderPlanner.fifoMask(previous: 0, q: 2, window: 32, fifo: 31)
        XCTAssertEqual(mask.count, 2 * 33)
        XCTAssertEqual(mask[31], 0)  // row 0 sees itself
        XCTAssertEqual(mask[30], -30000)  // empty fifo slot
        XCTAssertEqual(mask[33 + 31], 0)
        XCTAssertEqual(mask[33 + 32], 0)  // row 1 sees row 0 and itself
    }

    // MARK: - Real models (opt-in)

    /// Real published weights + real recordings (the SlidingWindow fixtures). Set `FLUIDAUDIO_LFM3_AUDIO_MODELS` to
    /// a local `lfm3-audio-2b-coreml` directory (or `FLUIDAUDIO_LFM3_AUDIO_DOWNLOAD=1` to download ~3.3 GB).
    @available(macOS 15, iOS 18, *)
    private func loadManager(speech: Bool) async throws -> LFM3AudioManager {
        let environment = ProcessInfo.processInfo.environment
        if let directory = environment["FLUIDAUDIO_LFM3_AUDIO_MODELS"] {
            let models = try LFM3AudioModels.load(from: URL(fileURLWithPath: directory), includeSpeech: speech)
            return try await LFM3AudioManager(models: models)
        }
        try XCTSkipUnless(
            environment["FLUIDAUDIO_LFM3_AUDIO_DOWNLOAD"] == "1",
            "Set FLUIDAUDIO_LFM3_AUDIO_MODELS or FLUIDAUDIO_LFM3_AUDIO_DOWNLOAD=1 for the real-model tests")
        return try await LFM3AudioManager.downloadAndLoad(includeSpeech: speech)
    }

    private func fixture(_ name: String) throws -> URL {
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil)
        else {
            throw XCTSkip("fixture \(name) missing from test bundle")
        }
        return url
    }

    func testOfflineAndLiveTranscriptionOfARealRecording() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip("LFM3-Audio needs macOS 15 / iOS 18") }
        let manager = try await loadManager(speech: false)
        let samples = try AudioConverter().resampleAudioFile(try fixture("01-validation-request-21.4s.wav"))
        let offline = try await manager.transcribe(samples)
        XCTAssertTrue(offline.text.lowercased().contains("help them out"), offline.text)
        // Concurrent calls share one LM context; the manager serializes them.
        async let first = manager.transcribe(samples)
        async let second = manager.transcribe(samples)
        let (a, b) = try await (first, second)
        XCTAssertEqual(a.text, offline.text)
        XCTAssertEqual(b.text, offline.text)

        try await manager.startLiveTranscription()
        for start in stride(from: 0, to: samples.count, by: 2560) {
            try await manager.appendLiveAudio(Array(samples[start..<min(start + 2560, samples.count)]))
        }
        let live = try await manager.finishLiveTranscription()
        XCTAssertTrue(live.lowercased().contains("help them out"), live)
        await XCTAssertThrowsErrorAsync { try await manager.finishLiveTranscription() }
    }

    func testSynthesizedSpeechTranscribesBack() async throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip("LFM3-Audio needs macOS 15 / iOS 18") }
        let manager = try await loadManager(speech: true)
        let speech = try await manager.synthesize("The quick brown fox jumps over the lazy dog.", seed: 0)
        XCTAssertTrue(speech.finished)
        XCTAssertEqual(speech.sampleRate, 24_000)
        XCTAssertGreaterThan(speech.duration, 1)
        let samples = try AudioConverter().resample(speech.samples, from: Double(speech.sampleRate))
        let text = try await manager.transcribe(samples).text.lowercased()
        XCTAssertTrue(text.contains("brown fox") && text.contains("lazy dog"), text)
    }
}
#endif
