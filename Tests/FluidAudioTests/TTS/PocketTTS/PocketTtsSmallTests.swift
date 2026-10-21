import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Small Pocket TTS models (pocket-tts 4.0.0, `v4/<model>/` packs).
///
/// Pure-logic tests always run. The functional test needs a pack in the
/// local cache (`PocketTtsSmallDownloader.ensurePack` location) and skips
/// otherwise, so CI without network or models stays green.
final class PocketTtsSmallTests: XCTestCase {

    func testModelNamesAndAliases() {
        XCTAssertEqual(PocketTtsSmallModel(name: "english_7m_2026-10"), .english7m)
        XCTAssertEqual(PocketTtsSmallModel(name: "7m"), .english7m)
        XCTAssertEqual(PocketTtsSmallModel(name: "18m-refl"), .english18mRefl)
        XCTAssertEqual(PocketTtsSmallModel(name: "2m_refl"), .english2mRefl)
        XCTAssertNil(PocketTtsSmallModel(name: "100m"))
        XCTAssertEqual(PocketTtsSmallModel.english2m.repoSubdirectory, "v4/english_2m_2026-10")
        XCTAssertEqual(PocketTtsSmallModel.allCases.count, 7)
        XCTAssertEqual(PocketTtsSmallModel(name: "phonon"), .phonon31m)
        XCTAssertEqual(PocketTtsSmallModel(name: "phonon-31m-en"), .phonon31m)
        XCTAssertEqual(PocketTtsSmallModel.phonon31m.repoSubdirectory, "phonon/phonon-31m-en")
        XCTAssertEqual(PocketTtsSmallModel.phonon31m.defaultVoice, "default")
        XCTAssertEqual(PocketTtsSmallModel.english7m.defaultVoice, "alba")
    }

    func testArchAndStateInfoDecode() throws {
        let archJSON = """
            {"model_id":"english_7m_2026-10","d_model":224,"num_layers":8,"num_heads":8,"num_kv_heads":2,
             "head_dim":64,"attn_dim":512,"kv_dim":128,"ff_dim":224,"z_dim":64,"ldim":32,"flow_type":"drifting",
             "rope_max_period":10000.0,"eos_threshold":-1.0,"temperature":0.3,"min_frames_before_eos":6,
             "text_vocab":4001,"text_rank":64,"insert_bos_before_voice":true,"sample_rate":24000,"frame_size":1920,
             "upsample_stride":16,"mimi_decoder_dim":128,"mimi_decoder_layers":12,"mimi_decoder_heads":4,
             "mimi_decoder_head_dim":32,"mimi_decoder_context":50}
            """
        let arch = try JSONDecoder().decode(PocketTtsSmallArch.self, from: Data(archJSON.utf8))
        XCTAssertEqual(arch.numKvHeads, 2)
        XCTAssertEqual(arch.textRank, 64)
        XCTAssertEqual(arch.eosThreshold, -1.0)
        XCTAssertEqual(arch.frameSize, 1920)
        // kyutai packs carry no fade / tail keys: unchanged behaviour
        XCTAssertTrue(arch.fadeInRamp.isEmpty)
        XCTAssertEqual(arch.framesAfterEosExtra, PocketTtsConstants.extraFramesAfterDetection)

        // Phonon: EOS tail +1, 25 ms raised-cosine fade
        var dict = try JSONSerialization.jsonObject(with: Data(archJSON.utf8)) as! [String: Any]
        dict["eos_tail_extra"] = 1
        dict["fade_in"] = true
        dict["fade_in_ms"] = 25.0
        dict["fade_in_shape"] = "cosine"
        let phonon = try JSONDecoder().decode(
            PocketTtsSmallArch.self, from: JSONSerialization.data(withJSONObject: dict))
        XCTAssertEqual(phonon.framesAfterEosExtra, 1)
        let ramp = phonon.fadeInRamp
        XCTAssertEqual(ramp.count, 600)
        XCTAssertEqual(ramp[0], 0, accuracy: 1e-7)
        XCTAssertEqual(ramp[300], 0.5, accuracy: 1e-3)
        XCTAssertLessThan(ramp[24], 0.01)  // first ms is near silent (cosine, not linear)

        let stateJSON = """
            {"functions":["prefill","generate","generate_decode"],"state":{"k_cache0":[1,512,2,64]},
             "state_dtype":"float16","kv_cache_len":512,"cond_prefill_t_max":256,
             "mimi_state":{"st_x_k_cache":[1,80,4,32]},"mimi_conv_io":{"decoder_model_0_previous":[1,128,2]},
             "mimi_phases":10,"mimi_capacity":80,"mimi_tokens_per_frame":16}
            """
        let state = try JSONDecoder().decode(PocketTtsSmallStateInfo.self, from: Data(stateJSON.utf8))
        XCTAssertTrue(state.hasFusedGenerateDecode)
        XCTAssertEqual(state.mimiPhases, 10)
        // warm-up phases 0..4 while the 80-slot cache fills, then cycle 5..9
        XCTAssertEqual((0..<12).map(state.mimiPhase(frame:)), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 6])
    }

    func testDownloaderSkipFilter() {
        let skip = PocketTtsSmallDownloader.shouldSkipAsset(at:)
        XCTAssertFalse(skip("v4/english_7m_2026-10/pocket_state.mlmodelc/model.mil"))
        XCTAssertFalse(skip("v4/english_7m_2026-10/arch.json"))
        XCTAssertFalse(skip("v4/english_7m_2026-10/pocket_state.json"))
        XCTAssertFalse(skip("v4/english_7m_2026-10/constants_bin/alba.safetensors"))
        XCTAssertTrue(skip("v4/english_7m_2026-10/flowlm_step.mlmodelc/model.mil"))
        XCTAssertFalse(skip("v4/english_7m_2026-10/mimi_decoder.mlmodelc/model.mil"))
        XCTAssertTrue(skip("v4/english_7m_2026-10/flow_decoder_fused.mlmodelc/coremldata.bin"))
        XCTAssertTrue(skip("v4/english_7m_2026-10/pocket_state.mlpackage/Manifest.json"))
        XCTAssertTrue(skip("v4/english_7m_2026-10/constants/bos_emb.npy"))
        XCTAssertTrue(skip("v4/english_7m_2026-10/manifest.json"))
    }

    func testReplaceUntrainedCharacters() throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) else { throw XCTSkip("MLState requires macOS 15+") }
        XCTAssertEqual(
            PocketTtsSmallSession.replaceUntrainedCharacters("It\u{2019}s fast \u{2014} really \u{2013} fast"),
            "It's fast , really , fast")
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, *)
    private func assertFp16BufferRoundsAndPads() {
        let src: [Float] = [0, 1, -1, 0.1, 65504, 3.14159265]
        let buf = PocketTtsSmallEngine.fp16Buffer(from: src, totalCount: 16)
        XCTAssertEqual(buf.count, 16)
        #if arch(arm64)
        for (i, v) in src.enumerated() { XCTAssertEqual(buf[i], Float16(v).bitPattern) }
        #endif
        XCTAssertTrue(buf[6...].allSatisfy { $0 == 0 })
    }

    func testFp16Buffer() throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) else { throw XCTSkip("MLState requires macOS 15+") }
        assertFp16BufferRoundsAndPads()
    }

    /// End to end on a cached pack: shipped voice, short sentence, expects EOS
    /// and 1920 samples per frame. Skips when no small-model pack is cached.
    func testMimiPrecisionRawValues() {
        XCTAssertEqual(PocketTtsSmallMimiPrecision.fp32.rawValue, "fp32")
        XCTAssertEqual(PocketTtsSmallMimiPrecision.fp16Fused.rawValue, "fp16-fused")
        XCTAssertEqual(PocketTtsSmallMimiPrecision(rawValue: "fp16-fused"), .fp16Fused)
        XCTAssertEqual(PocketTtsSmallMimiPrecision.allCases.count, 2)
    }

    /// Every state input of the fp32 Mimi I/O model has a same-sized `<name>_out`,
    /// and latent / phase / audio match `arch.json` — what the engine relies on.
    func testMimiDecoderIOContract() throws {
        guard let pack = Self.cachedPack() else { throw XCTSkip("no small Pocket TTS pack in the local cache") }
        let constants = try PocketTtsSmallConstants.load(from: pack)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let mimi = try MLModel(
            contentsOf: pack.appendingPathComponent(ModelNames.PocketTTS.mimiDecoderFile), configuration: config)
        let inputs = mimi.modelDescription.inputDescriptionsByName
        let outputs = mimi.modelDescription.outputDescriptionsByName
        func count(_ d: MLFeatureDescription?) -> Int? { d?.multiArrayConstraint?.shape.map(\.intValue).reduce(1, *) }
        XCTAssertEqual(count(inputs["latent"]), constants.arch.ldim)
        XCTAssertEqual(count(inputs["phase"]), constants.state.mimiPhases)
        XCTAssertEqual(count(outputs["audio"]), constants.arch.frameSize)
        var stateInputs = 0
        for (name, desc) in inputs where name != "latent" && name != "phase" {
            stateInputs += 1
            XCTAssertEqual(count(outputs["\(name)_out"]), count(desc), "state input \(name) needs a same-sized _out")
        }
        XCTAssertGreaterThan(stateInputs, constants.state.mimiConvIO.count)  // conv I/O + Mimi KV caches
    }

    private static func cachedPack() -> URL? {
        guard let base = try? TtsCacheDirectory.ensure() else { return nil }
        let repoDir = base.appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
            .appendingPathComponent(Repo.pocketTts.folderName)
        // Prefer the smallest pack: its first-time ANE compile is ~10 s vs ~150 s for 18m.
        let preferred: [PocketTtsSmallModel] =
            [.english2m, .english7m, .english18m] + PocketTtsSmallModel.allCases.filter { $0 != .phonon31m }
        return preferred.map { repoDir.appendingPathComponent($0.repoSubdirectory) }.first { pack in
            PocketTtsSmallDownloader.requiredFiles(for: .english7m).allSatisfy {
                FileManager.default.fileExists(atPath: pack.appendingPathComponent($0).path)
            }
        }
    }

    func testSynthesizeWithCachedPack() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) else { throw XCTSkip("MLState requires macOS 15+") }
        let base = try TtsCacheDirectory.ensure()
        let repoDir = base.appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
            .appendingPathComponent(Repo.pocketTts.folderName)
        // Prefer the smallest pack: its first-time ANE compile is ~10 s vs ~150 s for 18m.
        let preferred: [PocketTtsSmallModel] =
            [.english2m, .english7m, .english18m] + PocketTtsSmallModel.allCases.filter { $0 != .phonon31m }
        guard
            let model = preferred.first(where: { m in
                PocketTtsSmallDownloader.requiredFiles(for: m).allSatisfy {
                    FileManager.default.fileExists(
                        atPath: repoDir.appendingPathComponent(m.repoSubdirectory).appendingPathComponent($0).path)
                }
            })
        else {
            throw XCTSkip("no small Pocket TTS pack in the local cache")
        }
        for precision in PocketTtsSmallMimiPrecision.allCases {
            let manager = PocketTtsSmallManager(model: model, mimiPrecision: precision)
            try await manager.load(from: repoDir.appendingPathComponent(model.repoSubdirectory))
            let voices = try await manager.availableVoices()
            XCTAssertTrue(voices.contains("alba"), "voices: \(voices)")

            var frames = 0
            var peak: Float = 0
            let stream = try await manager.synthesizeStreaming(
                text: "Hello world, this is a test.", voice: "alba", seed: 42)
            for try await frame in stream {
                XCTAssertEqual(frame.samples.count, 1920)
                frames += 1
                peak = max(peak, frame.samples.map(abs).max() ?? 0)
            }
            // ~2.5 s of speech: between 15 and 80 frames, and audible.
            XCTAssertGreaterThan(frames, 15, "\(precision)")
            XCTAssertLessThan(frames, 80, "\(precision)")
            XCTAssertGreaterThan(peak, 0.05, "\(precision): silent output")
            let wav = try await manager.synthesize(text: "Hello world.", voice: "alba", seed: 1)
            XCTAssertGreaterThan(wav.count, 44 + 2 * 1920 * 10)
        }
    }

    /// Phonon 31m end to end: a KV-snapshot voice and a conditioning-row voice
    /// (prefilled on the fly), fp32 Mimi, chunk fade-in. Uses the cached
    /// `phonon/phonon-31m-en` pack or `PHONON_PACK=<dir>`; skips otherwise.
    func testPhononSynthesizeWithCachedPack() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) else { throw XCTSkip("MLState requires macOS 15+") }
        let pack: URL
        if let env = ProcessInfo.processInfo.environment["PHONON_PACK"] {
            pack = URL(fileURLWithPath: env)
        } else {
            let base = try TtsCacheDirectory.ensure()
            pack = base.appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
                .appendingPathComponent(Repo.pocketTts.folderName)
                .appendingPathComponent(PocketTtsSmallModel.phonon31m.repoSubdirectory)
        }
        guard
            PocketTtsSmallDownloader.requiredFiles(for: .phonon31m).allSatisfy({
                FileManager.default.fileExists(atPath: pack.appendingPathComponent($0).path)
            })
        else { throw XCTSkip("no Phonon pack (set PHONON_PACK)") }
        let manager = PocketTtsSmallManager(model: .phonon31m)
        try await manager.load(from: pack)
        let voices = try await manager.availableVoices()
        XCTAssertTrue(voices.contains("default"))
        XCTAssertGreaterThan(voices.count, 100, "voices/ conditioning rows should be listed")
        let archOrNil = await manager.arch
        let arch = try XCTUnwrap(archOrNil)
        XCTAssertEqual(arch.eosThreshold, 4.0)
        XCTAssertEqual(arch.minFramesBeforeEos, 0)
        for voice in ["default", "en_us_Juniper"] {  // KV snapshot, then conditioning rows
            var frames: [[Float]] = []
            for try await frame in try await manager.synthesizeStreaming(
                text: "Hello world. I am Phonon.", voice: voice, temperature: 0, seed: 0)
            {
                XCTAssertEqual(frame.samples.count, 1920)
                frames.append(frame.samples)
            }
            XCTAssertGreaterThan(frames.count, 15, voice)
            XCTAssertLessThan(frames.count, 80, voice)
            let peak = frames.flatMap { $0 }.map(abs).max() ?? 0
            XCTAssertGreaterThan(peak, 0.05, "\(voice): silent output")
            // the onset click (-40 dBFS raw) is faded: first 2 ms well under -60 dBFS
            let onset = frames[0].prefix(48).map(abs).max() ?? 1
            XCTAssertLessThan(onset, 1e-3, "\(voice): onset \(onset)")
        }
    }
}
