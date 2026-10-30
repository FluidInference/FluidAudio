import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class Phonon25Tests: XCTestCase {
    private var platformSupportsPhonon25: Bool {
        (try? AsrModels.checkPlatformSupport(for: .phonon25)) != nil
    }

    func testRegistrationAndV3Contract() {
        let version = AsrModelVersion.phonon25
        XCTAssertEqual(version.repo, .phonon25)
        XCTAssertEqual(version.repo.remotePath, "Muesli-HQ/phonon-2.5-coreml")
        XCTAssertEqual(version.repo.revision, "a042ef89966d289df295709b58ffdabb50bd2510")
        XCTAssertEqual(version.repo.folderName, "phonon-2.5")
        XCTAssertNotEqual(version.repo.folderName, Repo.phonon2.folderName)
        XCTAssertEqual(version.blankId, AsrModelVersion.v3.blankId)
        XCTAssertEqual(version.decoderLayers, AsrModelVersion.v3.decoderLayers)
        XCTAssertEqual(version.encoderHiddenSize, AsrModelVersion.v3.encoderHiddenSize)
        XCTAssertTrue(version.isV3Family)
        XCTAssertFalse(version.hasFusedEncoder)
        XCTAssertTrue(ChunkProcessor.supportsSuppressedPrefix(version))
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .phonon25, variant: nil), ModelNames.ASR.requiredModelsV3())
        XCTAssertEqual(AsrModels.defaultCacheDirectory(for: version).lastPathComponent, "phonon-2.5")
    }

    func testLongerPhononCacheNameDoesNotResolveToEnglishPhonon2() {
        XCTAssertEqual(AsrModels.inferredVersion(from: URL(fileURLWithPath: "/models/phonon-2.5")), .phonon25)
        XCTAssertEqual(AsrModels.inferredVersion(from: URL(fileURLWithPath: "/models/phonon-2")), .phonon2)
        // The model folder wins over a parent folder that contains another model's name.
        XCTAssertEqual(
            AsrModels.inferredVersion(from: URL(fileURLWithPath: "/eval/phonon-2.5-runs/phonon-2")), .phonon2)
        XCTAssertEqual(
            AsrModels.inferredVersion(from: URL(fileURLWithPath: "/eval/phonon-2-runs/phonon-2.5")), .phonon25)
    }

    func testPlatformGuard() {
        if #available(macOS 15, iOS 18, tvOS 18, *) {
            XCTAssertNoThrow(try AsrModels.checkPlatformSupport(for: .phonon25))
        } else {
            XCTAssertThrowsError(try AsrModels.checkPlatformSupport(for: .phonon25)) { error in
                XCTAssertTrue(error.localizedDescription.contains("Phonon-2.5"))
            }
        }
    }

    func testEncoderPlacementDoesNotChangeOtherModelsOrComponents() throws {
        let units = try AsrModels.resolvedEncoderComputeUnits(for: .phonon25, configuration: .cpuAndNeuralEngine)
        XCTAssertEqual(units, .cpuAndGPU)
        XCTAssertEqual(AsrModels.defaultConfiguration().computeUnits, .cpuAndNeuralEngine)
        for version: AsrModelVersion in [.v2, .v3, .redux, .ultra, .phonon2, .bark3, .bark3Clean] {
            XCTAssertEqual(
                try AsrModels.resolvedEncoderComputeUnits(for: version, configuration: .cpuAndNeuralEngine),
                .cpuAndNeuralEngine)
            XCTAssertEqual(
                try AsrModels.resolvedEncoderComputeUnits(
                    for: version, configuration: .cpuAndNeuralEngine, requested: .cpuAndGPU), .cpuAndGPU)
        }
        XCTAssertEqual(
            try AsrModels.resolvedEncoderComputeUnits(for: .phonon25, configuration: .all, requested: .cpuOnly),
            .cpuOnly)
        // A CPU-only configuration (background-safe, CI) keeps the Phonon-2.5 encoder on the CPU.
        XCTAssertEqual(try AsrModels.resolvedEncoderComputeUnits(for: .phonon25, configuration: .cpuOnly), .cpuOnly)
        XCTAssertEqual(try AsrModels.resolvedEncoderComputeUnits(for: .phonon25, configuration: .all), .cpuAndGPU)
        for units: MLComputeUnits in [.all, .cpuAndNeuralEngine] {
            XCTAssertThrowsError(
                try AsrModels.resolvedEncoderComputeUnits(for: .phonon25, configuration: .cpuOnly, requested: units))
        }
    }

    func testUnsupportedEncoderRequestFailsBeforeDownloadOrLocalFileAccess() async throws {
        // Every loader checks the platform first; testPlatformGuard covers that path.
        try XCTSkipUnless(platformSupportsPhonon25, "Phonon-2.5 requires iOS 18 / macOS 15 / tvOS 18")
        // Downloads go to `<parent>/phonon-2.5`, so watch a fresh parent. Offline mode turns any download attempt
        // into a different error, so a missing pre-check can't pass on load()'s identical message.
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let target = root.appendingPathComponent(Repo.phonon25.folderName)
        let wasOffline = ModelHub.offlineMode
        ModelHub.offlineMode = true
        defer { ModelHub.offlineMode = wasOffline }
        do {
            _ = try await AsrModels.downloadAndLoad(to: target, version: .phonon25, encoderComputeUnits: .all)
            XCTFail("ANE-capable placement must not download or load Phonon-2.5")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("ANE encoder parity"), error.localizedDescription)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        XCTAssertThrowsError(
            try AsrModels.loadLocal(from: target, version: .phonon25, encoderComputeUnits: .cpuAndNeuralEngine)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("ANE encoder parity"), error.localizedDescription)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }

    /// Uses only real published weights and public recordings, never generated models/audio.
    /// Supply a dedicated managed cache directory named `phonon-2.5`, not another app's active cache:
    /// ModelHub may repair/delete corrupt or revision-mismatched entries there.
    func testPublishedGPUModelWithPublicSpeech() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelDirectory = environment["FLUIDAUDIO_PHONON25_MODELS"],
            let fixtureDirectory = environment["FLUIDAUDIO_PHONON25_FIXTURES"]
        else {
            throw XCTSkip("Set FLUIDAUDIO_PHONON25_MODELS and FLUIDAUDIO_PHONON25_FIXTURES for real-model integration")
        }
        try XCTSkipUnless(platformSupportsPhonon25, "Phonon-2.5 requires iOS 18 / macOS 15 / tvOS 18")
        let directory = URL(fileURLWithPath: modelDirectory)
        // downloadAndLoad(to:) resolves the repo as a sibling named after the folder, so both
        // loaders only read the same files when the directory is already that folder.
        guard directory.lastPathComponent == Repo.phonon25.folderName else {
            XCTFail("FLUIDAUDIO_PHONON25_MODELS must end in \(Repo.phonon25.folderName): \(directory.path)")
            return
        }
        let local = try AsrModels.loadLocal(from: directory, version: .phonon25)
        XCTAssertEqual(local.encoder?.configuration.computeUnits, .cpuAndGPU)
        XCTAssertEqual(local.preprocessor.configuration.computeUnits, .cpuOnly)
        XCTAssertEqual(local.decoder.configuration.computeUnits, .cpuAndNeuralEngine)
        XCTAssertEqual(local.joint.configuration.computeUnits, .cpuAndNeuralEngine)
        XCTAssertEqual(local.vocabulary.count, 8192)
        // Exercise the registered, revision-aware managed loader as well as loadLocal.
        let models = try await AsrModels.downloadAndLoad(to: directory, version: .phonon25)
        XCTAssertEqual(models.encoder?.configuration.computeUnits, .cpuAndGPU)
        let manager = AsrManager(config: .default, models: models)
        let fixtures = [
            ("1089-134686-0000.wav", "stew for dinner"),
            ("fr_fr-10043298898524273336.wav", "terrain montagneux"),
            ("de_de-10004059102127011984.wav", "besten aussichten"),
            ("es_419-1001146817223348054.wav", "a los viajeros"),
        ]
        for (name, expected) in fixtures {
            let url = URL(fileURLWithPath: fixtureDirectory).appendingPathComponent(name)
            let samples = try AudioConverter().resampleAudioFile(url)
            var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
            let result = try await manager.transcribe(samples, decoderState: &state)
            XCTAssertTrue(result.text.lowercased().contains(expected), "\(name): \(result.text)")
        }
    }
}
