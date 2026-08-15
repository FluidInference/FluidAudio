import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Tests for the Trial 23 `.aneState` MLState pipeline.
///
/// The pure-logic tests (fp16 conversion, snapshot layout) run everywhere on
/// macOS 15+/iOS 18+. The functional tests additionally require the
/// `pocket_state.mlmodelc` multifunction artifact in the local cache (it is
/// not published on HuggingFace yet) and skip gracefully when absent.
final class PocketTtsStateEngineTests: XCTestCase {

    // MARK: - fp16 conversion (pure logic)

    @available(macOS 15.0, iOS 18.0, *)
    func assertFp16BufferMatchesSwiftRounding() {
        // Values chosen to exercise rounding, subnormals, and specials.
        let source: [Float] = [
            0, -0, 1, -1, 0.1, -0.1, 65504, 1e-8, 3.14159265, 2.0001, -2.0001,
            1024.05, 0.000061035156, 1.5e-5,
        ]
        let buffer = PocketTtsStateEngine.fp16Buffer(
            from: source, offset: 0, count: source.count)
        XCTAssertEqual(buffer.count, PocketTtsConstants.kvCacheMaxLen * 16 * 64)
        #if arch(arm64)
        for (i, value) in source.enumerated() {
            XCTAssertEqual(
                buffer[i], Float16(value).bitPattern,
                "fp16 rounding mismatch at \(i) for \(value)")
        }
        #endif
        // Padding beyond the source stays zero.
        XCTAssertTrue(buffer[source.count...].allSatisfy { $0 == 0 })
    }

    func testFp16BufferMatchesSwiftRounding() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else {
            throw XCTSkip("MLState requires macOS 15+/iOS 18+")
        }
        assertFp16BufferMatchesSwiftRounding()
    }

    func testFp16SnapshotSplitsKVAndCarriesPosition() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else {
            throw XCTSkip("MLState requires macOS 15+/iOS 18+")
        }
        // One layer, tiny seq: K block = 1..N, V block = offset by 1e6
        // (flat row-major [2, 1, seq, 16, 64], K then V — same layout the
        // IO `kvCacheStateFromSnapshot` consumes).
        let seqLen = 3
        let perKV = seqLen * 16 * 64
        var flat = [Float]()
        flat.append(contentsOf: (0..<perKV).map { Float($0) })
        flat.append(contentsOf: (0..<perKV).map { Float($0) + 1_000_000 })
        let snapshot = PocketTtsVoiceCacheSnapshot(
            layers: [.init(cache: flat, offset: seqLen)],
            cacheSeqLen: seqLen
        )

        let fp16 = try PocketTtsStateEngine.fp16Snapshot(from: snapshot, layers: 1)
        XCTAssertEqual(fp16.position, Float(seqLen))
        XCTAssertEqual(fp16.kBuffers.count, 1)
        XCTAssertEqual(fp16.vBuffers.count, 1)

        let expectedK = PocketTtsStateEngine.fp16Buffer(from: flat, offset: 0, count: perKV)
        let expectedV = PocketTtsStateEngine.fp16Buffer(from: flat, offset: perKV, count: perKV)
        XCTAssertEqual(fp16.kBuffers[0], expectedK)
        XCTAssertEqual(fp16.vBuffers[0], expectedV)
        // Beyond the valid prefix the buffers are zero (reset contract).
        XCTAssertTrue(fp16.kBuffers[0][perKV...].allSatisfy { $0 == 0 })
        XCTAssertTrue(fp16.vBuffers[0][perKV...].allSatisfy { $0 == 0 })

        XCTAssertThrowsError(
            try PocketTtsStateEngine.fp16Snapshot(from: snapshot, layers: 2))
    }

    func testIsContiguous() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else {
            throw XCTSkip("MLState requires macOS 15+/iOS 18+")
        }
        XCTAssertTrue(
            PocketTtsStateEngine.isContiguous(
                shape: [1, 512, 16, 64], strides: [524_288, 1024, 64, 1]))
        XCTAssertFalse(
            PocketTtsStateEngine.isContiguous(
                shape: [1, 512, 16, 64], strides: [524_288, 1056, 64, 1]))
        XCTAssertFalse(
            PocketTtsStateEngine.isContiguous(shape: [1, 512], strides: [1]))
    }

    // MARK: - Functional (artifact-gated)

    /// Path of the locally installed multifunction package, or nil.
    private func installedStateModelURL() -> URL? {
        guard let cacheRoot = try? TtsCacheDirectory.ensure() else { return nil }
        let url =
            cacheRoot
            .appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
            .appendingPathComponent(Repo.pocketTts.folderName)
            .appendingPathComponent(PocketTtsLanguage.english.repoSubdirectory)
            .appendingPathComponent(ModelNames.PocketTTS.pocketStateFile)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    @available(macOS 15.0, iOS 18.0, *)
    private func loadEngine(from url: URL) throws -> PocketTtsStateEngine {
        func config(_ functionName: String) -> MLModelConfiguration {
            let c = MLModelConfiguration()
            c.computeUnits = .cpuAndNeuralEngine
            c.functionName = functionName
            return c
        }
        let prefill = try MLModel(
            contentsOf: url,
            configuration: config(ModelNames.PocketTTS.StateFunction.prefill))
        let generate = try MLModel(
            contentsOf: url,
            configuration: config(ModelNames.PocketTTS.StateFunction.generate))
        return PocketTtsStateEngine(
            models: PocketTtsStateModels(prefill: prefill, generate: generate),
            layers: 6)
    }

    /// Host-side `MLState.withMultiArray` write → read round-trip must be
    /// bit-exact (the property Trial 23 proved from python; this validates
    /// the Swift plumbing — strides, fp16 handling, buffer naming).
    func testStateWriteReadRoundTripBitExact() async throws {
        guard #available(macOS 15.0, iOS 18.0, *) else {
            throw XCTSkip("MLState requires macOS 15+/iOS 18+")
        }
        guard let url = installedStateModelURL() else {
            throw XCTSkip("pocket_state.mlmodelc not installed in the local cache")
        }
        let engine = try loadEngine(from: url)

        // Deterministic pseudo-random fp32 KV values for a 7-position voice.
        let seqLen = 7
        let perKV = seqLen * 16 * 64
        var rng = SeededRNG(seed: 7)
        var layersData: [PocketTtsVoiceCacheSnapshot.LayerCache] = []
        for _ in 0..<6 {
            let cache = (0..<(2 * perKV)).map { _ in
                Float.gaussianRandom(using: &rng)
            }
            layersData.append(.init(cache: cache, offset: seqLen))
        }
        let snapshot = PocketTtsVoiceCacheSnapshot(layers: layersData, cacheSeqLen: seqLen)
        let fp16 = try PocketTtsStateEngine.fp16Snapshot(from: snapshot, layers: 6)

        try await engine.reset(with: fp16)
        let readBack = try await engine.captureSnapshot()
        XCTAssertEqual(readBack.position, Float(seqLen))
        for i in 0..<6 {
            XCTAssertEqual(readBack.kBuffers[i], fp16.kBuffers[i], "k_cache\(i) round-trip")
            XCTAssertEqual(readBack.vBuffers[i], fp16.vBuffers[i], "v_cache\(i) round-trip")
        }
    }

    /// Full prefill → generate over the shared state: positions advance and
    /// the fused step yields finite latent/EOS outputs.
    func testPrefillAndGenerateAdvanceSharedState() async throws {
        guard #available(macOS 15.0, iOS 18.0, *) else {
            throw XCTSkip("MLState requires macOS 15+/iOS 18+")
        }
        guard let url = installedStateModelURL() else {
            throw XCTSkip("pocket_state.mlmodelc not installed in the local cache")
        }
        let engine = try loadEngine(from: url)
        try await engine.resetToZero()
        let positionAfterReset = await engine.position
        XCTAssertEqual(positionAfterReset, 0)

        // 4-token zero conditioning block (content doesn't matter for the
        // position contract).
        let dim = PocketTtsConstants.embeddingDim
        let flat = [Float](repeating: 0, count: 4 * dim)
        try await engine.prefill(flatConditioning: flat, tokenCount: 4)
        let positionAfterPrefill = await engine.position
        XCTAssertEqual(positionAfterPrefill, 4)

        var rng = SeededRNG(seed: 42)
        let noise = (0..<PocketTtsConstants.latentDim).map { _ in
            Float.gaussianRandom(using: &rng) * sqrtf(PocketTtsConstants.temperature)
        }
        let sequence = [Float](repeating: 0, count: PocketTtsConstants.latentDim)
        let (latent, eosLogit) = try await engine.generateFrame(
            sequence: sequence, noise: noise)
        XCTAssertEqual(latent.count, PocketTtsConstants.latentDim)
        XCTAssertTrue(latent.allSatisfy(\.isFinite), "latent must be finite")
        XCTAssertTrue(eosLogit.isFinite, "eos logit must be finite")
        let positionAfterGenerate = await engine.position
        XCTAssertEqual(positionAfterGenerate, 5)
    }
}
