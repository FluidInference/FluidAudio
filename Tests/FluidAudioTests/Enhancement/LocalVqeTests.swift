import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Model-free checks of the LocalVQE naming / configuration surface.
final class LocalVqeNamingTests: XCTestCase {

    func testModelFileNames() {
        XCTAssertEqual(
            ModelNames.LocalVQE.modelFile(variant: .v13, chunk: .batch256ms),
            "localvqe-v1.3-4.8M-256ms.mlmodelc")
        XCTAssertEqual(
            ModelNames.LocalVQE.modelFile(variant: .v12, chunk: .realtime16ms),
            "localvqe-v1.2-1.3M-16ms.mlmodelc")
        XCTAssertEqual(ModelNames.LocalVQE.allModels.count, 4)
    }

    func testVariantKeyNarrowsRequiredSet() {
        let key = ModelNames.LocalVQE.variantKey(variant: .v13, chunk: .realtime16ms)
        XCTAssertEqual(key, "v1.3-16ms")
        XCTAssertEqual(
            ModelNames.LocalVQE.requiredModels(variant: key),
            ["localvqe-v1.3-4.8M-16ms.mlmodelc"])
        XCTAssertEqual(ModelNames.getRequiredModelNames(for: .localVqe, variant: key).count, 1)
        XCTAssertEqual(ModelNames.LocalVQE.requiredModels(variant: nil), ModelNames.LocalVQE.allModels)
        XCTAssertEqual(ModelNames.LocalVQE.requiredModels(variant: "bogus"), ModelNames.LocalVQE.allModels)
    }

    func testChunkSampleCounts() {
        XCTAssertEqual(LocalVqeChunk.realtime16ms.samplesPerCall, 256)
        XCTAssertEqual(LocalVqeChunk.batch256ms.samplesPerCall, 4096)
        XCTAssertEqual(LocalVqeManager.outputDelaySamples, LocalVqeManager.hopSize)
        XCTAssertEqual(Repo.localVqe.remotePath, "FluidInference/localvqe-coreml")
        XCTAssertEqual(Repo.localVqe.folderName, "localvqe")
    }
}

/// End-to-end checks against a locally available model bundle.
///
/// Set `FLUIDAUDIO_LOCALVQE_MODEL_DIR` to a directory holding the compiled
/// `localvqe-*.mlmodelc` bundles (e.g. the mobius conversion `build/` dir);
/// otherwise the default model cache is used, and the tests skip when the
/// model is absent or when running in CI.
final class LocalVqeStreamTests: XCTestCase {

    private static let fixture = "01-validation-request-21.4s"

    override func setUp() async throws {
        if ProcessInfo.processInfo.environment["CI"] != nil {
            throw XCTSkip("Skipping LocalVQE model tests in CI")
        }
    }

    private func loadManager(chunk: LocalVqeChunk) throws -> LocalVqeManager {
        let config = LocalVqeConfig(variant: .v13, chunk: chunk, computeUnits: .cpuOnly)
        let dir: URL
        if let override = ProcessInfo.processInfo.environment["FLUIDAUDIO_LOCALVQE_MODEL_DIR"] {
            dir = URL(fileURLWithPath: override)
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            dir = appSupport.appendingPathComponent("FluidAudio/Models/\(Repo.localVqe.folderName)")
        }
        let file = dir.appendingPathComponent(ModelNames.LocalVQE.modelFile(variant: .v13, chunk: chunk))
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw XCTSkip("LocalVQE model not available at \(file.path)")
        }
        return try LocalVqeManager(config: config, modelDirectory: dir)
    }

    private func loadFixture() throws -> [Float] {
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(Self.fixture)", withExtension: "wav")
                ?? Bundle.module.url(forResource: Self.fixture, withExtension: "wav")
        else {
            throw XCTSkip("fixture \(Self.fixture).wav not bundled")
        }
        return try AudioConverter().resampleAudioFile(url)
    }

    func testWholeClipOutputIsSameLengthAndBounded() async throws {
        let manager = try loadManager(chunk: .batch256ms)
        let mic = try loadFixture()
        // Silent far end: the model runs as a noise suppressor / dereverberator.
        let out = try await manager.process(mic: mic)
        XCTAssertEqual(out.count, mic.count)
        XCTAssertFalse(out.contains { !$0.isFinite })
        let inRms = (mic.reduce(0) { $0 + $1 * $1 } / Float(mic.count)).squareRoot()
        let outRms = (out.reduce(0) { $0 + $1 * $1 } / Float(out.count)).squareRoot()
        XCTAssertGreaterThan(outRms, inRms * 0.1, "enhancer removed almost all speech")
        XCTAssertLessThan(outRms, inRms * 4, "enhancer output level far above input")
    }

    func testStreamingMatchesWholeClipAcrossBufferSizes() async throws {
        let manager = try loadManager(chunk: .realtime16ms)
        let mic = Array(try loadFixture().prefix(16000 * 4))
        let reference = [Float](repeating: 0, count: mic.count)
        let whole = try await manager.process(mic: mic, reference: reference)
        XCTAssertEqual(whole.count, mic.count)

        for bufferSize in [100, 256, 1000, 4096] {
            let stream = try await manager.makeStream()
            var out: [Float] = []
            var offset = 0
            while offset < mic.count {
                let end = min(offset + bufferSize, mic.count)
                out.append(
                    contentsOf: try await stream.enhance(
                        mic: Array(mic[offset..<end]), reference: Array(reference[offset..<end])))
                offset = end
            }
            out.append(contentsOf: try await stream.flush())
            XCTAssertEqual(out.count, mic.count, "buffer \(bufferSize)")
            var maxDiff: Float = 0
            for i in 0..<min(out.count, whole.count) {
                maxDiff = max(maxDiff, abs(out[i] - whole[i]))
            }
            XCTAssertLessThan(maxDiff, 1e-4, "buffer \(bufferSize): streaming diverged from whole-clip")
        }
    }

    func testChunkSizesProduceIdenticalAudio() async throws {
        let small = try loadManager(chunk: .realtime16ms)
        let large = try loadManager(chunk: .batch256ms)
        let mic = Array(try loadFixture().prefix(16000 * 3))
        let a = try await small.process(mic: mic)
        let b = try await large.process(mic: mic)
        XCTAssertEqual(a.count, b.count)
        var maxDiff: Float = 0
        for i in 0..<a.count { maxDiff = max(maxDiff, abs(a[i] - b[i])) }
        XCTAssertLessThan(maxDiff, 1e-4)
    }

    func testLengthMismatchThrows() async throws {
        let manager = try loadManager(chunk: .realtime16ms)
        let stream = try await manager.makeStream()
        var caught: Error?
        do {
            _ = try await stream.enhance(mic: [Float](repeating: 0, count: 10), reference: [])
        } catch {
            caught = error
        }
        guard let vqeError = caught as? LocalVqeError, case .lengthMismatch(let m, let r) = vqeError else {
            XCTFail("expected lengthMismatch, got \(String(describing: caught))")
            return
        }
        XCTAssertEqual(m, 10)
        XCTAssertEqual(r, 0)
    }

    func testFlushOnEmptyStreamReturnsNothing() async throws {
        let manager = try loadManager(chunk: .realtime16ms)
        let stream = try await manager.makeStream()
        let out = try await stream.flush()
        XCTAssertTrue(out.isEmpty)
        let stateCount = await stream.stateCount
        XCTAssertEqual(stateCount, 33)
    }
}
