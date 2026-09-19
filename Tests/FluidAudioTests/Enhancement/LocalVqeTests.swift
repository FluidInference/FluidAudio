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
/// model is absent or when running in CI. An explicit directory enables
/// these tests in CI and a missing model in that directory is a failure.
final class LocalVqeStreamTests: XCTestCase {

    private static let fixture = "01-validation-request-21.4s"

    override func setUp() async throws {
        if ProcessInfo.processInfo.environment["CI"] != nil,
            ProcessInfo.processInfo.environment["FLUIDAUDIO_LOCALVQE_MODEL_DIR"] == nil
        {
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
            if ProcessInfo.processInfo.environment["FLUIDAUDIO_LOCALVQE_MODEL_DIR"] != nil {
                throw LocalVqeError.modelLoadingFailed("Required test model not available at \(file.path)")
            }
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

    private func waitForOperations(_ count: Int, on stream: LocalVqeStream) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await stream.operationCount < count {
            guard ContinuousClock.now < deadline else {
                throw NSError(
                    domain: "LocalVqeStreamTests", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Never observed \(count) overlapping operations"])
            }
            try await Task.sleep(for: .milliseconds(1))
        }
    }

    private func assertAudioEqual(
        _ actual: [Float], _ expected: [Float], file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        XCTAssertTrue(actual.allSatisfy(\.isFinite), file: file, line: line)
        let maxDiff = zip(actual, expected).reduce(Float.zero) { max($0, abs($1.0 - $1.1)) }
        XCTAssertLessThan(maxDiff, 1e-4, file: file, line: line)
    }

    func testOverlappingPushesAndFlushMatchSequentialAudio() async throws {
        let manager = try loadManager(chunk: .realtime16ms)
        let mic = Array(try loadFixture().prefix(16000 * 5 + 137))
        let split = 16000 * 4
        let firstMic = Array(mic[..<split])
        let secondMic = Array(mic[split...])
        let expected = try await manager.process(mic: mic)
        let stream = try await manager.makeStream()

        let first = Task {
            try await stream.enhance(mic: firstMic, reference: [Float](repeating: 0, count: firstMic.count))
        }
        defer { first.cancel() }
        try await waitForOperations(1, on: stream)
        let second = Task {
            try await stream.enhance(mic: secondMic, reference: [Float](repeating: 0, count: secondMic.count))
        }
        defer { second.cancel() }
        try await waitForOperations(2, on: stream)
        let flush = Task { try await stream.flush() }
        defer { flush.cancel() }

        var actual = try await first.value
        actual.append(contentsOf: try await second.value)
        actual.append(contentsOf: try await flush.value)
        assertAudioEqual(actual, expected)
        let operations = await stream.operationCount
        XCTAssertEqual(operations, 0)
    }

    func testResetBetweenQueuedPushesStartsFreshClip() async throws {
        let manager = try loadManager(chunk: .realtime16ms)
        let mic = Array(try loadFixture().prefix(16000 * 5 + 137))
        let firstMic = Array(mic.prefix(16000 * 4))
        let nextClip = Array(mic.suffix(16000 + 137))
        let expected = try await manager.process(mic: nextClip)
        let stream = try await manager.makeStream()
        let first = Task {
            try await stream.enhance(mic: firstMic, reference: [Float](repeating: 0, count: firstMic.count))
        }
        defer { first.cancel() }
        try await waitForOperations(1, on: stream)
        let reset = Task { try await stream.reset() }
        defer { reset.cancel() }
        try await waitForOperations(2, on: stream)
        let next = Task {
            try await stream.enhance(mic: nextClip, reference: [Float](repeating: 0, count: nextClip.count))
        }
        defer { next.cancel() }

        _ = try await first.value
        try await reset.value
        var actual = try await next.value
        actual.append(contentsOf: try await stream.flush())
        assertAudioEqual(actual, expected)
    }

    func testCancelledQueuedOperationsLeaveCurrentClipUntouched() async throws {
        let manager = try loadManager(chunk: .realtime16ms)
        let mic = Array(try loadFixture().prefix(16000 * 4 + 137))
        let expected = try await manager.process(mic: mic)
        for operation in ["enhance", "flush", "reset"] {
            let stream = try await manager.makeStream()
            let first = Task {
                try await stream.enhance(mic: mic, reference: [Float](repeating: 0, count: mic.count))
            }
            defer { first.cancel() }
            try await waitForOperations(1, on: stream)
            let queued = Task { () throws -> [Float] in
                switch operation {
                case "enhance":
                    return try await stream.enhance(mic: mic, reference: [Float](repeating: 0, count: mic.count))
                case "flush":
                    return try await stream.flush()
                default:
                    try await stream.reset()
                    return []
                }
            }
            defer { queued.cancel() }
            try await waitForOperations(2, on: stream)
            queued.cancel()
            do {
                _ = try await queued.value
                XCTFail("Queued \(operation) should throw CancellationError")
            } catch is CancellationError {
                // Cancellation must remove the queued operation without resetting the active clip.
            }
            var actual = try await first.value
            actual.append(contentsOf: try await stream.flush())
            assertAudioEqual(actual, expected)
        }
    }

    func testCancelledActivePushResetsBeforeReuse() async throws {
        let manager = try loadManager(chunk: .realtime16ms)
        let mic = try loadFixture()
        let nextClip = Array(mic.suffix(16000 + 137))
        let expected = try await manager.process(mic: nextClip)
        let stream = try await manager.makeStream()
        let active = Task {
            try await stream.enhance(mic: mic, reference: [Float](repeating: 0, count: mic.count))
        }
        defer { active.cancel() }
        try await waitForOperations(1, on: stream)
        active.cancel()
        do {
            _ = try await active.value
            XCTFail("Active push should throw CancellationError")
        } catch is CancellationError {
            // A partially processed clip must not contaminate the next one.
        }
        var actual = try await stream.enhance(
            mic: nextClip, reference: [Float](repeating: 0, count: nextClip.count))
        actual.append(contentsOf: try await stream.flush())
        assertAudioEqual(actual, expected)
    }

    func testIndependentStreamsCanRunConcurrently() async throws {
        let manager = try loadManager(chunk: .realtime16ms)
        let mic = Array(try loadFixture().prefix(16000 + 137))
        let otherMic = Array(try loadFixture().suffix(16000 + 73))
        let expected = try await manager.process(mic: mic)
        let otherExpected = try await manager.process(mic: otherMic)

        async let actual = manager.process(mic: mic)
        async let otherActual = manager.process(mic: otherMic)
        let results = try await (actual, otherActual)
        assertAudioEqual(results.0, expected)
        assertAudioEqual(results.1, otherExpected)
    }
}
