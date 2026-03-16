import Foundation
import XCTest

@testable import FluidAudio

final class LSEENDIntegrationTests: XCTestCase {
    private struct ErrorStats {
        let maxAbs: Double
        let meanAbs: Double
    }

    private static let repositoryRootURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    private static let fixtureAudioURL = repositoryRootURL.appendingPathComponent("audio.wav")
    nonisolated(unsafe) private static var cachedEngines: [LSEENDVariant: LSEENDInferenceEngine] = [:]

    func testVariantRegistryResolvesAllExportedArtifacts() async throws {
        let expectedColumns: [LSEENDVariant: Int] = [
            .ami: 4,
            .callhome: 7,
            .dihard2: 10,
            .dihard3: 10,
        ]

        for variant in LSEENDVariant.allCases {
            let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
            XCTAssertTrue(FileManager.default.fileExists(atPath: descriptor.modelURL.path))
            XCTAssertTrue(FileManager.default.fileExists(atPath: descriptor.metadataURL.path))

            let engine = try await makeEngine(variant: variant)
            XCTAssertEqual(engine.metadata.realOutputDim, expectedColumns[variant])
            XCTAssertEqual(engine.metadata.fullOutputDim, (expectedColumns[variant] ?? 0) + 2)
            XCTAssertGreaterThan(engine.streamingLatencySeconds, 0)
            XCTAssertGreaterThan(engine.modelFrameHz, 0)
        }
    }

    func testOfflineInferenceProducesConsistentShapesAcrossVariants() async throws {
        for variant in LSEENDVariant.allCases {
            let engine = try await makeEngine(variant: variant)
            let samples = try fixtureAudio(sampleRate: engine.targetSampleRate, limitSeconds: 2.0)
            let result = try engine.infer(samples: samples, sampleRate: engine.targetSampleRate)

            try assertResultInvariants(result, engine: engine, expectedDurationSeconds: duration(of: samples, sampleRate: engine.targetSampleRate))
            assertMatrixClose(result.probabilities, result.logits.applyingSigmoid(), maxAbs: 1e-7, meanAbs: 1e-8)
            assertMatrixClose(result.fullProbabilities, result.fullLogits.applyingSigmoid(), maxAbs: 1e-7, meanAbs: 1e-8)
        }
    }

    func testAudioFileInferenceMatchesInferenceOnResampledFixtureSamples() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let fileResult = try engine.infer(audioFileURL: Self.fixtureAudioURL)
        let resampled = try fixtureAudio(sampleRate: engine.targetSampleRate)
        let sampleResult = try engine.infer(samples: resampled, sampleRate: engine.targetSampleRate)

        assertMatrixClose(fileResult.logits, sampleResult.logits, maxAbs: 1e-6, meanAbs: 1e-7)
        assertMatrixClose(fileResult.probabilities, sampleResult.probabilities, maxAbs: 1e-6, meanAbs: 1e-7)
        assertMatrixClose(fileResult.fullLogits, sampleResult.fullLogits, maxAbs: 1e-6, meanAbs: 1e-7)
        assertMatrixClose(fileResult.fullProbabilities, sampleResult.fullProbabilities, maxAbs: 1e-6, meanAbs: 1e-7)
    }

    func testStreamingSessionMatchesOfflineInferenceOnRealFixtureAudio() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let samples = try fixtureAudio(sampleRate: engine.targetSampleRate, limitSeconds: 4.0)
        let offline = try engine.infer(samples: samples, sampleRate: engine.targetSampleRate)
        let session = try engine.createSession(inputSampleRate: engine.targetSampleRate)

        var totalEmitted = 0
        let chunkSizes = [617, 911, 1283, 743]
        var sawUpdate = false
        var start = 0
        var chunkIndex = 0
        while start < samples.count {
            let chunkSize = chunkSizes[chunkIndex % chunkSizes.count]
            let stop = min(samples.count, start + chunkSize)
            if let update = try session.pushAudio(Array(samples[start..<stop])) {
                sawUpdate = true
                XCTAssertLessThanOrEqual(update.startFrame, update.totalEmittedFrames)
                XCTAssertEqual(update.totalEmittedFrames, update.startFrame + update.probabilities.rows)
                XCTAssertEqual(update.previewStartFrame, update.totalEmittedFrames)
                XCTAssertGreaterThanOrEqual(update.totalEmittedFrames, totalEmitted)
                totalEmitted = update.totalEmittedFrames
            }
            start = stop
            chunkIndex += 1
        }

        let finalUpdate = try session.finalize()
        if let finalUpdate {
            sawUpdate = true
            XCTAssertEqual(finalUpdate.previewLogits.rows, 0)
            XCTAssertEqual(finalUpdate.previewProbabilities.rows, 0)
            XCTAssertLessThanOrEqual(finalUpdate.startFrame, totalEmitted)
            XCTAssertEqual(finalUpdate.totalEmittedFrames, finalUpdate.startFrame + finalUpdate.probabilities.rows)
            XCTAssertEqual(finalUpdate.previewStartFrame, finalUpdate.totalEmittedFrames)
            totalEmitted = finalUpdate.totalEmittedFrames
        }

        XCTAssertTrue(sawUpdate)
        XCTAssertNil(try session.finalize())
        XCTAssertThrowsError(try session.pushAudio(Array(samples.prefix(256))))

        let snapshot = session.snapshot()
        XCTAssertGreaterThan(totalEmitted, 0)
        XCTAssertLessThanOrEqual(totalEmitted, offline.probabilities.rows)
        XCTAssertEqual(snapshot.probabilities.rows, offline.probabilities.rows)
        assertMatrixClose(snapshot.logits, offline.logits, maxAbs: 1e-5, meanAbs: 1e-6)
        assertMatrixClose(snapshot.probabilities, offline.probabilities, maxAbs: 1e-5, meanAbs: 1e-6)
        assertMatrixClose(snapshot.fullLogits, offline.fullLogits, maxAbs: 1e-5, meanAbs: 1e-6)
    }

    func testStreamingSimulationMatchesOfflineInferenceAndReportsMonotonicProgress() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let offline = try engine.infer(audioFileURL: Self.fixtureAudioURL)
        let simulation = try engine.simulateStreaming(audioFileURL: Self.fixtureAudioURL, chunkSeconds: 0.37)

        assertMatrixClose(simulation.result.logits, offline.logits, maxAbs: 1e-5, meanAbs: 1e-6)
        assertMatrixClose(simulation.result.probabilities, offline.probabilities, maxAbs: 1e-5, meanAbs: 1e-6)
        XCTAssertFalse(simulation.updates.isEmpty)

        var previousFrames = 0
        var previousBufferSeconds = 0.0
        let flushIndices = simulation.updates.enumerated().filter { $0.element.flush }.map(\.offset)
        XCTAssertLessThanOrEqual(flushIndices.count, 1)
        if let flushIndex = flushIndices.first {
            XCTAssertEqual(flushIndex, simulation.updates.count - 1)
        }

        for (index, update) in simulation.updates.enumerated() {
            XCTAssertEqual(update.chunkIndex, index + 1)
            XCTAssertGreaterThanOrEqual(update.totalFramesEmitted, previousFrames)
            XCTAssertGreaterThanOrEqual(update.bufferSeconds, previousBufferSeconds)
            previousFrames = update.totalFramesEmitted
            previousBufferSeconds = update.bufferSeconds
        }
    }

    func testDiarizerProcessCompleteMatchesEngineInference() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let samples = try fixtureAudio(sampleRate: engine.targetSampleRate, limitSeconds: 4.0)
        let expected = try engine.infer(samples: samples, sampleRate: engine.targetSampleRate)
        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)

        let timeline = try diarizer.processComplete(samples)

        XCTAssertTrue(diarizer.isAvailable)
        XCTAssertEqual(diarizer.numFramesProcessed, expected.probabilities.rows)
        XCTAssertEqual(diarizer.numSpeakers, engine.metadata.realOutputDim)
        XCTAssertEqual(timeline.numFinalizedFrames, expected.probabilities.rows)
        XCTAssertEqual(timeline.finalizedPredictions.count, expected.probabilities.values.count)
        assertArrayClose(timeline.finalizedPredictions, expected.probabilities.values, maxAbs: 1e-6, meanAbs: 1e-7)
    }

    func testDiarizerStreamingFinalizeMatchesProcessComplete() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let samples = try fixtureAudio(sampleRate: engine.targetSampleRate, limitSeconds: 4.0)
        let expected = try engine.infer(samples: samples, sampleRate: engine.targetSampleRate)

        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)

        for chunk in chunk(samples, sizes: [701, 977, 1153]) {
            let _ = try diarizer.process(samples: chunk)
        }
        let _ = try diarizer.finalizeSession()

        XCTAssertEqual(diarizer.numFramesProcessed, expected.probabilities.rows)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, expected.probabilities.rows)
        assertArrayClose(diarizer.timeline.finalizedPredictions, expected.probabilities.values, maxAbs: 1e-5, meanAbs: 1e-6)

        diarizer.reset()
        XCTAssertEqual(diarizer.numFramesProcessed, 0)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, 0)
    }

    private func makeEngine(variant: LSEENDVariant) async throws -> LSEENDInferenceEngine {
        if let cached = Self.cachedEngines[variant] {
            return cached
        }
        let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
        let engine = try LSEENDInferenceEngine(descriptor: descriptor, computeUnits: .cpuOnly)
        Self.cachedEngines[variant] = engine
        return engine
    }

    private func fixtureAudio(sampleRate: Int, limitSeconds: Double? = nil) throws -> [Float] {
        XCTAssertTrue(FileManager.default.fileExists(atPath: Self.fixtureAudioURL.path))
        let converter = AudioConverter(sampleRate: Double(sampleRate))
        let audio = try converter.resampleAudioFile(Self.fixtureAudioURL)
        guard let limitSeconds else {
            return audio
        }
        let sampleCount = min(audio.count, Int(limitSeconds * Double(sampleRate)))
        return Array(audio.prefix(sampleCount))
    }

    private func duration(of samples: [Float], sampleRate: Int) -> Double {
        Double(samples.count) / Double(sampleRate)
    }

    private func chunk(_ samples: [Float], sizes: [Int]) -> [[Float]] {
        var chunks: [[Float]] = []
        var start = 0
        var index = 0
        while start < samples.count {
            let size = sizes[index % sizes.count]
            let stop = min(samples.count, start + size)
            chunks.append(Array(samples[start..<stop]))
            start = stop
            index += 1
        }
        return chunks
    }

    private func assertResultInvariants(
        _ result: LSEENDInferenceResult,
        engine: LSEENDInferenceEngine,
        expectedDurationSeconds: Double,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        XCTAssertGreaterThan(result.logits.rows, 0, file: file, line: line)
        XCTAssertEqual(result.logits.rows, result.probabilities.rows, file: file, line: line)
        XCTAssertEqual(result.logits.rows, result.fullLogits.rows, file: file, line: line)
        XCTAssertEqual(result.logits.columns, engine.metadata.realOutputDim, file: file, line: line)
        XCTAssertEqual(result.probabilities.columns, engine.metadata.realOutputDim, file: file, line: line)
        XCTAssertEqual(result.fullLogits.columns, engine.metadata.fullOutputDim, file: file, line: line)
        XCTAssertEqual(result.fullProbabilities.columns, engine.metadata.fullOutputDim, file: file, line: line)
        XCTAssertEqual(result.frameHz, engine.modelFrameHz, accuracy: 1e-9, file: file, line: line)
        XCTAssertEqual(result.durationSeconds, expectedDurationSeconds, accuracy: 1e-6, file: file, line: line)
    }

    private func assertMatrixClose(
        _ actual: LSEENDMatrix,
        _ expected: LSEENDMatrix,
        maxAbs: Double,
        meanAbs: Double,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.rows, expected.rows, file: file, line: line)
        XCTAssertEqual(actual.columns, expected.columns, file: file, line: line)
        XCTAssertEqual(actual.values.count, expected.values.count, file: file, line: line)
        let stats = compare(actual.values, expected.values)
        XCTAssertLessThanOrEqual(stats.maxAbs, maxAbs, file: file, line: line)
        XCTAssertLessThanOrEqual(stats.meanAbs, meanAbs, file: file, line: line)
    }

    private func assertArrayClose(
        _ actual: [Float],
        _ expected: [Float],
        maxAbs: Double,
        meanAbs: Double,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        let stats = compare(actual, expected)
        XCTAssertLessThanOrEqual(stats.maxAbs, maxAbs, file: file, line: line)
        XCTAssertLessThanOrEqual(stats.meanAbs, meanAbs, file: file, line: line)
    }

    private func compare(_ actual: [Float], _ expected: [Float]) -> ErrorStats {
        guard actual.count == expected.count else {
            return ErrorStats(maxAbs: .infinity, meanAbs: .infinity)
        }
        var maxAbs = 0.0
        var sumAbs = 0.0
        for (lhs, rhs) in zip(actual, expected) {
            let diff = abs(Double(lhs - rhs))
            maxAbs = max(maxAbs, diff)
            sumAbs += diff
        }
        return ErrorStats(
            maxAbs: maxAbs,
            meanAbs: actual.isEmpty ? 0 : sumAbs / Double(actual.count)
        )
    }
}
