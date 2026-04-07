import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@MainActor
final class LSEENDIntegrationTests: XCTestCase {
    private struct ErrorStats {
        let maxAbs: Double
        let meanAbs: Double
    }

    private static var cachedEngines: [LSEENDVariant: LSEENDInferenceHelper] = [:]

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
            let samples = try DiarizationTestFixtures.fixtureAudio(
                sampleRate: engine.targetSampleRate, limitSeconds: 2.0)
            let result = try engine.infer(samples: samples, sampleRate: engine.targetSampleRate)

            try assertResultInvariants(
                result, engine: engine,
                expectedDurationSeconds: duration(of: samples, sampleRate: engine.targetSampleRate))
            assertMatrixClose(result.probabilities, result.logits.applyingSigmoid(), maxAbs: 1e-7, meanAbs: 1e-8)
            assertMatrixClose(
                result.fullProbabilities, result.fullLogits.applyingSigmoid(), maxAbs: 1e-7, meanAbs: 1e-8)
        }
    }

    func testAudioFileInferenceMatchesInferenceOnResampledFixtureSamples() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let fileResult = try engine.infer(audioFileURL: try DiarizationTestFixtures.fixtureAudioFileURL())
        let resampled = try DiarizationTestFixtures.fixtureAudio(sampleRate: engine.targetSampleRate)
        let sampleResult = try engine.infer(samples: resampled, sampleRate: engine.targetSampleRate)

        assertMatrixClose(fileResult.logits, sampleResult.logits, maxAbs: 1e-6, meanAbs: 1e-7)
        assertMatrixClose(fileResult.probabilities, sampleResult.probabilities, maxAbs: 1e-6, meanAbs: 1e-7)
        assertMatrixClose(fileResult.fullLogits, sampleResult.fullLogits, maxAbs: 1e-6, meanAbs: 1e-7)
        assertMatrixClose(fileResult.fullProbabilities, sampleResult.fullProbabilities, maxAbs: 1e-6, meanAbs: 1e-7)
    }

    func testStreamingSessionMatchesOfflineInferenceOnRealFixtureAudio() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: engine.targetSampleRate, limitSeconds: 4.0)
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

    func testStreamingFinalizationUsesExactPaddingForTailFlush() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let sampleRate = engine.targetSampleRate
        let stableBlockSize = engine.metadata.resolvedHopLength * engine.metadata.resolvedSubsampling
        let rawSamples = try DiarizationTestFixtures.fixtureAudio(sampleRate: sampleRate, limitSeconds: 6.0)
        let sampleCount = stableBlockSize * 2 + stableBlockSize / 2 + 37
        let samples = Array(rawSamples.prefix(sampleCount))
        let expected = try engine.infer(samples: samples, sampleRate: sampleRate)

        let targetEndFrame = Int(round(Double(samples.count) / Double(sampleRate) * engine.modelFrameHz))
        let expectedPaddingSamples = max(0, targetEndFrame * stableBlockSize - samples.count)

        XCTAssertGreaterThan(expectedPaddingSamples, 0)
        XCTAssertEqual((samples.count + expectedPaddingSamples) % stableBlockSize, 0)

        let session = try engine.createSession(inputSampleRate: sampleRate)
        _ = try session.pushAudio(samples)
        let finalUpdate = try session.finalize()

        XCTAssertNotNil(finalUpdate)
        XCTAssertEqual(finalUpdate?.previewLogits.rows, 0)
        XCTAssertEqual(finalUpdate?.previewProbabilities.rows, 0)
        XCTAssertNil(try session.finalize())

        let snapshot = session.snapshot()
        XCTAssertEqual(snapshot.probabilities.rows, expected.probabilities.rows)
        XCTAssertEqual(snapshot.logits.rows, expected.logits.rows)
        assertMatrixClose(snapshot.logits, expected.logits, maxAbs: 1e-5, meanAbs: 1e-6)
        assertMatrixClose(snapshot.probabilities, expected.probabilities, maxAbs: 1e-5, meanAbs: 1e-6)
        assertMatrixClose(snapshot.fullLogits, expected.fullLogits, maxAbs: 1e-5, meanAbs: 1e-6)
        assertMatrixClose(snapshot.fullProbabilities, expected.fullProbabilities, maxAbs: 1e-5, meanAbs: 1e-6)
    }

    func testEmptyAudioFinalizationProducesNoOutput() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let session = try engine.createSession(inputSampleRate: engine.targetSampleRate)

        XCTAssertNil(try session.finalize())
        XCTAssertNil(try session.finalize())

        let snapshot = session.snapshot()
        XCTAssertEqual(snapshot.logits.rows, 0)
        XCTAssertEqual(snapshot.probabilities.rows, 0)
        XCTAssertEqual(snapshot.fullLogits.rows, 0)
        XCTAssertEqual(snapshot.fullProbabilities.rows, 0)
        XCTAssertEqual(snapshot.durationSeconds, 0)
    }

    func testStreamingSimulationMatchesOfflineInferenceAndReportsMonotonicProgress() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let fixtureURL = try DiarizationTestFixtures.fixtureAudioFileURL()
        let offline = try engine.infer(audioFileURL: fixtureURL)
        let simulation = try engine.simulateStreaming(audioFileURL: fixtureURL, chunkSeconds: 0.37)

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
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: engine.targetSampleRate, limitSeconds: 4.0)
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
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: engine.targetSampleRate, limitSeconds: 4.0)
        let expected = try engine.infer(samples: samples, sampleRate: engine.targetSampleRate)

        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)

        for chunk in DiarizationTestFixtures.chunk(samples, sizes: [701, 977, 1153]) {
            let _ = try diarizer.process(samples: chunk)
        }
        let finalChunk = try diarizer.finalizeSession()

        XCTAssertEqual(diarizer.numFramesProcessed, expected.probabilities.rows)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, expected.probabilities.rows)
        XCTAssertEqual(finalChunk?.tentativeFrameCount, 0)
        XCTAssertEqual(finalChunk?.tentativePredictions.count, 0)
        XCTAssertEqual(diarizer.timeline.numTentativeFrames, 0)
        assertArrayClose(
            diarizer.timeline.finalizedPredictions, expected.probabilities.values, maxAbs: 1e-5, meanAbs: 1e-6)

        diarizer.reset()
        XCTAssertEqual(diarizer.numFramesProcessed, 0)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, 0)
    }

    func testEnrollSpeakerThrowsWhenNotInitialized() {
        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        let samples = [Float](repeating: 0.1, count: 16000)

        XCTAssertThrowsError(try diarizer.enrollSpeaker(withSamples: samples)) { error in
            guard case LSEENDError.modelPredictionFailed(let message) = error else {
                XCTFail("Expected modelPredictionFailed but got \(error)")
                return
            }
            XCTAssertTrue(message.contains("not initialized"))
        }
    }

    func testEnrollSpeakerResetsVisibleTimelineAndAllowsStreaming() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: engine.targetSampleRate, limitSeconds: 6.0)
        let enrollmentCount = min(samples.count / 2, engine.targetSampleRate * 2)
        let enrollment = Array(samples.prefix(enrollmentCount))
        let live = Array(samples.dropFirst(enrollmentCount))

        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)

        let speaker = try diarizer.enrollSpeaker(withSamples: enrollment, named: "Alice")

        if let speaker {
            XCTAssertEqual(speaker.name, "Alice")
        }
        XCTAssertEqual(diarizer.numFramesProcessed, 0)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, 0)

        var firstUpdate: DiarizerTimelineUpdate?
        for chunk in DiarizationTestFixtures.chunk(live, sizes: [977, 1231, 1607]) {
            if let update = try diarizer.process(samples: chunk) {
                firstUpdate = update
                break
            }
        }

        let finalChunk = try diarizer.finalizeSession()

        XCTAssertTrue(firstUpdate != nil || finalChunk != nil)
        if let firstUpdate {
            XCTAssertEqual(firstUpdate.chunkResult.startFrame, 0)
        }
        XCTAssertGreaterThan(diarizer.timeline.numFinalizedFrames, 0)
        XCTAssertEqual(diarizer.numFramesProcessed, diarizer.timeline.numFinalizedFrames)
    }

    func testProcessCompleteKeepsPrimedSessionOnlyWhenRequested() async throws {
        let engine = try await makeEngine(variant: .dihard3)
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: engine.targetSampleRate, limitSeconds: 6.0)
        let enrollmentSampleCount = engine.targetSampleRate * 2
        let enrollment = Array(samples.prefix(enrollmentSampleCount))
        let complete = Array(samples.dropFirst(enrollmentSampleCount).prefix(enrollmentSampleCount))

        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)

        _ = try diarizer.enrollSpeaker(withSamples: enrollment, named: "Alice")
        XCTAssertTrue(diarizer.hasActiveSession)

        _ = try diarizer.processComplete(complete, keepingEnrolledSpeakers: true)
        XCTAssertFalse(diarizer.hasActiveSession)

        _ = try diarizer.processComplete(complete, keepingEnrolledSpeakers: false)
        XCTAssertFalse(diarizer.hasActiveSession)
    }

    func testDiarizerTimelineCountAndDurationPropertiesReflectFrames() throws {
        let frameDurationSeconds: Float = 0.25
        let timeline = DiarizerTimeline(config: .default(numSpeakers: 3, frameDurationSeconds: frameDurationSeconds))

        XCTAssertEqual(timeline.numFinalizedFrames, 0)
        XCTAssertEqual(timeline.numTentativeFrames, 0)
        XCTAssertEqual(timeline.numFrames, 0)
        XCTAssertEqual(timeline.finalizedDuration, 0)
        XCTAssertEqual(timeline.tentativeDuration, 0)
        XCTAssertEqual(timeline.duration, 0)

        let chunk = DiarizerChunkResult(
            startFrame: 5,
            finalizedPredictions: [
                0.10, 0.20, 0.30,
                0.40, 0.50, 0.60,
            ],
            finalizedFrameCount: 2,
            tentativePredictions: [
                0.70, 0.80, 0.90,
            ],
            tentativeFrameCount: 1
        )

        try timeline.addChunk(chunk)

        XCTAssertEqual(timeline.numFinalizedFrames, 2)
        XCTAssertEqual(timeline.numTentativeFrames, 1)
        XCTAssertEqual(timeline.numFrames, 3)
        XCTAssertEqual(timeline.finalizedDuration, 0.5, accuracy: 1e-6)
        XCTAssertEqual(timeline.tentativeDuration, 0.25, accuracy: 1e-6)
        XCTAssertEqual(timeline.duration, 0.75, accuracy: 1e-6)

        timeline.finalize()

        XCTAssertEqual(timeline.numFinalizedFrames, 3)
        XCTAssertEqual(timeline.numTentativeFrames, 0)
        XCTAssertEqual(timeline.numFrames, 3)
        XCTAssertEqual(timeline.finalizedDuration, 0.75, accuracy: 1e-6)
        XCTAssertEqual(timeline.tentativeDuration, 0, accuracy: 1e-6)
        XCTAssertEqual(timeline.duration, 0.75, accuracy: 1e-6)
    }

    func testLogitsActivityTypeReportsLogitScaleSegmentActivity() throws {
        let frameDuration: Float = 0.1
        let config = DiarizerTimelineConfig(
            numSpeakers: 1,
            frameDurationSeconds: frameDuration,
            onsetThreshold: 0.5,
            offsetThreshold: 0.5,
            activityType: .logits
        )
        let timeline = DiarizerTimeline(config: config)

        // Three frames all above onset — single contiguous segment with known probabilities.
        // logit(0.6) = log(0.6/0.4), logit(0.8) = log(0.8/0.2), logit(0.9) = log(0.9/0.1)
        let probs: [Float] = [0.6, 0.8, 0.9]
        let chunk = DiarizerChunkResult(
            startFrame: 0,
            finalizedPredictions: probs,
            finalizedFrameCount: probs.count,
            tentativePredictions: [],
            tentativeFrameCount: 0
        )
        try timeline.addChunk(chunk)
        timeline.finalize()

        let expectedActivity: Float =
            (log(0.6 / (1 - 0.6)) + log(0.8 / (1 - 0.8)) + log(0.9 / (1 - 0.9))) / 3

        XCTAssertEqual(timeline.numFinalizedFrames, 3)
        let segments = timeline.speakers.values.flatMap { $0.finalizedSegments }
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].activity, expectedActivity, accuracy: 1e-6)
    }

    private func makeEngine(variant: LSEENDVariant) async throws -> LSEENDInferenceHelper {
        if let cached = Self.cachedEngines[variant] {
            return cached
        }
        let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
        let engine = try LSEENDInferenceHelper(descriptor: descriptor, computeUnits: .cpuOnly)
        Self.cachedEngines[variant] = engine
        return engine
    }

    private func duration(of samples: [Float], sampleRate: Int) -> Double {
        Double(samples.count) / Double(sampleRate)
    }

    private func assertResultInvariants(
        _ result: LSEENDInferenceResult,
        engine: LSEENDInferenceHelper,
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
