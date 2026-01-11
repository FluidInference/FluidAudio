import XCTest

@testable import FluidAudio

final class SortformerTests: XCTestCase {

    // MARK: - Feature Provider Tests

    func testFeatureProviderEquivalency() throws {
        let config = SortformerConfig.default

        // Create 5 seconds of deterministic random audio
        let sampleRate = 16000
        let audioCount = sampleRate * 5
        srand48(Int(Date().timeIntervalSince1970 * 1e6))
        let audio = (0..<audioCount).map { _ in Float(drand48() - 0.5) }

        // 1. Get chunks from Batch Feature Provider
        var batchChunks: [[Float]] = []
        var featureProvider = SortformerFeatureLoader(config: config, audio: audio)

        while let (chunk, _, _, _) = featureProvider.next() {
            batchChunks.append(chunk)
        }

        // 2. Get chunks from Streaming Diarizer
        let diarizer = SortformerDiarizer(config: config)
        var streamingChunks: [[Float]] = []

        diarizer.addAudio(audio)

        while let (mel, _) = diarizer.getNextChunkFeatures() {
            streamingChunks.append(mel)
        }

        // 3. Compare
        XCTAssertEqual(batchChunks.count, streamingChunks.count, "Chunk count mismatch")

        for i in 0..<min(batchChunks.count, streamingChunks.count) {
            let batch = batchChunks[i]
            let stream = streamingChunks[i]

            XCTAssertEqual(batch.count, stream.count, "Chunk \(i) size mismatch")

            for j in 0..<batch.count {
                XCTAssertEqual(batch[j], stream[j], accuracy: 1e-5, "Chunk \(i) mismatch at \(j)")
            }
        }
    }

    // MARK: - Timeline Tests

    func testTimelineEquivalency() throws {
        let config = SortformerPostProcessingConfig.default
        let numSpeakers = config.numSpeakers

        // Create random meaningful probabilities for 20 frames
        // Simulate a scenario: 5 frames silence, 5 frames Spk0, 5 frames Spk1, 5 frames silence
        var predictions: [Float] = []
        for i in 0..<20 {
            if i >= 5 && i < 10 {
                // Speaker 0 active
                predictions.append(contentsOf: [0.9, 0.1, 0.1, 0.1])
            } else if i >= 10 && i < 15 {
                // Speaker 1 active
                predictions.append(contentsOf: [0.1, 0.9, 0.1, 0.1])
            } else {
                // Silence
                predictions.append(contentsOf: [0.1, 0.1, 0.1, 0.1])
            }
        }

        // 1. Batch Timeline (Finalized immediately)
        let batchTimeline = SortformerTimeline(
            allPredictions: predictions,
            config: config,
            isComplete: true
        )

        // 2. Streaming Timeline (Add chunk by chunk)
        let streamingTimeline = SortformerTimeline(config: config)

        // Feed in chunks of 5 frames
        let chunkSize = 5
        for i in stride(from: 0, to: 20, by: chunkSize) {
            let chunkPreds = Array(predictions[i * numSpeakers..<(i + chunkSize) * numSpeakers])

            let chunk = SortformerChunkResult(
                startFrame: i,
                speakerPredictions: chunkPreds,
                frameCount: chunkSize,
                tentativePredictions: [],  // No tentative for this basic test
                tentativeFrameCount: 0
            )

            streamingTimeline.addChunk(chunk)
        }

        // Finalize streaming timeline
        streamingTimeline.finalize()

        // 3. Compare Results
        XCTAssertEqual(batchTimeline.numFrames, streamingTimeline.numFrames, "Total frames mismatch")
        XCTAssertEqual(batchTimeline.segments.count, streamingTimeline.segments.count, "Segment count mismatch")

        // Compare segments
        for (batchSpk, streamSpk) in zip(batchTimeline.segments, streamingTimeline.segments) {
            for (batchSeg, streamSeg) in zip(batchSpk, streamSpk) {
                XCTAssertEqual(batchSeg.speakerIndex, streamSeg.speakerIndex, "Segment speaker mismatch")
                XCTAssertEqual(batchSeg.startFrame, streamSeg.startFrame, "Segment start mismatch")
                XCTAssertEqual(batchSeg.endFrame, streamSeg.endFrame, "Segment end mismatch")
            }
        }

        // Compare frame predictions
        for i in 0..<batchTimeline.framePredictions.count {
            XCTAssertEqual(
                batchTimeline.framePredictions[i],
                streamingTimeline.framePredictions[i],
                accuracy: 1e-5,
                "Prediction mismatch at index \(i)"
            )
        }
    }

    // Note: Incremental audio feeding (adding audio in small chunks during real-time streaming)
    // produces different mel features than batch mode due to NeMo's center padding being applied
    // at each audio boundary. This is a known architectural limitation.
    // For exact batch-matching results, feed all audio at once via addAudio() before extracting chunks.

    func testBufferBounds() throws {
        var config = SortformerPostProcessingConfig.default
        let numSpeakers = config.numSpeakers
        config.maxStoredFrames = 50

        // Create timeline with maxFrames limit
        let timeline = SortformerTimeline(
            config: config,
        )

        // Feed 200 frames of predictions (way more than maxFrames)
        let totalFrames = 200
        for frameOffset in stride(from: 0, to: totalFrames, by: 10) {
            var chunkPreds: [Float] = []
            for _ in 0..<10 {
                chunkPreds.append(contentsOf: [Float](repeating: 0.5, count: numSpeakers))
            }

            let chunk = SortformerChunkResult(
                startFrame: frameOffset,
                speakerPredictions: chunkPreds,
                frameCount: 10,
                tentativePredictions: [],
                tentativeFrameCount: 0
            )
            timeline.addChunk(chunk)
        }

        // Verify framePredictions is bounded to maxFrames
        XCTAssertLessThanOrEqual(
            timeline.framePredictions.count, config.maxStoredFrames! * config.numSpeakers,
            "framePredictions should be bounded to maxFrames")

        // Verify we still have some predictions (not all trimmed)
        XCTAssertGreaterThan(timeline.numFrames, 0, "Should have some predictions")

    }

    func testSegmentExtraction() throws {
        let config = SortformerConfig.default
        let numSpeakers = config.numSpeakers

        // Create predictions with clear speaker pattern:
        // Frames 0-9: Speaker 0 active
        // Frames 10-19: Speaker 1 active
        // Frames 20-29: Both speakers active (overlap)
        // Frames 30-39: Silence
        var predictions: [Float] = []

        for i in 0..<40 {
            if i < 10 {
                predictions.append(contentsOf: [0.9, 0.1, 0.1, 0.1])
            } else if i < 20 {
                predictions.append(contentsOf: [0.1, 0.9, 0.1, 0.1])
            } else if i < 30 {
                predictions.append(contentsOf: [0.9, 0.9, 0.1, 0.1])  // Overlap
            } else {
                predictions.append(contentsOf: [0.1, 0.1, 0.1, 0.1])  // Silence
            }
        }

        let timeline = SortformerTimeline(
            allPredictions: predictions,
            config: .default,
            isComplete: true
        )

        // Check that we have segments
        XCTAssertGreaterThan(timeline.segments.count, 0, "Should have extracted segments")

        // Verify segment speakers are valid
        for speaker in timeline.segments {
            for segment in speaker {
                XCTAssertGreaterThanOrEqual(segment.speakerIndex, 0)
                XCTAssertLessThan(segment.speakerIndex, numSpeakers)
                XCTAssertLessThanOrEqual(segment.startFrame, segment.endFrame)
            }
        }
    }

    func testReset() throws {
        let config = SortformerConfig.default
        let numSpeakers = config.numSpeakers

        let timeline = SortformerTimeline(config: .default)

        // Add some data
        let chunk = SortformerChunkResult(
            startFrame: 0,
            speakerPredictions: [Float](repeating: 0.5, count: 10 * numSpeakers),
            frameCount: 10,
            tentativePredictions: [],
            tentativeFrameCount: 0
        )
        timeline.addChunk(chunk)
        timeline.finalize()

        XCTAssertGreaterThan(timeline.numFrames, 0, "Should have frames before reset")

        // Reset
        timeline.reset()

        // Verify state is cleared
        XCTAssertEqual(timeline.numFrames, 0, "numFrames should be 0 after reset")
        XCTAssertEqual(timeline.framePredictions.count, 0, "framePredictions should be empty")
        XCTAssertEqual(timeline.segments.reduce(0) { $0 + $1.count }, 0, "segments should be empty")
        XCTAssertEqual(timeline.tentativePredictions.count, 0, "tentativePredictions should be empty")
    }
}
