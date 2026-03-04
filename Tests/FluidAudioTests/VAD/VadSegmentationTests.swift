import XCTest

@testable import FluidAudio

final class VadSegmentationTests: XCTestCase {

    func testSilenceProducesNoSegments() async throws {
        let vad = VadManager(skipModelLoading: true)
        let (res, total) = makeVadResults([(false, 2.0)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total)
        XCTAssertTrue(segments.isEmpty)
    }

    func testContinuousSpeechProducesSegment() async throws {
        let vad = VadManager(skipModelLoading: true)
        let duration = 5.0
        let (res, total) = makeVadResults([(true, duration)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total)
        XCTAssertEqual(segments.count, 1)
        let totalDur = segments.reduce(0.0) { $0 + ($1.endTime - $1.startTime) }
        XCTAssertGreaterThan(totalDur, duration * 0.9)
    }

    func testSingleSegmentAmidSilence() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.15)
        let (res, total) = makeVadResults([(false, 1.0), (true, 3.0), (false, 1.0)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertEqual(segments.count, 1)
        let dur = segments[0].endTime - segments[0].startTime
        XCTAssertGreaterThan(dur, 2.7)
        XCTAssertLessThan(dur, 3.4)
    }

    func testMultipleSegmentsSeparatedBySilence() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.15, minSilenceDuration: 0.75)
        let pattern: [(Bool, Double)] = [(false, 1.0), (true, 2.0), (false, 1.0), (true, 2.0), (false, 1.0)]
        let (res, total) = makeVadResults(pattern)
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertEqual(segments.count, 2)
    }

    func testMergingWithinMinSilence() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.15, minSilenceDuration: 0.75)
        let (res, total) = makeVadResults([(true, 1.0), (false, 0.5), (true, 1.0)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertEqual(segments.count, 1)
        // Merged segment includes the short silence between runs due to merge behavior
        let dur = segments[0].endTime - segments[0].startTime
        XCTAssertGreaterThan(dur, 2.4)
        XCTAssertLessThan(dur, 2.6)
    }

    func testNoMergingBeyondMinSilence() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.15, minSilenceDuration: 0.75)
        let (res, total) = makeVadResults([(true, 1.0), (false, 1.0), (true, 1.0)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertEqual(segments.count, 2)
        for seg in segments {
            let dur = seg.endTime - seg.startTime
            XCTAssertGreaterThan(dur, 0.9)
            XCTAssertLessThan(dur, 1.3)
        }
    }

    func testMinSpeechDurationFilter() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.5, minSilenceDuration: 0.75)
        let (res, total) = makeVadResults([(true, 0.2), (false, 1.0), (true, 0.8), (false, 1.0), (true, 0.1)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertEqual(segments.count, 1)
        let dur = segments[0].endTime - segments[0].startTime
        XCTAssertGreaterThan(dur, 0.7)
        XCTAssertLessThan(dur, 1.1)
    }

    func testSplitLongContinuousSpeech() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.15, maxSpeechDuration: 15.0)
        let (res, total) = makeVadResults([(true, 30.0)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertGreaterThanOrEqual(segments.count, 2)
        for seg in segments { XCTAssertLessThan(seg.endTime - seg.startTime, 15.1) }
    }

    func testMaxSpeechDurationEnforcement() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.15, maxSpeechDuration: 10.0)
        let (res, total) = makeVadResults([(true, 25.0)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertGreaterThanOrEqual(segments.count, 3)
        for seg in segments { XCTAssertLessThan(seg.endTime - seg.startTime, 10.1) }
    }

    func testExactlyMaxDurationSegment() async throws {
        let vad = VadManager(skipModelLoading: true)
        let (res, total) = makeVadResults([(true, 5.0)])
        let chunkDuration = Double(VadManager.chunkSize) / Double(VadManager.sampleRate)
        let exactDuration = chunkDuration * Double(res.count)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.15, maxSpeechDuration: exactDuration)
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertEqual(segments.count, 1)
        let duration = segments[0].endTime - segments[0].startTime
        XCTAssertEqual(duration, exactDuration, accuracy: chunkDuration)
    }

    func testAlternatingSpeechSilence() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.1, minSilenceDuration: 0.2)
        var pattern: [(Bool, Double)] = []
        for _ in 0..<5 {
            pattern.append((true, 0.3))
            pattern.append((false, 0.3))
        }
        let (res, total) = makeVadResults(pattern)
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertGreaterThanOrEqual(segments.count, 1)
    }

    func testCustomSegmentationConfig() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(
            minSpeechDuration: 1.0,
            minSilenceDuration: 2.0,
            maxSpeechDuration: 8.0,
            speechPadding: 0.2,
            silenceThresholdForSplit: 0.5
        )
        let (res, total) = makeVadResults([(true, 20.0)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertGreaterThanOrEqual(segments.count, 3)
        for seg in segments { XCTAssertLessThan(seg.endTime - seg.startTime, 8.1) }
    }

    func testRealWorldPattern() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.15, minSilenceDuration: 0.75, maxSpeechDuration: 15.0)
        let (res, total) = makeVadResults([(true, 5.0), (false, 85.0), (true, 30.0)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertGreaterThanOrEqual(segments.count, 3)
        XCTAssertLessThanOrEqual(segments.count, 4)
        for seg in segments { XCTAssertLessThan(seg.endTime - seg.startTime, 15.1) }
    }

    func testEmptyInput() async throws {
        let vad = VadManager(skipModelLoading: true)
        let segments = await vad.segmentSpeech(from: [], totalSamples: 0)
        XCTAssertTrue(segments.isEmpty)
    }

    func testVeryShortSpeechFiltered() async throws {
        let vad = VadManager(skipModelLoading: true)
        let cfg = VadSegmentationConfig(minSpeechDuration: 0.15)
        let (res, total) = makeVadResults([(true, 0.05)])
        let segments = await vad.segmentSpeech(from: res, totalSamples: total, config: cfg)
        XCTAssertTrue(segments.isEmpty)
    }
}
