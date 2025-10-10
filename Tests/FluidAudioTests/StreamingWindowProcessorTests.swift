import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class StreamingWindowProcessorTests: XCTestCase {
    // Helper to express configs in raw sample counts for easier reasoning.
    private func makeConfig(chunk: Int, left: Int, right: Int) -> StreamingAsrConfig {
        StreamingAsrConfig(
            chunkSeconds: Double(chunk) / 16000.0,
            leftContextSeconds: Double(left) / 16000.0,
            rightContextSeconds: Double(right) / 16000.0,
            stabilizer: StreamingStabilizerConfig(),
            vad: .disabled
        )
    }

    func testAppendWithoutSamplesReturnsNoWindows() {
        let config = makeConfig(chunk: 4, left: 2, right: 1)
        var processor = StreamingWindowProcessor(config: config)

        let windows = processor.append([], allowPartialChunk: false)

        XCTAssertTrue(windows.isEmpty)
        XCTAssertFalse(processor.hasBufferedAudio())
    }

    func testPartialChunkWithSingleSampleProducesWindow() throws {
        let config = makeConfig(chunk: 4, left: 2, right: 1)
        var processor = StreamingWindowProcessor(config: config)

        let windows = processor.append([0.75], allowPartialChunk: true, minimumCenterSamples: 1)

        XCTAssertEqual(windows.count, 1)
        let window = try XCTUnwrap(windows.first)
        XCTAssertEqual(window.startSample, 0)
        XCTAssertEqual(window.samples, [0.75])
    }

    func testPendingLeadingSilenceAddsLeftPadding() throws {
        let config = makeConfig(chunk: 4, left: 2, right: 0)
        var processor = StreamingWindowProcessor(config: config)
        processor.advanceBySilence(3)

        let windows = processor.append([0.1, 0.2], allowPartialChunk: true, minimumCenterSamples: 2)

        XCTAssertEqual(windows.count, 1)
        let window = try XCTUnwrap(windows.first)
        XCTAssertEqual(window.startSample, 0)
        XCTAssertEqual(window.samples, [0.0, 0.0, 0.0, 0.1])
    }

    func testTrailingSilenceProvidesRightPadding() throws {
        let config = makeConfig(chunk: 4, left: 2, right: 2)
        var processor = StreamingWindowProcessor(config: config)

        XCTAssertTrue(processor.append([1, 2, 3, 4], allowPartialChunk: false).isEmpty)

        processor.advanceBySilence(2)
        let windows = processor.append([], allowPartialChunk: false)

        XCTAssertEqual(windows.count, 1)
        let window = try XCTUnwrap(windows.first)
        XCTAssertEqual(window.startSample, 0)
        XCTAssertEqual(window.samples, [1, 2, 3, 4, 0, 0])
    }

    func testExactBoundaryProducesConsistentStartSamples() {
        let config = makeConfig(chunk: 4, left: 2, right: 0)
        var processor = StreamingWindowProcessor(config: config)

        let samples: [Float] = (1...8).map { Float($0) }
        let windows = processor.append(samples, allowPartialChunk: false)

        XCTAssertEqual(windows.count, 2)
        XCTAssertEqual(windows[0].startSample, 0)
        XCTAssertEqual(windows[0].samples, [1, 2, 3, 4])
        XCTAssertEqual(windows[1].startSample, 2)
        XCTAssertEqual(windows[1].samples, [3, 4, 5, 6, 7, 8])
    }
}
