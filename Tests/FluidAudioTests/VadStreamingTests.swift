import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class VadStreamingTests: XCTestCase {

    func testStreamingEmitsStartAndEndEvents() async {
        let vad = VadManager(skipModelLoading: true)
        var state = VadStreamState.initial()
        let config = VadSegmentationConfig()

        // First chunk triggers speech start
        let startResult = await vad.streamingStateMachine(
            probability: 0.9,
            chunkSampleCount: VadManager.chunkSize,
            modelState: state.modelState,
            state: state,
            config: config,
            returnSeconds: false,
            timeResolution: 1
        )
        XCTAssertEqual(startResult.event?.kind, .speechStart)
        XCTAssertEqual(startResult.event?.sampleIndex, 0)
        state = startResult.state
        XCTAssertTrue(state.triggered)

        // Feed silence chunks until the minimum silence duration elapses
        var capturedEnd: VadStreamEvent?
        for _ in 0..<5 {
            let silentResult = await vad.streamingStateMachine(
                probability: 0.05,
                chunkSampleCount: VadManager.chunkSize,
                modelState: state.modelState,
                state: state,
                config: config,
                returnSeconds: false,
                timeResolution: 1
            )
            state = silentResult.state
            if let event = silentResult.event {
                capturedEnd = event
                break
            }
        }

        XCTAssertNotNil(capturedEnd)
        XCTAssertEqual(capturedEnd?.kind, .speechEnd)
        XCTAssertFalse(state.triggered)
        XCTAssertGreaterThan(capturedEnd?.sampleIndex ?? -1, 0)
    }

    func testStreamingReturnsSecondsWhenRequested() async {
        let vad = VadManager(skipModelLoading: true)
        var state = VadStreamState.initial()
        let config = VadSegmentationConfig()

        // Trigger start event
        state =
            (await vad.streamingStateMachine(
                probability: 0.9,
                chunkSampleCount: VadManager.chunkSize,
                modelState: state.modelState,
                state: state,
                config: config,
                returnSeconds: true,
                timeResolution: 2
            )).state

        var endEvent: VadStreamEvent?
        for _ in 0..<5 {
            let result = await vad.streamingStateMachine(
                probability: 0.05,
                chunkSampleCount: VadManager.chunkSize,
                modelState: state.modelState,
                state: state,
                config: config,
                returnSeconds: true,
                timeResolution: 2
            )
            state = result.state
            if let event = result.event {
                endEvent = event
                break
            }
        }

        XCTAssertNotNil(endEvent)
        if let event = endEvent {
            let expectedSeconds = Double(event.sampleIndex) / Double(VadManager.sampleRate)
            XCTAssertEqual(event.time, (expectedSeconds * 100).rounded() / 100)
        }
    }
}
