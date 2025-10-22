import XCTest

@testable import FluidAudio

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

    func testStreamingRespectsThresholdOverride() async {
        let vad = VadManager(skipModelLoading: true, config: VadConfig(defaultThreshold: 0.8))
        let state = VadStreamState.initial()
        let overrideConfig = VadSegmentationConfig(negativeThreshold: 0.2, negativeThresholdOffset: 0.05)

        // Below derived threshold (0.25) should not trigger speech.
        let belowResult = await vad.streamingStateMachine(
            probability: 0.24,
            chunkSampleCount: VadManager.chunkSize,
            modelState: state.modelState,
            state: state,
            config: overrideConfig,
            returnSeconds: false,
            timeResolution: 1
        )
        XCTAssertNil(belowResult.event)

        // Crossing the derived threshold should trigger a speech start.
        let triggerResult = await vad.streamingStateMachine(
            probability: 0.3,
            chunkSampleCount: VadManager.chunkSize,
            modelState: belowResult.state.modelState,
            state: belowResult.state,
            config: overrideConfig,
            returnSeconds: false,
            timeResolution: 1
        )
        XCTAssertEqual(triggerResult.event?.kind, .speechStart)
        let speechPadSamples = Int(overrideConfig.speechPadding * Double(VadManager.sampleRate))
        let expectedStart = max(0, VadManager.chunkSize - speechPadSamples)
        XCTAssertEqual(triggerResult.event?.sampleIndex, expectedStart)
    }

    func testStreamingUsesDefaultThresholdWithoutOverride() async {
        let vad = VadManager(skipModelLoading: true, config: VadConfig(defaultThreshold: 0.6))
        let state = VadStreamState.initial()
        let defaultConfig = VadSegmentationConfig()

        // below default threshold should not trigger
        let belowResult = await vad.streamingStateMachine(
            probability: 0.59,
            chunkSampleCount: VadManager.chunkSize,
            modelState: state.modelState,
            state: state,
            config: defaultConfig,
            returnSeconds: false,
            timeResolution: 1
        )
        XCTAssertNil(belowResult.event)

        // Above default threshold should trigger
        let triggerResult = await vad.streamingStateMachine(
            probability: 0.7,
            chunkSampleCount: VadManager.chunkSize,
            modelState: belowResult.state.modelState,
            state: belowResult.state,
            config: defaultConfig,
            returnSeconds: false,
            timeResolution: 1
        )
        XCTAssertEqual(triggerResult.event?.kind, .speechStart)
        let speechPadSamples = Int(defaultConfig.speechPadding * Double(VadManager.sampleRate))
        let expectedStart = max(0, VadManager.chunkSize - speechPadSamples)
        XCTAssertEqual(triggerResult.event?.sampleIndex, expectedStart)
    }
}
