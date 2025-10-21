import Foundation

extension VadManager {

    /// Construct a fresh streaming state that mirrors Silero's `reset_states` call.
    public func makeStreamState() -> VadStreamState {
        VadStreamState.initial()
    }

    /// Process a streaming chunk and emit start/end events using Silero-style hysteresis.
    public func processStreamingChunk(
        _ audioChunk: [Float],
        state: VadStreamState,
        config: VadSegmentationConfig = .default,
        returnSeconds: Bool = false,
        timeResolution: Int = 1
    ) async throws -> VadStreamResult {
        let result = try await processChunk(audioChunk, inputState: state.modelState)
        return streamingStateMachine(
            probability: result.probability,
            chunkSampleCount: audioChunk.count,
            modelState: result.outputState,
            state: state,
            config: config,
            returnSeconds: returnSeconds,
            timeResolution: timeResolution
        )
    }

    /// Internal helper that exposes the state machine for unit testing with synthetic probabilities.
    internal func streamingStateMachine(
        probability: Float,
        chunkSampleCount: Int,
        modelState: VadState,
        state: VadStreamState,
        config: VadSegmentationConfig,
        returnSeconds: Bool,
        timeResolution: Int
    ) -> VadStreamResult {
        var nextState = state
        nextState.modelState = modelState
        nextState.processedSamples += chunkSampleCount

        // If the caller pins the negative threshold, derive a matching entry threshold via the offset
        let thresholdOverride: Float? = {
            guard let negative = config.negativeThreshold else { return nil }
            return min(1.0, negative + config.negativeThresholdOffset)
        }()
        let threshold = thresholdOverride ?? self.config.defaultThreshold

        let negativeThreshold = config.effectiveNegativeThreshold(baseThreshold: threshold)
        let speechPadSamples = Int(config.speechPadding * Double(Self.sampleRate))
        let minSilenceSamples = Int(config.minSilenceDuration * Double(Self.sampleRate))

        var event: VadStreamEvent?

        if probability >= threshold {
            nextState.tempEndSample = nil
            if !nextState.triggered {
                nextState.triggered = true
                let rawStart = nextState.processedSamples - speechPadSamples - chunkSampleCount
                let startSample = max(0, rawStart)
                event = makeStreamEvent(
                    kind: .speechStart,
                    sampleIndex: startSample,
                    returnSeconds: returnSeconds,
                    timeResolution: timeResolution
                )
            }
        } else if probability < negativeThreshold && nextState.triggered {
            if nextState.tempEndSample == nil {
                nextState.tempEndSample = nextState.processedSamples
            }
            if let silenceStart = nextState.tempEndSample,
                nextState.processedSamples - silenceStart >= minSilenceSamples
            {
                let rawEnd = silenceStart + speechPadSamples - chunkSampleCount
                let endSample = max(0, rawEnd)
                nextState.triggered = false
                nextState.tempEndSample = nil
                event = makeStreamEvent(
                    kind: .speechEnd,
                    sampleIndex: endSample,
                    returnSeconds: returnSeconds,
                    timeResolution: timeResolution
                )
            }
        }

        return VadStreamResult(state: nextState, event: event, probability: probability)
    }

    private func makeStreamEvent(
        kind: VadStreamEvent.Kind,
        sampleIndex: Int,
        returnSeconds: Bool,
        timeResolution: Int
    ) -> VadStreamEvent {
        let clampedSample = max(0, sampleIndex)
        if returnSeconds {
            let seconds = Double(clampedSample) / Double(Self.sampleRate)
            let factor = pow(10.0, Double(timeResolution))
            let rounded = (seconds * factor).rounded() / factor
            return VadStreamEvent(kind: kind, sampleIndex: clampedSample, time: rounded)
        }
        return VadStreamEvent(kind: kind, sampleIndex: clampedSample, time: nil)
    }
}
