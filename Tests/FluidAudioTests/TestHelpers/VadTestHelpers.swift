import Foundation

@testable import FluidAudio

// Deterministic VAD results builder (256ms chunks at 16kHz)
// Shared for use across multiple VAD-related tests.
func makeVadResults(_ pattern: [(isActive: Bool, seconds: Double)]) -> ([VadResult], Int) {
    // Match VadManager expectations: 4096-sample chunks at 16kHz
    let chunkSize = VadManager.chunkSize
    let sampleRate = Double(VadManager.sampleRate)
    let chunkDuration = Double(chunkSize) / sampleRate

    let initialState = VadState.initial()
    let hiddenCount = initialState.hiddenState.count
    let cellCount = initialState.cellState.count
    let contextCount = initialState.context.count

    func makeState(forChunk index: Int, isActive: Bool) -> VadState {
        if index == 0 {
            return initialState
        }

        let offset = Float(index) * 0.0005
        let hiddenBase: Float = isActive ? 0.75 : -0.75
        let cellBase: Float = isActive ? 0.5 : -0.5
        let contextBase: Float = isActive ? 0.1 : -0.1

        let hidden = (0..<hiddenCount).map { i in hiddenBase + offset + Float(i) * 0.00001 }
        let cell = (0..<cellCount).map { i in cellBase + offset + Float(i) * 0.00002 }
        let context = (0..<contextCount).map { i in contextBase + offset + Float(i) * 0.00005 }

        return VadState(hiddenState: hidden, cellState: cell, context: context)
    }

    var results: [VadResult] = []
    var chunkIndex = 0
    for (active, seconds) in pattern {
        let chunks = max(0, Int((seconds / chunkDuration).rounded()))
        if chunks == 0 { continue }
        let prob: Float = active ? 0.95 : 0.05
        for _ in 0..<chunks {
            let state = makeState(forChunk: chunkIndex, isActive: active)
            results.append(
                VadResult(
                    probability: prob,
                    isVoiceActive: active,
                    processingTime: 0,
                    outputState: state
                )
            )
            chunkIndex += 1
        }
    }

    let totalSamples = results.count * chunkSize
    return (results, totalSamples)
}
