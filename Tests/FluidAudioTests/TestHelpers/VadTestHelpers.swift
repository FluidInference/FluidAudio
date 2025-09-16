import Foundation

@testable import FluidAudio

// Deterministic VAD results builder (256ms chunks at 16kHz)
// Shared for use across multiple VAD-related tests.
func makeVadResults(_ pattern: [(isActive: Bool, seconds: Double)]) -> ([VadResult], Int) {
    // Match VadManager expectations: 4096-sample chunks at 16kHz
    let chunkSize = VadManager.chunkSize
    let sampleRate = Double(VadManager.sampleRate)
    let chunkDuration = Double(chunkSize) / sampleRate

    var results: [VadResult] = []
    for (active, seconds) in pattern {
        let chunks = max(0, Int((seconds / chunkDuration).rounded()))
        if chunks == 0 { continue }
        let prob: Float = active ? 0.95 : 0.05
        for _ in 0..<chunks {
            results.append(VadResult(probability: prob, isVoiceActive: active, processingTime: 0))
        }
    }

    let totalSamples = results.count * chunkSize
    return (results, totalSamples)
}
