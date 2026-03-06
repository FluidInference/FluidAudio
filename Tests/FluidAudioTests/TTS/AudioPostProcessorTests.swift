import Foundation
import XCTest

@testable import FluidAudio

final class AudioPostProcessorTests: XCTestCase {

    // MARK: - De-Ess

    func testDeEssSilenceStaysSilent() {
        var samples = [Float](repeating: 0, count: 100)
        AudioPostProcessor.deEss(&samples)

        for (i, value) in samples.enumerated() {
            XCTAssertEqual(value, 0, accuracy: 1e-10, "Silent sample \(i) should remain 0 after de-essing")
        }
    }

    // MARK: - Smooth High Frequencies

    func testSmoothHighFrequenciesReducesPeaks() {
        // Alternating +1/-1 (Nyquist frequency — highest possible frequency)
        var squareWave = (0..<200).map { Float($0 % 2 == 0 ? 1 : -1) }
        let peakBefore = squareWave.map { abs($0) }.max()!

        AudioPostProcessor.smoothHighFrequencies(&squareWave)

        let peakAfter = squareWave.suffix(from: 10).map { abs($0) }.max()!
        XCTAssertLessThan(peakAfter, peakBefore, "Low-pass filter should reduce high-frequency peaks")
    }

    // MARK: - Remove Rumble

    func testRemoveRumblePreservesHighFrequency() {
        // Generate 1kHz sine wave (well above 80Hz cutoff)
        let sampleRate: Float = 24000
        let frequency: Float = 1000
        let duration = 200
        var samples = (0..<duration).map { i -> Float in
            sin(2 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let rmsBefore = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))
        AudioPostProcessor.removeRumble(&samples, sampleRate: sampleRate)

        // Skip transient at the start
        let steadyState = Array(samples.suffix(from: 50))
        let rmsAfter = sqrt(steadyState.map { $0 * $0 }.reduce(0, +) / Float(steadyState.count))

        // High-pass at 80Hz should preserve most of a 1kHz signal
        XCTAssertGreaterThan(rmsAfter, rmsBefore * 0.8, "1kHz signal should be largely preserved")
    }

    // MARK: - Full Pipeline

    func testApplyTtsPostProcessingNoNaN() {
        // Random-ish samples
        var samples = (0..<500).map { i -> Float in
            sin(Float(i) * 0.1) * 0.5 + cos(Float(i) * 0.3) * 0.3
        }

        AudioPostProcessor.applyTtsPostProcessing(&samples)

        for (i, value) in samples.enumerated() {
            XCTAssertFalse(value.isNaN, "Sample \(i) should not be NaN")
            XCTAssertFalse(value.isInfinite, "Sample \(i) should not be Inf")
        }
    }

    // MARK: - Short Input Edge Cases

    func testShortInputsDoNotCrash() {
        // Empty
        var empty = [Float]()
        AudioPostProcessor.deEss(&empty)
        AudioPostProcessor.smoothHighFrequencies(&empty)
        AudioPostProcessor.removeRumble(&empty)
        AudioPostProcessor.applyTtsPostProcessing(&empty)

        // Single element
        var single: [Float] = [0.5]
        AudioPostProcessor.deEss(&single)
        AudioPostProcessor.smoothHighFrequencies(&single)
        AudioPostProcessor.removeRumble(&single)

        // Two elements
        var two: [Float] = [0.5, -0.5]
        AudioPostProcessor.deEss(&two)
        AudioPostProcessor.smoothHighFrequencies(&two)
        AudioPostProcessor.removeRumble(&two)
    }
}
