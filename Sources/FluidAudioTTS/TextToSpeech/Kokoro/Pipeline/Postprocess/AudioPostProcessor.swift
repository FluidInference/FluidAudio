import Accelerate
import Foundation

/// Audio post-processing utilities for improving TTS quality
public enum AudioPostProcessor {

    /// Apply de-essing to reduce harsh sibilant sounds (s, sh, z, etc.)
    /// Uses a simple high-shelf filter to reduce frequencies above the cutoff.
    ///
    /// - Parameters:
    ///   - samples: Audio samples to process (modified in place)
    ///   - sampleRate: Sample rate in Hz (e.g., 24000)
    ///   - cutoffHz: Frequency above which to reduce (default 6000 Hz for sibilants)
    ///   - reductionDb: Amount to reduce high frequencies in dB (default -3 dB)
    public static func deEss(
        _ samples: inout [Float],
        sampleRate: Float = 24000,
        cutoffHz: Float = 6000,
        reductionDb: Float = -3.0
    ) {
        guard samples.count > 2 else { return }

        // Biquad high-shelf filter coefficients
        // H(s) = A * (s^2 + sqrt(A)/Q * s + A) / (A*s^2 + sqrt(A)/Q * s + 1)
        let A = powf(10, reductionDb / 40)  // sqrt of linear gain
        let omega = 2 * Float.pi * cutoffHz / sampleRate
        let sinOmega = sin(omega)
        let cosOmega = cos(omega)
        let Q: Float = 0.707  // Butterworth Q

        let alpha = sinOmega / (2 * Q)
        let sqrtA = sqrt(A)

        // High-shelf coefficients
        let b0 = A * ((A + 1) + (A - 1) * cosOmega + 2 * sqrtA * alpha)
        let b1 = -2 * A * ((A - 1) + (A + 1) * cosOmega)
        let b2 = A * ((A + 1) + (A - 1) * cosOmega - 2 * sqrtA * alpha)
        let a0 = (A + 1) - (A - 1) * cosOmega + 2 * sqrtA * alpha
        let a1 = 2 * ((A - 1) - (A + 1) * cosOmega)
        let a2 = (A + 1) - (A - 1) * cosOmega - 2 * sqrtA * alpha

        // Normalize
        let b0n = b0 / a0
        let b1n = b1 / a0
        let b2n = b2 / a0
        let a1n = a1 / a0
        let a2n = a2 / a0

        // Apply biquad filter using direct form II transposed
        var z1: Float = 0
        var z2: Float = 0

        for i in 0..<samples.count {
            let x = samples[i]
            let y = b0n * x + z1
            z1 = b1n * x - a1n * y + z2
            z2 = b2n * x - a2n * y
            samples[i] = y
        }
    }

    /// Apply gentle high-frequency smoothing using a simple one-pole low-pass filter.
    /// This is lighter weight than the full de-esser but can help with harshness.
    ///
    /// - Parameters:
    ///   - samples: Audio samples to process (modified in place)
    ///   - sampleRate: Sample rate in Hz
    ///   - cutoffHz: Cutoff frequency for the low-pass filter
    public static func smoothHighFrequencies(
        _ samples: inout [Float],
        sampleRate: Float = 24000,
        cutoffHz: Float = 10000
    ) {
        guard samples.count > 1 else { return }

        // One-pole low-pass: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        let rc = 1.0 / (2.0 * Float.pi * cutoffHz)
        let dt = 1.0 / sampleRate
        let alpha = dt / (rc + dt)

        var prev = samples[0]
        for i in 1..<samples.count {
            let filtered = alpha * samples[i] + (1 - alpha) * prev
            samples[i] = filtered
            prev = filtered
        }
    }

    /// Apply a high-pass filter to reduce low-frequency rumble.
    ///
    /// - Parameters:
    ///   - samples: Audio samples to process (modified in place)
    ///   - sampleRate: Sample rate in Hz
    ///   - cutoffHz: Cutoff frequency (default 80 Hz)
    public static func removeRumble(
        _ samples: inout [Float],
        sampleRate: Float = 24000,
        cutoffHz: Float = 80
    ) {
        guard samples.count > 1 else { return }

        // One-pole high-pass: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        let rc = 1.0 / (2.0 * Float.pi * cutoffHz)
        let dt = 1.0 / sampleRate
        let alpha = rc / (rc + dt)

        var prevX = samples[0]
        var prevY: Float = 0

        for i in 1..<samples.count {
            let x = samples[i]
            let y = alpha * (prevY + x - prevX)
            samples[i] = y
            prevX = x
            prevY = y
        }
    }

    /// Apply a complete post-processing chain for improved TTS quality.
    /// Includes: rumble removal, de-essing, and optional smoothing.
    ///
    /// - Parameters:
    ///   - samples: Audio samples to process (modified in place)
    ///   - sampleRate: Sample rate in Hz
    ///   - deEssAmount: De-essing reduction in dB (0 to disable, -3 to -6 typical)
    ///   - smoothing: Whether to apply additional high-frequency smoothing
    public static func applyTtsPostProcessing(
        _ samples: inout [Float],
        sampleRate: Float = 24000,
        deEssAmount: Float = -3.0,
        smoothing: Bool = false
    ) {
        // Remove low-frequency rumble
        removeRumble(&samples, sampleRate: sampleRate, cutoffHz: 80)

        // Apply de-essing if requested
        if deEssAmount < 0 {
            deEss(&samples, sampleRate: sampleRate, cutoffHz: 6000, reductionDb: deEssAmount)
        }

        // Optional smoothing
        if smoothing {
            smoothHighFrequencies(&samples, sampleRate: sampleRate, cutoffHz: 10000)
        }
    }
}
