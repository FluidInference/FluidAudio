#if os(macOS)
import AVFoundation
import Foundation
import FluidAudio

/// Audio loading and processing utilities
struct AudioProcessor {

    /// Load an audio file and convert to 16kHz mono Float32 samples using the shared library converter
    static func loadAudioFile(path: String) async throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        return try await loadAudioFileDirectly(url: url)
    }

    private static func loadAudioFileDirectly(url: URL) async throws -> [Float] {
        let converter = AudioConverter()
        return try await converter.convertFileToAsrSamples(url)
    }

    // Keep resampleAudio for internal callers (e.g. benchmarks) that may need array-only resampling.
    static func resampleAudio(
        _ samples: [Float], from sourceSampleRate: Double, to targetSampleRate: Double
    ) async throws -> [Float] {
        if sourceSampleRate == targetSampleRate { return samples }

        let ratio = sourceSampleRate / targetSampleRate
        let outputLength = Int(Double(samples.count) / ratio)
        var resampled: [Float] = []
        resampled.reserveCapacity(outputLength)

        for i in 0..<outputLength {
            let sourceIndex = Double(i) * ratio
            let index = Int(sourceIndex)

            if index < samples.count - 1 {
                let fraction = sourceIndex - Double(index)
                let sample = samples[index] * Float(1.0 - fraction) + samples[index + 1] * Float(fraction)
                resampled.append(sample)
            } else if index < samples.count {
                resampled.append(samples[index])
            }
        }

        return resampled
    }
}

#endif
