import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

/// Shared fixture infrastructure for diarization tests.
///
/// Generates a deterministic multi-segment waveform with silence gaps, writes it to a
/// temporary WAV file, and caches it for reuse across tests within the same process.
enum DiarizationTestFixtures {
    static let fixtureSampleRate = 16_000

    nonisolated(unsafe) private static var cachedFixtureAudioURL: URL?

    /// Returns a cached URL to the fixture WAV file, creating it on first access.
    static func fixtureAudioFileURL() throws -> URL {
        if let cached = cachedFixtureAudioURL,
            FileManager.default.fileExists(atPath: cached.path)
        {
            return cached
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("diarization-fixture-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        try writeFixtureAudio(to: url)
        cachedFixtureAudioURL = url
        return url
    }

    /// Loads fixture audio resampled to the given sample rate, optionally limited to a duration.
    static func fixtureAudio(sampleRate: Int, limitSeconds: Double? = nil) throws -> [Float] {
        let converter = AudioConverter(sampleRate: Double(sampleRate))
        let audio = try converter.resampleAudioFile(try fixtureAudioFileURL())
        guard let limitSeconds else {
            return audio
        }
        let sampleCount = min(audio.count, Int(limitSeconds * Double(sampleRate)))
        return Array(audio.prefix(sampleCount))
    }

    /// Loads a slice of fixture audio at the given sample rate.
    static func fixtureAudio(
        sampleRate: Int, startSeconds: Double, durationSeconds: Double
    ) throws -> [Float] {
        let converter = AudioConverter(sampleRate: Double(sampleRate))
        let audio = try converter.resampleAudioFile(try fixtureAudioFileURL())
        let startSample = min(audio.count, Int(startSeconds * Double(sampleRate)))
        let endSample = min(audio.count, startSample + Int(durationSeconds * Double(sampleRate)))
        return Array(audio[startSample..<endSample])
    }

    /// Splits samples into chunks with rotating sizes.
    static func chunk(_ samples: [Float], sizes: [Int]) -> [[Float]] {
        var chunks: [[Float]] = []
        var start = 0
        var index = 0
        while start < samples.count {
            let size = sizes[index % sizes.count]
            let stop = min(samples.count, start + size)
            chunks.append(Array(samples[start..<stop]))
            start = stop
            index += 1
        }
        return chunks
    }

    // MARK: - Private

    private static func writeFixtureAudio(to url: URL) throws {
        let sampleRate = Double(fixtureSampleRate)
        let samples = makeFixtureSamples(sampleRate: sampleRate)
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count)
            )
        else {
            XCTFail("Failed to allocate fixture audio buffer")
            return
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            guard let destination = buffer.floatChannelData?[0] else { return }
            destination.update(from: source.baseAddress!, count: samples.count)
        }

        let file = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        try file.write(from: buffer)
    }

    private static func makeFixtureSamples(sampleRate: Double) -> [Float] {
        let segments: [(duration: Double, amplitude: Float, frequency: Double)] = [
            (1.0, 0.20, 220),
            (0.35, 0.00, 0),
            (1.1, 0.32, 330),
            (0.25, 0.00, 0),
            (1.0, 0.28, 180),
            (0.40, 0.00, 0),
            (1.3, 0.36, 260),
            (0.30, 0.00, 0),
            (1.1, 0.24, 410),
        ]

        var output: [Float] = []
        for (duration, amplitude, frequency) in segments {
            let frameCount = Int(duration * sampleRate)
            guard amplitude > 0, frequency > 0 else {
                output.append(contentsOf: repeatElement(0, count: frameCount))
                continue
            }

            for frame in 0..<frameCount {
                let time = Double(frame) / sampleRate
                let envelope = Float(min(1.0, time * 12.0)) * Float(min(1.0, (duration - time) * 12.0))
                let carrier = sin(2.0 * Double.pi * frequency * time)
                let harmonic = 0.35 * sin(2.0 * Double.pi * frequency * 2.03 * time)
                output.append(Float((carrier + harmonic) * Double(amplitude * envelope)))
            }
        }
        return output
    }
}
