#if os(macOS)
import AVFoundation
import Foundation

/// Audio loading and processing utilities
struct AudioProcessor {

    static func loadAudioFile(path: String) async throws -> [Float] {
        let url = URL(fileURLWithPath: path)

        // Try to load the file directly first
        do {
            return try await loadAudioFileDirectly(url: url)
        } catch {
            // If direct loading fails (e.g., FLAC in CI), try converting with ffmpeg
            print("Direct audio loading failed, attempting ffmpeg conversion: \(error.localizedDescription)")
            return try await loadAudioFileWithFFmpeg(path: path)
        }
    }

    private static func loadAudioFileDirectly(url: URL) async throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)

        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw NSError(
                domain: "AudioError", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
        }

        try audioFile.read(into: buffer)

        guard let floatChannelData = buffer.floatChannelData else {
            throw NSError(
                domain: "AudioError", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to get float channel data"])
        }

        let actualFrameCount = Int(buffer.frameLength)
        var samples: [Float] = []

        if format.channelCount == 1 {
            samples = Array(
                UnsafeBufferPointer(start: floatChannelData[0], count: actualFrameCount))
        } else {
            // Mix stereo to mono
            let leftChannel = UnsafeBufferPointer(
                start: floatChannelData[0], count: actualFrameCount)
            let rightChannel = UnsafeBufferPointer(
                start: floatChannelData[1], count: actualFrameCount)

            samples = zip(leftChannel, rightChannel).map { (left, right) in
                (left + right) / 2.0
            }
        }

        // Resample to 16kHz if necessary
        if format.sampleRate != 16000 {
            samples = try await resampleAudio(samples, from: format.sampleRate, to: 16000)
        }

        return samples
    }

    static func resampleAudio(
        _ samples: [Float], from sourceSampleRate: Double, to targetSampleRate: Double
    ) async throws -> [Float] {
        if sourceSampleRate == targetSampleRate {
            return samples
        }

        let ratio = sourceSampleRate / targetSampleRate
        let outputLength = Int(Double(samples.count) / ratio)
        var resampled: [Float] = []
        resampled.reserveCapacity(outputLength)

        for i in 0..<outputLength {
            let sourceIndex = Double(i) * ratio
            let index = Int(sourceIndex)

            if index < samples.count - 1 {
                let fraction = sourceIndex - Double(index)
                let sample =
                    samples[index] * Float(1.0 - fraction) + samples[index + 1] * Float(fraction)
                resampled.append(sample)
            } else if index < samples.count {
                resampled.append(samples[index])
            }
        }

        return resampled
    }

    /// Load audio file using ffmpeg conversion as fallback for unsupported formats
    private static func loadAudioFileWithFFmpeg(path: String) async throws -> [Float] {
        let fileManager = FileManager.default
        let tempDir = fileManager.temporaryDirectory
        let tempWavPath = tempDir.appendingPathComponent("\(UUID().uuidString).wav")

        defer {
            // Clean up temp file
            try? fileManager.removeItem(at: tempWavPath)
        }

        // Convert to WAV using ffmpeg
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "ffmpeg",
            "-i", path,  // Input file
            "-ar", "16000",  // Sample rate
            "-ac", "1",  // Mono
            "-f", "wav",  // WAV format
            "-y",  // Overwrite output
            tempWavPath.path,  // Output path
            "-loglevel", "error",  // Only show errors
        ]

        let pipe = Pipe()
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()

            if process.terminationStatus != 0 {
                let errorData = pipe.fileHandleForReading.readDataToEndOfFile()
                let errorMessage = String(data: errorData, encoding: .utf8) ?? "Unknown error"
                throw NSError(
                    domain: "AudioError", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "ffmpeg conversion failed: \(errorMessage)"])
            }

            // Now load the converted WAV file
            return try await loadAudioFileDirectly(url: tempWavPath)

        } catch {
            // If ffmpeg is not available or fails, throw a more informative error
            throw NSError(
                domain: "AudioError", code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Failed to load audio file. FLAC files require ffmpeg for conversion in CI environment. Error: \(error.localizedDescription)"
                ])
        }
    }
}

#endif
