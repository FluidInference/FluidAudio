#if os(macOS)
import AVFoundation
import Foundation

public struct AudioProcessing {

// MARK: - Audio Loading

public static func loadAudioFile(path: String) async throws -> [Float] {
    let url = URL(fileURLWithPath: path)
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

public static func resampleAudio(
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
                samples[index] * Float(1.0 - fraction) + samples[index + 1]
                * Float(fraction)
            resampled.append(sample)
        } else if index < samples.count {
            resampled.append(samples[index])
        }
    }

    return resampled
}

// MARK: - Audio Validation

public static func isValidAudioFile(_ url: URL) async -> Bool {
    do {
        let _ = try AVAudioFile(forReading: url)
        return true
    } catch {
        return false
    }
}

// MARK: - VAD Audio Loading

public static func loadVadAudioData(_ audioFile: AVAudioFile) async throws -> [Float] {
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    // Early exit if already 16kHz - avoid resampling overhead
    let needsResampling = format.sampleRate != 16000

    // Use smaller buffer size for GitHub Actions memory constraints
    let bufferSize: AVAudioFrameCount = min(frameCount, 4096)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: bufferSize) else {
        throw NSError(domain: "AudioError", code: 1, userInfo: nil)
    }

    var allSamples: [Float] = []
    allSamples.reserveCapacity(Int(frameCount))

    // Read file in chunks to reduce memory pressure
    var remainingFrames = frameCount

    while remainingFrames > 0 {
        let framesToRead = min(remainingFrames, bufferSize)
        buffer.frameLength = 0  // Reset buffer

        try audioFile.read(into: buffer, frameCount: framesToRead)

        guard let floatData = buffer.floatChannelData?[0] else {
            throw NSError(domain: "AudioError", code: 2, userInfo: nil)
        }

        let actualFrameCount = Int(buffer.frameLength)
        if actualFrameCount == 0 { break }

        // Direct append without intermediate array creation
        let bufferPointer = UnsafeBufferPointer(start: floatData, count: actualFrameCount)
        allSamples.append(contentsOf: bufferPointer)

        remainingFrames -= AVAudioFrameCount(actualFrameCount)
    }

    // Resample to 16kHz if needed
    if needsResampling {
        allSamples = try await resampleVadAudio(
            allSamples, from: format.sampleRate, to: 16000)
    }

    return allSamples
}

public static func resampleVadAudio(
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
                samples[index] * Float(1.0 - fraction) + samples[index + 1]
                * Float(fraction)
            resampled.append(sample)
        } else if index < samples.count {
            resampled.append(samples[index])
        }
    }

    return resampled
}
}
#endif
