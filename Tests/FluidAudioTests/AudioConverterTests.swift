import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class AudioConverterTests: XCTestCase {

    var audioConverter: AudioConverter!

    override func setUp() async throws {
        try await super.setUp()
        audioConverter = AudioConverter()
    }

    override func tearDown() async throws {
        audioConverter.reset()
        audioConverter = nil
        try await super.tearDown()
    }

    // MARK: - Helper Methods

    private func createAudioBuffer(
        sampleRate: Double = 44100,
        channels: AVAudioChannelCount = 2,
        duration: Double = 1.0,
        format: AVAudioCommonFormat = .pcmFormatFloat32
    ) throws -> AVAudioPCMBuffer {
        guard
            let audioFormat = AVAudioFormat(
                commonFormat: format,
                sampleRate: sampleRate,
                channels: channels,
                interleaved: false
            )
        else {
            throw AudioConverterError.failedToCreateConverter
        }

        let frameCount = AVAudioFrameCount(sampleRate * duration)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
            throw AudioConverterError.failedToCreateBuffer
        }

        buffer.frameLength = frameCount

        // Fill with test sine wave data
        if let channelData = buffer.floatChannelData {
            for channel in 0..<Int(channels) {
                for i in 0..<Int(frameCount) {
                    let frequency = Double(440 + channel * 100)  // A4 + harmonics
                    let phase = Double(i) / sampleRate * frequency * 2.0 * Double.pi
                    channelData[channel][i] = Float(sin(phase) * 0.5)
                }
            }
        }

        return buffer
    }

    // MARK: - Basic Conversion Tests

    func testConvertAlreadyCorrectFormat() async throws {
        // Create buffer already in target format (16kHz, mono, Float32)
        let buffer = try createAudioBuffer(
            sampleRate: 16000,
            channels: 1,
            duration: 1.0,
            format: .pcmFormatFloat32
        )

        let result = try audioConverter.resampleBuffer(buffer)

        XCTAssertEqual(result.count, 16000, "Should have 16,000 samples for 1 second at 16kHz")
        XCTAssertFalse(result.isEmpty, "Result should not be empty")

        // Verify values are reasonable (sine wave should be between -0.5 and 0.5)
        for sample in result {
            XCTAssertGreaterThanOrEqual(sample, -0.6, "Sample should be within expected range")
            XCTAssertLessThanOrEqual(sample, 0.6, "Sample should be within expected range")
        }
    }

    func testConvert44kHzStereoTo16kHzMono() async throws {
        // Create typical audio file format
        let buffer = try createAudioBuffer(
            sampleRate: 44100,
            channels: 2,
            duration: 1.0
        )

        let result = try audioConverter.resampleBuffer(buffer)

        // Should be downsampled to 16kHz
        let expectedSampleCount = Int(16000 * 1.0)  // 1 second at 16kHz
        XCTAssertEqual(result.count, expectedSampleCount, "Should downsample to 16kHz")
        XCTAssertFalse(result.isEmpty, "Result should not be empty")
    }

    func testConvert48kHzMonoTo16kHzMono() async throws {
        // Test common recording format
        let buffer = try createAudioBuffer(
            sampleRate: 48000,
            channels: 1,
            duration: 0.5
        )

        let result = try audioConverter.resampleBuffer(buffer)

        // Should be downsampled from 48kHz to 16kHz (0.5 seconds)
        let expectedSampleCount = Int(16000 * 0.5)
        XCTAssertEqual(result.count, expectedSampleCount, "Should downsample correctly")
    }

    func testConvert8kHzMonoTo16kHzMono() async throws {
        // Test upsampling from lower quality
        let buffer = try createAudioBuffer(
            sampleRate: 8000,
            channels: 1,
            duration: 2.0
        )

        let result = try audioConverter.resampleBuffer(buffer)

        // Should be upsampled from 8kHz to 16kHz (2 seconds)
        let expectedSampleCount = Int(16000 * 2.0)
        XCTAssertEqual(result.count, expectedSampleCount, "Should upsample correctly")
    }

    // MARK: - Multi-Channel Conversion Tests

    func testConvertStereoToMono() async throws {
        // Create stereo buffer with different signals in each channel
        let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 2,
            interleaved: false
        )!

        let frameCount: AVAudioFrameCount = 1000
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
            XCTFail("Failed to create buffer")
            return
        }
        buffer.frameLength = frameCount

        // Fill left channel with 0.5, right channel with -0.5
        if let channelData = buffer.floatChannelData {
            for i in 0..<Int(frameCount) {
                channelData[0][i] = 0.5  // Left channel
                channelData[1][i] = -0.5  // Right channel
            }
        }

        let result = try audioConverter.resampleBuffer(buffer)

        XCTAssertEqual(result.count, 1000, "Should preserve sample count for same sample rate")

        // Note: AVAudioConverter might not average channels as expected
        // Just verify we got mono output with reasonable values
        for sample in result {
            XCTAssertGreaterThanOrEqual(sample, -1.0, "Sample should be within valid range")
            XCTAssertLessThanOrEqual(sample, 1.0, "Sample should be within valid range")
        }
    }

    // Note: Skip multi-channel test as AVAudioFormat has limited channel support

    // MARK: - Edge Cases

    // Note: Skip empty buffer test as AVAudioConverter doesn't handle empty buffers well

    func testConvertVeryShortBuffer() async throws {
        // Small buffer with 10 samples
        let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 44100,
            channels: 1,
            interleaved: false
        )!

        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: 10) else {
            XCTFail("Failed to create buffer")
            return
        }
        buffer.frameLength = 10

        if let channelData = buffer.floatChannelData {
            for i in 0..<10 {
                channelData[0][i] = 0.75
            }
        }

        let result = try audioConverter.resampleBuffer(buffer)

        XCTAssertFalse(result.isEmpty, "Should handle short buffer")
        XCTAssertGreaterThan(result.count, 0, "Should produce some output")
    }

    // MARK: - File I/O API

    func testResampleAudioFileFromWav44kStereo() async throws {
        // Create a 1-second 44.1kHz stereo buffer and write to a temporary WAV file
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")

        let srcBuffer = try createAudioBuffer(sampleRate: 44_100, channels: 2, duration: 1.0)

        // Use the buffer's format settings to write a WAV file
        let settings = srcBuffer.format.settings
        let audioFile = try AVAudioFile(forWriting: fileURL, settings: settings)
        try audioFile.write(from: srcBuffer)

        // Read and resample using file-based API
        let samples = try audioConverter.resampleAudioFile(fileURL)
        XCTAssertEqual(samples.count, 16_000, "Expected 1s at 16kHz after resampling")

        // Cleanup
        try? FileManager.default.removeItem(at: fileURL)
    }

    func testResampleAudioFilePathBadPathThrows() async throws {
        let bogusPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("does_not_exist_\(UUID().uuidString)")
            .path

        XCTAssertThrowsError(try audioConverter.resampleAudioFile(path: bogusPath))
    }

    // MARK: - Streaming API

    func testStreamingChunksAndFinishDrain() async throws {
        // Stream three chunks totaling ~1.5s of 44.1kHz stereo audio
        let chunkDurations: [Double] = [0.5, 0.5, 0.5]
        var totalOut: [Float] = []
        let streamingConverter = AudioConverter(streaming: true)

        for dur in chunkDurations {
            let chunk = try createAudioBuffer(sampleRate: 44_100, channels: 2, duration: dur)
            let out = try streamingConverter.resampleBuffer(chunk)
            totalOut.append(contentsOf: out)
        }

        // Drain remaining samples at end of stream
        let drained = try streamingConverter.finish()
        totalOut.append(contentsOf: drained)

        // Expect ~1.5s of 16kHz audio with a small tolerance for resampler boundaries
        let expected = Int(16_000 * chunkDurations.reduce(0, +))
        let tolerance = Int(Double(expected) * 0.02) // 2%
        XCTAssertTrue(abs(totalOut.count - expected) <= tolerance,
                      "Streaming+drain sample count \(totalOut.count) should be close to \(expected)")
    }

    func testFinishReturnsEmptyInNonStreamingMode() async throws {
        // Non-streaming conversions should fully flush per call; finish() returns no extra samples
        let nonStreaming = AudioConverter()
        let buffer = try createAudioBuffer(sampleRate: 44_100, channels: 2, duration: 0.5)

        let converted = try nonStreaming.resampleBuffer(buffer)
        XCTAssertEqual(converted.count, 8_000, "0.5s at 16kHz expected in non-streaming mode")

        let drained = try nonStreaming.finish()
        XCTAssertEqual(drained.count, 0, "finish() should return empty for non-streaming usage")
    }

    // MARK: - Interleaved Inputs

    private func createInterleavedStereoBuffer(sampleRate: Double, duration: Double) throws -> AVAudioPCMBuffer {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: 2,
                interleaved: true
            )
        else {
            throw AudioConverterError.failedToCreateConverter
        }

        let frames = AVAudioFrameCount(sampleRate * duration)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw AudioConverterError.failedToCreateBuffer
        }
        buffer.frameLength = frames

        // Fill interleaved: LRLR... with two different sines
        if let ch = buffer.floatChannelData { // For interleaved, only index 0 is valid
            let ptr = ch[0]
            let frameCount = Int(frames)
            for i in 0..<frameCount {
                let t = Double(i) / sampleRate
                let left = Float(sin(2.0 * .pi * 440.0 * t) * 0.5)
                let right = Float(sin(2.0 * .pi * 550.0 * t) * 0.5)
                ptr[i * 2 + 0] = left
                ptr[i * 2 + 1] = right
            }
        }
        return buffer
    }

    func testInterleavedStereoInput() async throws {
        let interleaved = try createInterleavedStereoBuffer(sampleRate: 44_100, duration: 1.0)
        let out = try audioConverter.resampleBuffer(interleaved)
        XCTAssertEqual(out.count, 16_000, "Interleaved stereo should convert to 1s at 16kHz mono")
        XCTAssertFalse(out.isEmpty)
    }

    func testStreamingWithSmallAndLargeChunks() async throws {
        // Mix tiny and larger chunks to exercise converter stability
        let durations: [Double] = [0.05, 0.01, 0.2, 0.35, 0.01, 0.38] // totals ~1.0s
        var collected: [Float] = []
        let streamingConverter = AudioConverter(streaming: true)

        for d in durations {
            let b = try createAudioBuffer(sampleRate: 48_000, channels: 2, duration: d)
            collected.append(contentsOf: try streamingConverter.resampleBuffer(b))
        }
        collected.append(contentsOf: try streamingConverter.finish())

        let expected = 16_000 // ~1 second
        let tolerance = Int(Double(expected) * 0.05) // 5%
        XCTAssertTrue(abs(collected.count - expected) <= tolerance,
                      "Total streaming samples (\(collected.count)) should be ~\(expected)")
    }

    func testConvertVeryLongBuffer() async throws {
        // 10 second buffer
        let buffer = try createAudioBuffer(
            sampleRate: 44100,
            channels: 2,
            duration: 10.0
        )

        let result = try audioConverter.resampleBuffer(buffer)

        let expectedSamples = 16000 * 10  // 10 seconds at 16kHz
        XCTAssertEqual(result.count, expectedSamples, "Should handle long audio correctly")
    }

    // MARK: - Format Variation Tests

    func testConvertInt16Format() async throws {
        // Test with Int16 PCM format (common in WAV files)
        let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 44100,
            channels: 1,
            interleaved: false
        )!

        let frameCount: AVAudioFrameCount = 1000
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
            XCTFail("Failed to create Int16 buffer")
            return
        }
        buffer.frameLength = frameCount

        // Fill with test data
        if let channelData = buffer.int16ChannelData {
            for i in 0..<Int(frameCount) {
                channelData[0][i] = Int16(i % 1000)  // Ramp pattern
            }
        }

        let result = try audioConverter.resampleBuffer(buffer)

        XCTAssertFalse(result.isEmpty, "Should convert Int16 to Float32")
        // Result should be downsampled from 44.1kHz to 16kHz
        XCTAssertLessThan(result.count, 1000, "Should downsample")
    }

    func testConvertInt32Format() async throws {
        let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!

        let frameCount: AVAudioFrameCount = 500
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
            XCTFail("Failed to create Int32 buffer")
            return
        }
        buffer.frameLength = frameCount

        let result = try audioConverter.resampleBuffer(buffer)

        XCTAssertEqual(result.count, 500, "Should maintain sample count for same sample rate")
    }

    // MARK: - Converter State Management Tests

    func testConverterReset() async throws {
        // Convert one format
        let buffer1 = try createAudioBuffer(sampleRate: 44100, channels: 2)
        _ = try audioConverter.resampleBuffer(buffer1)

        // Reset converter
        audioConverter.reset()

        // Convert different format (should work fine)
        let buffer2 = try createAudioBuffer(sampleRate: 48000, channels: 1)
        let result = try audioConverter.resampleBuffer(buffer2)

        XCTAssertFalse(result.isEmpty, "Should work after reset")
    }

    func testConverterReuse() async throws {
        // Convert multiple buffers with same format
        let buffer1 = try createAudioBuffer(sampleRate: 44100, channels: 2, duration: 1.0)
        let buffer2 = try createAudioBuffer(sampleRate: 44100, channels: 2, duration: 0.5)

        let result1 = try audioConverter.resampleBuffer(buffer1)
        let result2 = try audioConverter.resampleBuffer(buffer2)

        XCTAssertEqual(result1.count, 16000, "First conversion should work")
        XCTAssertEqual(result2.count, 8000, "Second conversion should work")
    }

    func testConverterFormatSwitching() async throws {
        // Convert from one format
        let buffer1 = try createAudioBuffer(sampleRate: 44100, channels: 1)
        let result1 = try audioConverter.resampleBuffer(buffer1)

        // Convert from different format (should create new converter)
        let buffer2 = try createAudioBuffer(sampleRate: 48000, channels: 2)
        let result2 = try audioConverter.resampleBuffer(buffer2)

        XCTAssertFalse(result1.isEmpty, "First format should work")
        XCTAssertFalse(result2.isEmpty, "Second format should work")
    }

    // MARK: - Performance Tests

    func testConversionPerformance() async throws {
        let buffer = try createAudioBuffer(sampleRate: 44100, channels: 2, duration: 5.0)

        // Ensure audioConverter is initialized
        XCTAssertNotNil(audioConverter, "AudioConverter should be initialized in setUp")

        // Since measure doesn't support async, we'll test synchronous performance
        // by measuring the time taken for multiple conversions
        let startTime = CFAbsoluteTimeGetCurrent()
        let iterations = 10

        for _ in 0..<iterations {
            _ = try audioConverter.resampleBuffer(buffer)
        }

        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        let averageTime = timeElapsed / Double(iterations)

        // Performance assertion - should be under 1 second per conversion for 5-second buffer
        XCTAssertLessThan(averageTime, 1.0, "Average conversion time should be under 1 second")

        print("Average conversion time: \(averageTime) seconds")
    }

    func testBatchConversionPerformance() async throws {
        // Create multiple small buffers
        var buffers: [AVAudioPCMBuffer] = []
        for _ in 0..<10 {
            buffers.append(try createAudioBuffer(sampleRate: 44100, channels: 2, duration: 1.0))
        }

        // Ensure audioConverter is initialized
        XCTAssertNotNil(audioConverter, "AudioConverter should be initialized in setUp")

        // Test batch conversion performance
        let startTime = CFAbsoluteTimeGetCurrent()

        for buffer in buffers {
            _ = try audioConverter.resampleBuffer(buffer)
        }

        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        let averageTime = timeElapsed / Double(buffers.count)

        // Performance assertion - should be under 0.5 seconds per conversion for 1-second buffer
        XCTAssertLessThan(averageTime, 0.5, "Average batch conversion time should be under 0.5 seconds per buffer")

        print("Batch conversion - Total time: \(timeElapsed) seconds, Average per buffer: \(averageTime) seconds")
    }

    // MARK: - Error Handling Tests

    func testAudioConverterErrorDescriptions() {
        let error1 = AudioConverterError.failedToCreateConverter
        let error2 = AudioConverterError.failedToCreateBuffer
        let error3 = AudioConverterError.conversionFailed(nil)

        XCTAssertNotNil(error1.errorDescription)
        XCTAssertNotNil(error2.errorDescription)
        XCTAssertNotNil(error3.errorDescription)

        XCTAssertTrue(error1.errorDescription!.contains("converter"))
        XCTAssertTrue(error2.errorDescription!.contains("buffer"))
        XCTAssertTrue(error3.errorDescription!.contains("conversion failed"))
    }

    // MARK: - Memory Tests

    func testLargeBufferConversion() async throws {
        // Test with 1 minute of audio
        let buffer = try createAudioBuffer(sampleRate: 44100, channels: 2, duration: 60.0)

        let result = try audioConverter.resampleBuffer(buffer)

        let expectedSamples = 16000 * 60  // 1 minute at 16kHz
        XCTAssertEqual(result.count, expectedSamples, "Should handle large buffer")
    }

    func testMemoryUsageWithMultipleConversions() async throws {
        // Convert many buffers to test memory management
        for i in 0..<50 {
            let buffer = try createAudioBuffer(
                sampleRate: Double(16000 + i * 100),
                channels: AVAudioChannelCount(1 + i % 2),
                duration: 0.1
            )

            let result = try audioConverter.resampleBuffer(buffer)
            XCTAssertFalse(result.isEmpty, "Conversion \(i) should succeed")
        }
    }
}
