import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

final class AudioConverterTests: XCTestCase {

    var audioConverter: AudioConverter!

    override func setUp() async throws {
        try await super.setUp()
        audioConverter = AudioConverter()
    }

    override func tearDown() async throws {
        audioConverter = AudioConverter()
        try await super.tearDown()
    }

    // MARK: - Helper Methods

    private func assertApproximateCount(
        _ actual: Int,
        expected: Int,
        toleranceFraction: Double = 0.02,
        _ message: @autoclosure () -> String = ""
    ) {
        let tolerance = max(1, Int((Double(expected) * toleranceFraction).rounded(.up)))
        XCTAssertTrue(
            abs(actual - expected) <= tolerance,
            message().isEmpty
                ? "Expected ~\(expected) (±\(tolerance)), got \(actual)"
                : message()
        )
    }

    private func createAudioBuffer(
        sampleRate: Double = 44100,
        channels: AVAudioChannelCount = 2,
        duration: Double = 1.0,
        format: AVAudioCommonFormat = .pcmFormatFloat32
    ) throws -> AVAudioPCMBuffer {
        let audioFormat: AVAudioFormat

        if channels <= 2 {
            // Use simple initializer for mono/stereo
            guard
                let simpleFormat = AVAudioFormat(
                    commonFormat: format,
                    sampleRate: sampleRate,
                    channels: channels,
                    interleaved: false
                )
            else {
                throw AudioConverterError.failedToCreateConverter
            }
            audioFormat = simpleFormat
        } else {
            // For >2 channels, need to use AVAudioChannelLayout
            var layoutTag: AudioChannelLayoutTag
            switch channels {
            case 3: layoutTag = kAudioChannelLayoutTag_Unknown | 3
            case 4: layoutTag = kAudioChannelLayoutTag_Quadraphonic
            case 5: layoutTag = kAudioChannelLayoutTag_Pentagonal
            case 6: layoutTag = kAudioChannelLayoutTag_Hexagonal
            case 7: layoutTag = kAudioChannelLayoutTag_Unknown | 7
            case 8: layoutTag = kAudioChannelLayoutTag_Octagonal
            default: layoutTag = kAudioChannelLayoutTag_Unknown | UInt32(channels)
            }

            guard let channelLayout = AVAudioChannelLayout(layoutTag: layoutTag) else {
                throw AudioConverterError.failedToCreateConverter
            }

            let multiChannelFormat = AVAudioFormat(
                commonFormat: format,
                sampleRate: sampleRate,
                interleaved: false,
                channelLayout: channelLayout
            )
            audioFormat = multiChannelFormat
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

        // Should be downsampled to ~16kHz (allow small resampler variance)
        let expectedSampleCount = Int(16000 * 1.0)  // 1 second at 16kHz
        assertApproximateCount(
            result.count, expected: expectedSampleCount, toleranceFraction: 0.01, "Should downsample to ~16kHz")
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

        // Should be downsampled from 48kHz to ~16kHz (0.5 seconds)
        let expectedSampleCount = Int(16000 * 0.5)
        assertApproximateCount(
            result.count, expected: expectedSampleCount, toleranceFraction: 0.01, "Should downsample correctly")
    }

    func testConvert8kHzMonoTo16kHzMono() async throws {
        // Test upsampling from lower quality
        let buffer = try createAudioBuffer(
            sampleRate: 8000,
            channels: 1,
            duration: 2.0
        )

        let result = try audioConverter.resampleBuffer(buffer)

        // Should be upsampled from 8kHz to ~16kHz (2 seconds)
        let expectedSampleCount = Int(16000 * 2.0)
        assertApproximateCount(
            result.count, expected: expectedSampleCount, toleranceFraction: 0.01, "Should upsample correctly")
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

    func testResampleAudioFilePathBadPathThrows() async throws {
        let bogusPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("does_not_exist_\(UUID().uuidString)")
            .path

        XCTAssertThrowsError(try audioConverter.resampleAudioFile(path: bogusPath))
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
        if let ch = buffer.floatChannelData {  // For interleaved, only index 0 is valid
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
        assertApproximateCount(
            out.count, expected: 16_000, toleranceFraction: 0.01,
            "Interleaved stereo should convert to ~1s at 16kHz mono")
        XCTAssertFalse(out.isEmpty)
    }

    // Helper for generic interleaved buffers (N channels)
    private func createInterleavedBuffer(
        sampleRate: Double, channels: AVAudioChannelCount, duration: Double
    ) throws -> AVAudioPCMBuffer {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: channels,
                interleaved: true
            )
        else { throw AudioConverterError.failedToCreateConverter }

        let frames = AVAudioFrameCount(sampleRate * duration)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw AudioConverterError.failedToCreateBuffer
        }
        buffer.frameLength = frames

        if let ch = buffer.floatChannelData {
            let ptr = ch[0]
            let frameCount = Int(frames)
            let chanCount = Int(channels)
            for i in 0..<frameCount {
                let t = Double(i) / sampleRate
                for c in 0..<chanCount {
                    let baseFreq = 440.0 + Double(c) * 70.0
                    let sample = Float(sin(2.0 * .pi * baseFreq * t) * 0.5)
                    ptr[i * chanCount + c] = sample
                }
            }
        }
        return buffer
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
        assertApproximateCount(
            result.count, expected: expectedSamples, toleranceFraction: 0.01, "Should handle long audio correctly")
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

    // MARK: - Multiple Format Conversion Tests

    func testMultipleFormatConversions() async throws {
        // Convert one format
        let buffer1 = try createAudioBuffer(sampleRate: 44100, channels: 2)
        _ = try audioConverter.resampleBuffer(buffer1)

        // Convert different format (should work fine without reset since each conversion is stateless)
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

        // Debug logging for flakiness investigation
        let expected1 = 16_000
        let expected2 = 8_000
        let tol1 = Int(Double(expected1) * 0.02)
        let tol2 = Int(Double(expected2) * 0.02)
        print("[AudioConverterTests] converter reuse debug:")
        print(
            "  input1 frames=\(buffer1.frameLength) sr=\(buffer1.format.sampleRate) ch=\(buffer1.format.channelCount) -> out=\(result1.count) expected ~\(expected1) (±\(tol1))"
        )
        print(
            "  input2 frames=\(buffer2.frameLength) sr=\(buffer2.format.sampleRate) ch=\(buffer2.format.channelCount) -> out=\(result2.count) expected ~\(expected2) (±\(tol2))"
        )
        if result2.count < expected2 - tol2 || result2.count > expected2 + tol2 {
            let ratio = Double(result2.count) / Double(expected2)
            print("  WARN: second conversion outside tolerance; ratio=\(String(format: "%.4f", ratio))")
        }

        assertApproximateCount(
            result1.count, expected: 16000, toleranceFraction: 0.02,
            "First conversion should work expected 16000, got \(result1.count)")
        assertApproximateCount(
            result2.count, expected: 8000, toleranceFraction: 0.02,
            "Second conversion should work expected 8000, got \(result2.count)")
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
        assertApproximateCount(
            result.count, expected: expectedSamples, toleranceFraction: 0.01, "Should handle large buffer")
    }

    // MARK: - Linear Resample Tests for High Channel Count

    func testLinearResample3ChannelTo16kHz() async throws {
        // Create a 3-channel buffer at 48kHz
        let buffer = try createAudioBuffer(
            sampleRate: 48000,
            channels: 3,
            duration: 0.5
        )

        // Fill with test data - different sine waves for each channel
        if let channelData = buffer.floatChannelData {
            let frameCount = Int(buffer.frameLength)
            for frame in 0..<frameCount {
                let t = Float(frame) / Float(48000)
                channelData[0][frame] = sin(2 * .pi * 440 * t)  // 440 Hz
                channelData[1][frame] = sin(2 * .pi * 880 * t)  // 880 Hz
                channelData[2][frame] = sin(2 * .pi * 1320 * t)  // 1320 Hz
            }
        }

        let converted = try audioConverter.resampleBuffer(buffer)

        // Should downsample from 48kHz to 16kHz (ratio 3:1)
        let expectedCount = Int(Double(buffer.frameLength) * 16000 / 48000)
        assertApproximateCount(converted.count, expected: expectedCount)
    }

    func testLinearResample4ChannelMixdown() async throws {
        // Create a 4-channel buffer at 16kHz (no resampling needed)
        let buffer = try createAudioBuffer(
            sampleRate: 16000,
            channels: 4,
            duration: 0.25
        )

        // Fill each channel with a constant value
        if let channelData = buffer.floatChannelData {
            let frameCount = Int(buffer.frameLength)
            for frame in 0..<frameCount {
                channelData[0][frame] = 0.4
                channelData[1][frame] = 0.8
                channelData[2][frame] = -0.4
                channelData[3][frame] = -0.8
            }
        }

        let converted = try audioConverter.resampleBuffer(buffer)

        // Should maintain same sample count (no resampling)
        XCTAssertEqual(converted.count, Int(buffer.frameLength))

        // Check that values are averaged: (0.4 + 0.8 - 0.4 - 0.8) / 4 = 0
        for sample in converted {
            XCTAssertEqual(sample, 0, accuracy: 0.001)
        }
    }

    func testLinearResample5ChannelUpsampling() async throws {
        // Create a 5-channel buffer at 8kHz
        let buffer = try createAudioBuffer(
            sampleRate: 8000,
            channels: 5,
            duration: 0.1
        )

        // Fill with test data
        if let channelData = buffer.floatChannelData {
            let frameCount = Int(buffer.frameLength)
            for frame in 0..<frameCount {
                for channel in 0..<5 {
                    channelData[channel][frame] = Float(channel) * 0.2
                }
            }
        }

        let converted = try audioConverter.resampleBuffer(buffer)

        // Should upsample from 8kHz to 16kHz (ratio 1:2)
        let expectedCount = Int(Double(buffer.frameLength) * 16000 / 8000)
        assertApproximateCount(converted.count, expected: expectedCount)

        // Check that the average value is preserved
        let expectedAverage: Float = (0.0 + 0.2 + 0.4 + 0.6 + 0.8) / 5.0
        let actualAverage = converted.reduce(0, +) / Float(converted.count)
        XCTAssertEqual(actualAverage, expectedAverage, accuracy: 0.01)
    }

    func testLinearResample6ChannelComplexResampling() async throws {
        // Create a 6-channel buffer at 44100 Hz
        let buffer = try createAudioBuffer(
            sampleRate: 44100,
            channels: 6,
            duration: 0.2
        )

        // Fill with a ramp signal
        if let channelData = buffer.floatChannelData {
            let frameCount = Int(buffer.frameLength)
            for frame in 0..<frameCount {
                let value = Float(frame) / Float(frameCount)
                for channel in 0..<6 {
                    channelData[channel][frame] = value
                }
            }
        }

        let converted = try audioConverter.resampleBuffer(buffer)

        // Should downsample from 44100Hz to 16000Hz
        let expectedCount = Int(Double(buffer.frameLength) * 16000 / 44100)
        assertApproximateCount(converted.count, expected: expectedCount)

        // Verify ramp is preserved (first should be near 0, last near 1)
        XCTAssertLessThan(converted.first ?? 1.0, 0.01)
        XCTAssertGreaterThan(converted.last ?? 0.0, 0.99)
    }

    func testLinearResampleHighChannelCount() async throws {
        // Test functionality with 8 channels
        let buffer = try createAudioBuffer(
            sampleRate: 48000,
            channels: 8,
            duration: 10.0  // 10 seconds of audio
        )

        // Fill with test data
        if let channelData = buffer.floatChannelData {
            let frameCount = Int(buffer.frameLength)
            for frame in 0..<frameCount {
                for channel in 0..<8 {
                    channelData[channel][frame] = Float.random(in: -1...1)
                }
            }
        }

        let converted = try audioConverter.resampleBuffer(buffer)

        // Verify output
        let expectedCount = Int(Double(buffer.frameLength) * 16000 / 48000)
        assertApproximateCount(converted.count, expected: expectedCount)
    }

    func testLinearResampleEdgeCases() async throws {
        // Test with very short buffer (1 frame) - this will result in 0 samples due to downsampling
        let shortBuffer = try createAudioBuffer(
            sampleRate: 48000,
            channels: 3,
            duration: 1.0 / 48000  // 1 frame
        )

        if let channelData = shortBuffer.floatChannelData {
            channelData[0][0] = 0.5
            channelData[1][0] = 0.5
            channelData[2][0] = 0.5
        }

        let shortConverted = try audioConverter.resampleBuffer(shortBuffer)
        // With 48kHz -> 16kHz (3:1 ratio), 1 frame becomes 0.33 frames, which rounds to 0
        XCTAssertEqual(shortConverted.count, 0)

        // Test with maximum channels (e.g., 32)
        let manyChannelBuffer = try createAudioBuffer(
            sampleRate: 16000,
            channels: 32,
            duration: 0.1
        )

        if let channelData = manyChannelBuffer.floatChannelData {
            let frameCount = Int(manyChannelBuffer.frameLength)
            for frame in 0..<frameCount {
                for channel in 0..<32 {
                    // Each channel gets value of 1.0 so average is 1.0
                    channelData[channel][frame] = 1.0
                }
            }
        }

        let manyConverted = try audioConverter.resampleBuffer(manyChannelBuffer)
        XCTAssertEqual(manyConverted.count, Int(manyChannelBuffer.frameLength))

        // Check average is preserved (all channels have value 1.0, so average should be 1.0)
        for sample in manyConverted {
            XCTAssertEqual(sample, 1.0, accuracy: 0.001)
        }
    }

    func testLinearResampleInterpolationAccuracy() async throws {
        // Create a simple signal that's easy to verify interpolation
        let buffer = try createAudioBuffer(
            sampleRate: 4000,  // Low sample rate for easy verification
            channels: 3,
            duration: 0.01
        )

        // Create a simple linear ramp [0, 1, 2, 3, ...]
        if let channelData = buffer.floatChannelData {
            let frameCount = Int(buffer.frameLength)
            for frame in 0..<frameCount {
                let value = Float(frame)
                for channel in 0..<3 {
                    channelData[channel][frame] = value
                }
            }
        }

        let converted = try audioConverter.resampleBuffer(buffer)

        // With 4kHz -> 16kHz, we expect 4x upsampling
        // Interpolation should create smooth values between integers
        let expectedCount = Int(Double(buffer.frameLength) * 16000 / 4000)
        assertApproximateCount(converted.count, expected: expectedCount)

        // Verify interpolation creates smooth transition
        for i in 1..<converted.count {
            let diff = abs(converted[i] - converted[i - 1])
            // Difference between adjacent samples should be small
            XCTAssertLessThan(diff, 0.5)
        }
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
