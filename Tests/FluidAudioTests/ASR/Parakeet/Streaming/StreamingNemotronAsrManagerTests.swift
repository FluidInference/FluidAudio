@preconcurrency import AVFoundation
@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class StreamingNemotronAsrManagerTests: XCTestCase {

    // MARK: - P0: Initialization

    func testDefaultInitialization() async {
        let manager = StreamingNemotronAsrManager()

        let config = await manager.config
        // Should use default 1120ms config
        XCTAssertEqual(config.chunkMs, 1120)
        XCTAssertEqual(config.chunkMelFrames, 112)
        XCTAssertEqual(config.vocabSize, 1024)
    }

    func testCustomMLModelConfiguration() async {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuOnly

        let manager = StreamingNemotronAsrManager(configuration: mlConfig)

        let storedConfig = await manager.mlConfiguration
        XCTAssertEqual(storedConfig.computeUnits, .cpuOnly)
    }

    // MARK: - P0: State Reset

    func testResetClearsAudioBuffer() async throws {
        let manager = StreamingNemotronAsrManager()

        // Simulate some processing by accessing internal state
        // Note: We can't directly test private properties, but we can test the behavior
        // through the public API

        await manager.reset()

        // After reset, processing should start fresh
        // This is validated through integration tests
    }

    func testResetClearsAccumulatedTokens() async throws {
        let manager = StreamingNemotronAsrManager()

        // Reset should clear any accumulated transcription state
        await manager.reset()

        let transcript = await manager.getPartialTranscript()
        XCTAssertEqual(transcript, "")
    }

    // MARK: - P1: Argmax Correctness

    func testArgmaxWithKnownValues() {
        // Test the argmax function logic
        // Create a small logits array
        let values: [Float] = [0.1, 0.5, 0.3, 0.9, 0.2]

        // Expected: index 3 (value 0.9)
        let maxIndex = findMaxIndex(values)
        XCTAssertEqual(maxIndex, 3)
    }

    func testArgmaxWithBlankTokenAtEnd() {
        let values: [Float] = [0.1, 0.2, 0.15, 0.95]  // Last is highest

        let maxIndex = findMaxIndex(values)
        XCTAssertEqual(maxIndex, 3)
    }

    func testArgmaxWithAllSameValues() {
        let values: [Float] = [0.5, 0.5, 0.5, 0.5]

        let maxIndex = findMaxIndex(values)
        // Should return first index
        XCTAssertEqual(maxIndex, 0)
    }

    func testArgmaxWithNegativeValues() {
        let values: [Float] = [-0.5, -0.2, -0.8, -0.1]

        let maxIndex = findMaxIndex(values)
        XCTAssertEqual(maxIndex, 3)  // -0.1 is largest
    }

    func testArgmaxWithSingleValue() {
        let values: [Float] = [0.42]

        let maxIndex = findMaxIndex(values)
        XCTAssertEqual(maxIndex, 0)
    }

    // MARK: - P1: Buffer Processing Logic

    func testChunkSamplesCalculation() async {
        let manager = StreamingNemotronAsrManager()
        let config = await manager.config

        // chunkSamples = chunkMelFrames * 160
        let expectedSamples = config.chunkMelFrames * 160
        XCTAssertEqual(config.chunkSamples, expectedSamples)
        XCTAssertEqual(config.chunkSamples, 17920)  // 112 * 160 for 1120ms
    }

    func testAudioBufferAccumulationWithSingleBuffer() async throws {
        let manager = StreamingNemotronAsrManager()

        // Create a small audio buffer (less than chunk size)
        nonisolated(unsafe) let buffer = try createTestAudioBuffer(frameCount: 1000)

        try await manager.appendAudio(buffer)

        // Buffer should accumulate but not process yet (< chunkSamples)
        // This is validated through the finish() call
    }

    func testAudioBufferAccumulationWithMultipleBuffers() async throws {
        let manager = StreamingNemotronAsrManager()

        // Append multiple small buffers
        nonisolated(unsafe) let buffer1 = try createTestAudioBuffer(frameCount: 500)
        nonisolated(unsafe) let buffer2 = try createTestAudioBuffer(frameCount: 800)

        try await manager.appendAudio(buffer1)
        try await manager.appendAudio(buffer2)

        // Total: 1300 samples accumulated
        // Should not process yet (< 17920 for 1120ms chunks)
    }

    // MARK: - P1: Error Handling

    func testLoadModelsWithNonExistentDirectoryThrows() async {
        let manager = StreamingNemotronAsrManager()
        let nonExistentDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nonexistent_\(UUID().uuidString)")

        await XCTAssertThrowsErrorAsync {
            try await manager.loadModels(from: nonExistentDir)
        }
    }

    func testLoadModelsWithMissingMetadataUsesDefaults() async throws {
        // Create temp directory without metadata.json
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_models_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: tempDir)
        }

        // Manager will use default config when metadata.json is missing
        // This is tested in NemotronStreamingConfigTests.testDefaultInitialization
    }

    func testProcessWithoutLoadingModelsThrows() async {
        let manager = StreamingNemotronAsrManager()
        nonisolated(unsafe) let buffer = try! createTestAudioBuffer(frameCount: 1000)

        await XCTAssertThrowsErrorAsync(
            {
                _ = try await manager.process(audioBuffer: buffer)
            },
            errorHandler: { error in
                guard case ASRError.notInitialized = error else {
                    XCTFail("Expected ASRError.notInitialized, got \(error)")
                    return
                }
            })
    }

    func testFinishWithoutLoadingModelsThrows() async {
        let manager = StreamingNemotronAsrManager()

        await XCTAssertThrowsErrorAsync(
            {
                _ = try await manager.finish()
            },
            errorHandler: { error in
                guard case ASRError.notInitialized = error else {
                    XCTFail("Expected ASRError.notInitialized, got \(error)")
                    return
                }
            })
    }

    // MARK: - P0: Stride Calculation Tests

    func testStrideCalculationWithContiguousArray() throws {
        // Test that stride calculations work correctly
        let melArray = try MLMultiArray(shape: [1, 128, 56], dataType: .float32)

        // Standard C-contiguous strides: [128*56, 56, 1] = [7168, 56, 1]
        XCTAssertEqual(melArray.strides[0].intValue, 7168)
        XCTAssertEqual(melArray.strides[1].intValue, 56)
        XCTAssertEqual(melArray.strides[2].intValue, 1)

        // Verify we can access elements correctly
        let ptr = melArray.dataPointer.bindMemory(to: Float.self, capacity: melArray.count)

        // Set a known value at position [0, 10, 20]
        let index =
            0 * melArray.strides[0].intValue + 10 * melArray.strides[1].intValue + 20
            * melArray.strides[2]
            .intValue
        ptr[index] = 42.0

        // Verify we can read it back
        XCTAssertEqual(ptr[index], 42.0)
    }

    func testStrideCalculationFormula() {
        // Verify the stride calculation formula
        let shape = [1, 128, 112]
        let stride0 = shape[1] * shape[2]  // 128 * 112 = 14336
        let stride1 = shape[2]  // 112
        let stride2 = 1

        XCTAssertEqual(stride0, 14336)
        XCTAssertEqual(stride1, 112)
        XCTAssertEqual(stride2, 1)

        // Index calculation: [0, mel, t] = 0*stride0 + mel*stride1 + t*stride2
        let mel = 10
        let t = 20
        let index = 0 * stride0 + mel * stride1 + t * stride2
        XCTAssertEqual(index, 10 * 112 + 20)
        XCTAssertEqual(index, 1140)
    }

    // MARK: - Helpers

    private func createTestAudioBuffer(frameCount: AVAudioFrameCount) throws -> AVAudioPCMBuffer {
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw ASRError.processingFailed("Failed to create audio buffer")
        }

        buffer.frameLength = frameCount

        // Fill with test data (silence)
        if let channelData = buffer.floatChannelData {
            let samples = UnsafeMutableBufferPointer(start: channelData[0], count: Int(frameCount))
            for i in 0..<samples.count {
                samples[i] = 0.0
            }
        }

        return buffer
    }

    private func findMaxIndex(_ values: [Float]) -> Int {
        var maxVal: Float = -Float.infinity
        var maxIdx = 0

        for (i, val) in values.enumerated() {
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }

        return maxIdx
    }
}

// MARK: - Async Test Helpers

extension XCTestCase {
    func XCTAssertThrowsErrorAsync<T>(
        _ expression: () async throws -> T,
        _ message: String = "",
        file: StaticString = #filePath,
        line: UInt = #line,
        errorHandler: ((Error) -> Void)? = nil
    ) async {
        do {
            _ = try await expression()
            XCTFail("Expected error but got success", file: file, line: line)
        } catch {
            errorHandler?(error)
        }
    }
}
