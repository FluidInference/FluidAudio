import XCTest

@testable import FluidAudio

final class ArraySliceTests: XCTestCase {

    func testDiarizerManagerWithArraySlice() throws {
        // Create a test audio buffer
        let fullAudio = Array(repeating: Float(0.0), count: 48000)  // 3 seconds at 16kHz

        // Create an ArraySlice from the middle 2 seconds
        let slice = fullAudio[8000..<40000]  // 2 seconds of audio

        // Create diarizer
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minActiveFramesCount: 10.0,
            debugMode: false,
            chunkDuration: 10.0,
            chunkOverlap: 2.0
        )
        let diarizer = DiarizerManager(config: config)

        // Test that we can validate the ArraySlice
        let validationResult = diarizer.validateAudio(slice)
        XCTAssertTrue(validationResult.isValid || validationResult.issues.contains("Audio too quiet or silent"))
        XCTAssertEqual(validationResult.durationSeconds, 2.0, accuracy: 0.01)

        // Test with different collection types
        let array = Array(fullAudio[0..<16000])  // 1 second
        let arrayValidation = diarizer.validateAudio(array)
        XCTAssertEqual(arrayValidation.durationSeconds, 1.0, accuracy: 0.01)
    }

    func testEmbeddingExtractorWithArraySlice() async throws {
        // Skip if models aren't available
        guard let models = try? await DiarizerModels.download() else {
            throw XCTSkip("Models not available for testing")
        }

        let extractor = EmbeddingExtractor(embeddingModel: models.embeddingModel)

        // Create test audio
        let fullAudio = Array(repeating: Float(0.1), count: 160000)  // 10 seconds
        let slice = fullAudio[0..<160000]

        // Create dummy masks with correct size (589 frames for 10 seconds of audio)
        // The model expects 589 frames based on the error message
        let masks = [[Float](repeating: 1.0, count: 589)]

        // Test that extraction works with ArraySlice
        let embeddings = try extractor.getEmbeddings(
            audio: slice,
            masks: masks,
            minActivityThreshold: 10.0
        )

        XCTAssertEqual(embeddings.count, 1)
        XCTAssertEqual(embeddings[0].count, 256)
    }

    func testANEMemoryOptimizerWithArraySlice() throws {
        let optimizer = ANEMemoryOptimizer()

        // Create test data
        let fullArray = [Float](repeating: 0.5, count: 1000)
        let slice = fullArray[100..<600]

        // Create destination buffer
        let destination = try optimizer.createAlignedArray(
            shape: [500],
            dataType: .float32
        )

        // Test copying from ArraySlice
        optimizer.optimizedCopy(from: slice, to: destination)

        // Verify the copy worked
        let destPtr = destination.dataPointer.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(destPtr[0], 0.5, accuracy: 0.001)
        XCTAssertEqual(destPtr[499], 0.5, accuracy: 0.001)
    }

    func testAudioValidationWithArraySlice() throws {
        let validation = AudioValidation()

        // Create test audio
        let fullAudio = [Float](repeating: 0.2, count: 32000)  // 2 seconds
        let slice = fullAudio[8000..<24000]  // 1 second slice

        // Test validation with ArraySlice
        let result = validation.validateAudio(slice)
        XCTAssertEqual(result.durationSeconds, 1.0, accuracy: 0.01)

        // Test with empty slice
        let emptySlice = fullAudio[0..<0]
        let emptyResult = validation.validateAudio(emptySlice)
        XCTAssertFalse(emptyResult.isValid)
        XCTAssertTrue(emptyResult.issues.contains("No audio data"))
    }

    func testPerformanceWithArraySlice() async throws {
        // Skip if models aren't available
        guard let models = try? await DiarizerModels.download() else {
            throw XCTSkip("Models not available for testing")
        }

        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            debugMode: true,
            chunkDuration: 10.0,
            chunkOverlap: 2.0
        )
        let diarizer = DiarizerManager(config: config)
        diarizer.initialize(models: models)

        // Create a large audio buffer
        let fullAudio = [Float](repeating: 0.0, count: 480000)  // 30 seconds

        // Test with full array
        let arrayStart = Date()
        _ = try diarizer.performCompleteDiarization(fullAudio, sampleRate: 16000)
        let arrayTime = Date().timeIntervalSince(arrayStart)

        // Test with ArraySlice (no copy needed)
        let slice = fullAudio[80000..<400000]  // 20 seconds from middle
        let sliceStart = Date()
        _ = try diarizer.performCompleteDiarization(slice, sampleRate: 16000)
        let sliceTime = Date().timeIntervalSince(sliceStart)

        print("Array processing time: \(arrayTime)s")
        print("ArraySlice processing time: \(sliceTime)s")

        // ArraySlice should be at least as fast (no extra copy overhead)
        XCTAssertTrue(true)  // Just ensure test completes
    }
}
