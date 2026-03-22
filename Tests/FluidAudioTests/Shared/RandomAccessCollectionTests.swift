import XCTest

@testable import FluidAudio

final class RandomAccessCollectionTests: XCTestCase {

    // MARK: - Custom Collection Types for Testing

    /// A custom RandomAccessCollection that wraps an array
    struct CustomAudioBuffer: RandomAccessCollection {
        typealias Element = Float
        typealias Index = Int

        private let storage: [Float]

        init(_ data: [Float]) {
            self.storage = data
        }

        var startIndex: Int { storage.startIndex }
        var endIndex: Int { storage.endIndex }

        func index(after i: Int) -> Int {
            storage.index(after: i)
        }

        func index(before i: Int) -> Int {
            storage.index(before: i)
        }

        subscript(position: Int) -> Float {
            storage[position]
        }
    }

    /// A strided collection that skips every other sample (for decimation)
    struct StridedAudioBuffer: RandomAccessCollection {
        typealias Element = Float
        typealias Index = Int

        private let storage: [Float]
        private let stride: Int

        init(_ data: [Float], stride: Int = 2) {
            self.storage = data
            self.stride = stride
        }

        var startIndex: Int { 0 }
        var endIndex: Int { (storage.count + stride - 1) / stride }

        func index(after i: Int) -> Int { i + 1 }
        func index(before i: Int) -> Int { i - 1 }

        subscript(position: Int) -> Float {
            storage[position * stride]
        }
    }

    // MARK: - DiarizerManager Tests

    func testDiarizerWithContiguousArray() throws {
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minActiveFramesCount: 10.0,
            debugMode: false
        )
        let diarizer = DiarizerManager(config: config)

        // ContiguousArray is optimized for performance
        let audio = ContiguousArray<Float>(repeating: 0.1, count: 32000)  // 2 seconds

        let result = diarizer.validateAudio(audio)
        XCTAssertEqual(result.durationSeconds, 2.0, accuracy: 0.01)
    }

    func testDiarizerWithCustomCollection() throws {
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minActiveFramesCount: 10.0,
            debugMode: false
        )
        let diarizer = DiarizerManager(config: config)

        // Custom collection wrapper
        let audioData = [Float](repeating: 0.1, count: 16000)  // 1 second
        let customBuffer = CustomAudioBuffer(audioData)

        let result = diarizer.validateAudio(customBuffer)
        XCTAssertEqual(result.durationSeconds, 1.0, accuracy: 0.01)
    }

    func testDiarizerWithLazyCollection() throws {
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minActiveFramesCount: 10.0,
            debugMode: false
        )
        let diarizer = DiarizerManager(config: config)

        // Lazy collection - transforms values on access
        let baseAudio = [Float](repeating: 0.05, count: 16000)
        let lazyAudio = baseAudio.lazy.map { $0 * 2.0 }  // Amplify by 2x lazily

        let result = diarizer.validateAudio(lazyAudio)
        XCTAssertEqual(result.durationSeconds, 1.0, accuracy: 0.01)
    }

    func testDiarizerWithReversedCollection() throws {
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minActiveFramesCount: 10.0,
            debugMode: false
        )
        let diarizer = DiarizerManager(config: config)

        // Reversed collection - useful for backward processing
        let audioData = [Float](repeating: 0.1, count: 16000)
        let reversedAudio = audioData.reversed()

        let result = diarizer.validateAudio(reversedAudio)
        XCTAssertEqual(result.durationSeconds, 1.0, accuracy: 0.01)
    }

    func testDiarizerWithChainedCollections() throws {
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minActiveFramesCount: 10.0,
            debugMode: false
        )
        let diarizer = DiarizerManager(config: config)

        // Chained collections - slice of lazy map
        let baseAudio = [Float](repeating: 0.05, count: 32000)
        let processedAudio = baseAudio[8000..<24000]  // Slice first
            .lazy.map { $0 * 2.0 }  // Then amplify lazily

        let result = diarizer.validateAudio(processedAudio)
        XCTAssertEqual(result.durationSeconds, 1.0, accuracy: 0.01)
    }

    // MARK: - ANEMemoryOptimizer Tests

    func testANEOptimizerWithContiguousArray() throws {
        let optimizer = ANEMemoryOptimizer()

        let audio = ContiguousArray<Float>(repeating: 0.5, count: 1000)
        let destination = try optimizer.createAlignedArray(
            shape: [1000],
            dataType: .float32
        )

        optimizer.optimizedCopy(from: audio, to: destination)

        let destPtr = destination.dataPointer.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(destPtr[0], 0.5, accuracy: 0.001)
        XCTAssertEqual(destPtr[999], 0.5, accuracy: 0.001)
    }

    func testANEOptimizerWithCustomCollection() throws {
        let optimizer = ANEMemoryOptimizer()

        let audioData = [Float](repeating: 0.3, count: 500)
        let customBuffer = CustomAudioBuffer(audioData)

        let destination = try optimizer.createAlignedArray(
            shape: [500],
            dataType: .float32
        )

        optimizer.optimizedCopy(from: customBuffer, to: destination)

        let destPtr = destination.dataPointer.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(destPtr[0], 0.3, accuracy: 0.001)
        XCTAssertEqual(destPtr[499], 0.3, accuracy: 0.001)
    }

    func testANEOptimizerWithLazyCollection() throws {
        let optimizer = ANEMemoryOptimizer()

        // Lazy collection that transforms values
        let baseData = [Float](repeating: 0.25, count: 100)
        let lazyData = baseData.lazy.map { $0 * 4.0 }  // Should become 1.0

        let destination = try optimizer.createAlignedArray(
            shape: [100],
            dataType: .float32
        )

        optimizer.optimizedCopy(from: lazyData, to: destination)

        let destPtr = destination.dataPointer.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(destPtr[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(destPtr[99], 1.0, accuracy: 0.001)
    }

    // MARK: - EmbeddingExtractor Tests

    func testEmbeddingExtractorWithContiguousArray() async throws {
        guard let models = try? await DiarizerModels.download() else {
            throw XCTSkip("Models not available for testing")
        }

        let extractor = EmbeddingExtractor(embeddingModel: models.embeddingModel)

        let audio = ContiguousArray<Float>(repeating: 0.1, count: 160000)
        let masks = [[Float](repeating: 1.0, count: 589)]

        let embeddings = try extractor.getEmbeddings(
            audio: audio,
            masks: masks,
            minActivityThreshold: 10.0
        )

        XCTAssertEqual(embeddings.count, 1)
        XCTAssertEqual(embeddings[0].count, 256)
    }

    func testEmbeddingExtractorWithCustomCollection() async throws {
        guard let models = try? await DiarizerModels.download() else {
            throw XCTSkip("Models not available for testing")
        }

        let extractor = EmbeddingExtractor(embeddingModel: models.embeddingModel)

        let audioData = [Float](repeating: 0.1, count: 160000)
        let customBuffer = CustomAudioBuffer(audioData)
        let masks = [[Float](repeating: 1.0, count: 589)]

        let embeddings = try extractor.getEmbeddings(
            audio: customBuffer,
            masks: masks,
            minActivityThreshold: 10.0
        )

        XCTAssertEqual(embeddings.count, 1)
        XCTAssertEqual(embeddings[0].count, 256)
    }

    // MARK: - Performance Comparison Tests

    func testPerformanceComparisonAcrossCollectionTypes() throws {
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minActiveFramesCount: 10.0,
            debugMode: false
        )
        let diarizer = DiarizerManager(config: config)

        let sampleCount = 160000  // 10 seconds

        // Test Array
        let array = [Float](repeating: 0.1, count: sampleCount)
        let arrayStart = Date()
        _ = diarizer.validateAudio(array)
        let arrayTime = Date().timeIntervalSince(arrayStart)

        // Test ContiguousArray
        let contiguous = ContiguousArray<Float>(repeating: 0.1, count: sampleCount)
        let contiguousStart = Date()
        _ = diarizer.validateAudio(contiguous)
        let contiguousTime = Date().timeIntervalSince(contiguousStart)

        // Test ArraySlice
        let slice = array[0..<sampleCount]
        let sliceStart = Date()
        _ = diarizer.validateAudio(slice)
        let sliceTime = Date().timeIntervalSince(sliceStart)

        // Test Custom Collection
        let custom = CustomAudioBuffer(array)
        let customStart = Date()
        _ = diarizer.validateAudio(custom)
        let customTime = Date().timeIntervalSince(customStart)

        // Test Lazy Collection
        let lazy = array.lazy.map { $0 }
        let lazyStart = Date()
        _ = diarizer.validateAudio(lazy)
        let lazyTime = Date().timeIntervalSince(lazyStart)

        print("Performance Comparison:")
        print("  Array:           \(arrayTime * 1000)ms")
        print("  ContiguousArray: \(contiguousTime * 1000)ms")
        print("  ArraySlice:      \(sliceTime * 1000)ms")
        print("  CustomBuffer:    \(customTime * 1000)ms")
        print("  LazyCollection:  \(lazyTime * 1000)ms")

        // All should complete successfully
        XCTAssertTrue(true)
    }

    // MARK: - Edge Cases

    func testEmptyCollections() throws {
        let validation = AudioValidation()

        // Empty array
        let emptyArray = [Float]()
        let arrayResult = validation.validateAudio(emptyArray)
        XCTAssertFalse(arrayResult.isValid)

        // Empty slice
        let array = [Float](repeating: 0, count: 100)
        let emptySlice = array[50..<50]
        let sliceResult = validation.validateAudio(emptySlice)
        XCTAssertFalse(sliceResult.isValid)

        // Empty custom collection
        let emptyCustom = CustomAudioBuffer([])
        let customResult = validation.validateAudio(emptyCustom)
        XCTAssertFalse(customResult.isValid)
    }

    func testSingleElementCollections() throws {
        let validation = AudioValidation()

        // Single element array
        let singleArray = [Float(0.5)]
        let arrayResult = validation.validateAudio(singleArray)
        XCTAssertEqual(arrayResult.durationSeconds, 1.0 / 16000.0, accuracy: 0.0001)

        // Single element custom collection
        let singleCustom = CustomAudioBuffer([0.5])
        let customResult = validation.validateAudio(singleCustom)
        XCTAssertEqual(customResult.durationSeconds, 1.0 / 16000.0, accuracy: 0.0001)
    }

    func testComplexCollectionChaining() async throws {
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

        // Create a complex chained collection
        let baseAudio = [Float](repeating: 0.0, count: 320000)  // 20 seconds

        // Slice -> Lazy map -> Process
        let complexCollection = baseAudio[80000..<240000]  // Take middle 10 seconds
            .lazy.map { $0 * 0.5 }  // Reduce amplitude lazily

        // This should work without issues
        let result = try diarizer.performCompleteDiarization(
            complexCollection,
            sampleRate: 16000
        )

        XCTAssertNotNil(result)
    }
}
