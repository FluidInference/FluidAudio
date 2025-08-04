import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class SpeakerClusteringTests: XCTestCase {

    var clustering: SpeakerClustering!
    var config: DiarizerConfig!

    override func setUp() {
        super.setUp()
        config = DiarizerConfig(clusteringThreshold: 0.7)
        clustering = SpeakerClustering(config: config)
    }

    override func tearDown() {
        clustering = nil
        config = nil
        super.tearDown()
    }

    // MARK: - Cosine Distance Tests

    func testCosineDistanceIdenticalVectors() {
        let vec1: [Float] = [1.0, 0.0, 0.0]
        let vec2: [Float] = [1.0, 0.0, 0.0]

        let distance = clustering.cosineDistance(vec1, vec2)
        XCTAssertEqual(distance, 0.0, accuracy: 0.0001)
    }

    func testCosineDistanceOrthogonalVectors() {
        let vec1: [Float] = [1.0, 0.0, 0.0]
        let vec2: [Float] = [0.0, 1.0, 0.0]

        let distance = clustering.cosineDistance(vec1, vec2)
        XCTAssertEqual(distance, 1.0, accuracy: 0.0001)
    }

    func testCosineDistanceOppositeVectors() {
        let vec1: [Float] = [1.0, 0.0, 0.0]
        let vec2: [Float] = [-1.0, 0.0, 0.0]

        let distance = clustering.cosineDistance(vec1, vec2)
        XCTAssertEqual(distance, 2.0, accuracy: 0.0001)
    }

    func testCosineDistanceNormalizedVectors() {
        let vec1: [Float] = [0.6, 0.8, 0.0]
        let vec2: [Float] = [0.8, 0.6, 0.0]

        let distance = clustering.cosineDistance(vec1, vec2)
        // cos(θ) = 0.96, distance = 1 - 0.96 = 0.04
        XCTAssertEqual(distance, 0.04, accuracy: 0.01)
    }

    // MARK: - Speaker Assignment Tests

    func testAssignSpeakerFirstSpeaker() {
        var speakerDB: [String: [Float]] = [:]
        let embedding = createEmbedding(pattern: [1.0, 0.0, 0.0])

        let speakerId = clustering.assignSpeaker(embedding: embedding, speakerDB: &speakerDB)

        XCTAssertEqual(speakerId, "Speaker 1")
        XCTAssertEqual(speakerDB.count, 1)
        XCTAssertNotNil(speakerDB["Speaker 1"])
    }

    func testAssignSpeakerSimilarEmbedding() {
        var speakerDB: [String: [Float]] = [:]
        let embedding1 = createEmbedding(pattern: [1.0, 0.0, 0.0])
        let embedding2 = createEmbedding(pattern: [0.95, 0.05, 0.0])

        // Add first speaker
        let speaker1 = clustering.assignSpeaker(embedding: embedding1, speakerDB: &speakerDB)
        XCTAssertEqual(speaker1, "Speaker 1")

        // Similar embedding should be assigned to same speaker
        let speaker2 = clustering.assignSpeaker(embedding: embedding2, speakerDB: &speakerDB)
        XCTAssertEqual(speaker2, "Speaker 1")
        XCTAssertEqual(speakerDB.count, 1)
    }

    func testAssignSpeakerNewSpeaker() {
        var speakerDB: [String: [Float]] = [:]
        let embedding1 = createEmbedding(pattern: [1.0, 0.0, 0.0])
        let embedding2 = createEmbedding(pattern: [0.0, 1.0, 0.0])

        // Add first speaker
        let speaker1 = clustering.assignSpeaker(embedding: embedding1, speakerDB: &speakerDB)
        XCTAssertEqual(speaker1, "Speaker 1")

        // Different embedding should create new speaker
        let speaker2 = clustering.assignSpeaker(embedding: embedding2, speakerDB: &speakerDB)
        XCTAssertEqual(speaker2, "Speaker 2")
        XCTAssertEqual(speakerDB.count, 2)
    }

    func testUpdateSpeakerEmbedding() {
        var speakerDB: [String: [Float]] = [:]
        let embedding1 = createEmbedding(pattern: [1.0, 0.0, 0.0])
        let embedding2 = createEmbedding(pattern: [0.8, 0.2, 0.0])

        // Add initial speaker
        _ = clustering.assignSpeaker(embedding: embedding1, speakerDB: &speakerDB)
        let originalEmbedding = speakerDB["Speaker 1"]!

        // Update with new embedding
        clustering.updateSpeakerEmbedding("Speaker 1", embedding2, speakerDB: &speakerDB)
        let updatedEmbedding = speakerDB["Speaker 1"]!

        // Verify embedding was updated (should be weighted average)
        XCTAssertNotEqual(originalEmbedding[0], updatedEmbedding[0])
        XCTAssertNotEqual(originalEmbedding[1], updatedEmbedding[1])
    }

    func testClusteringWithCustomThreshold() {
        var speakerDB: [String: [Float]] = [:]
        let embedding1 = createEmbedding(pattern: [1.0, 0.0, 0.0])
        let embedding2 = createEmbedding(pattern: [0.5, 0.5, 0.0])  // ~50% similar, cosine distance ~0.29

        // With low threshold 0.2, should create new speaker
        clustering = SpeakerClustering(config: DiarizerConfig(clusteringThreshold: 0.2))
        let speaker1 = clustering.assignSpeaker(embedding: embedding1, speakerDB: &speakerDB)
        let speaker2 = clustering.assignSpeaker(embedding: embedding2, speakerDB: &speakerDB)
        XCTAssertEqual(speaker1, "Speaker 1")
        XCTAssertEqual(speaker2, "Speaker 2")

        // With threshold 0.5, should assign to same speaker
        speakerDB = [:]
        clustering = SpeakerClustering(config: DiarizerConfig(clusteringThreshold: 0.5))
        let speaker3 = clustering.assignSpeaker(embedding: embedding1, speakerDB: &speakerDB)
        let speaker4 = clustering.assignSpeaker(embedding: embedding2, speakerDB: &speakerDB)
        XCTAssertEqual(speaker3, "Speaker 1")
        XCTAssertEqual(speaker4, "Speaker 1")
    }

    func testCalculateSpeakerActivities() {
        // Create binarized segments: [batch][frames][speakers]
        let binarizedSegments: [[[Float]]] = [
            [
                [1.0, 0.0],  // Frame 0: Speaker 0 active
                [1.0, 0.0],  // Frame 1: Speaker 0 active
                [0.0, 1.0],  // Frame 2: Speaker 1 active
                [0.0, 1.0],  // Frame 3: Speaker 1 active
                [1.0, 0.0],  // Frame 4: Speaker 0 active
            ]
        ]

        let activities = clustering.calculateSpeakerActivities(binarizedSegments)

        XCTAssertEqual(activities.count, 2)
        XCTAssertEqual(activities[0], 3.0)  // Speaker 0 active in 3 frames
        XCTAssertEqual(activities[1], 2.0)  // Speaker 1 active in 2 frames
    }

    func testAssignSpeakerSequential() {
        var speakerDB: [String: [Float]] = [:]

        // Test sequential assignment of multiple speakers
        let embeddings = [
            createEmbedding(pattern: [1.0, 0.0, 0.0]),  // Speaker 1
            createEmbedding(pattern: [0.0, 1.0, 0.0]),  // Speaker 2
            createEmbedding(pattern: [0.0, 0.0, 1.0]),  // Speaker 3
            createEmbedding(pattern: [0.95, 0.05, 0.0]),  // Back to Speaker 1
        ]

        var speakerIds: [String] = []
        for embedding in embeddings {
            let speakerId = clustering.assignSpeaker(embedding: embedding, speakerDB: &speakerDB)
            speakerIds.append(speakerId)
        }

        XCTAssertEqual(speakerIds[0], "Speaker 1")
        XCTAssertEqual(speakerIds[1], "Speaker 2")
        XCTAssertEqual(speakerIds[2], "Speaker 3")
        XCTAssertEqual(speakerIds[3], "Speaker 1")  // Should match first speaker
        XCTAssertEqual(speakerDB.count, 3)
    }

    // MARK: - Performance Tests

    func testAssignSpeakerPerformance() {
        var speakerDB: [String: [Float]] = [:]

        // Pre-populate with 10 speakers
        for i in 0..<10 {
            let embedding = createEmbedding(pattern: createBaseVector(for: i))
            _ = clustering.assignSpeaker(embedding: embedding, speakerDB: &speakerDB)
        }

        // Measure assignment performance
        measure {
            for i in 0..<100 {
                let embedding = createEmbedding(pattern: createBaseVector(for: i % 10))
                _ = clustering.assignSpeaker(embedding: embedding, speakerDB: &speakerDB)
            }
        }
    }

    // MARK: - Helper Methods

    private func createEmbedding(pattern: [Float]) -> [Float] {
        // Normalize the pattern to create a proper embedding
        let magnitude = sqrt(pattern.reduce(0) { $0 + $1 * $1 })
        let normalized = pattern.map { $0 / magnitude }

        // Extend to realistic embedding size (e.g., 192 dimensions)
        var fullEmbedding = normalized
        while fullEmbedding.count < 192 {
            fullEmbedding.append(0.0)
        }

        return fullEmbedding
    }

    private func createBaseVector(for speaker: Int) -> [Float] {
        // Create distinct base vectors for different speakers
        var vector = Array(repeating: Float(0.0), count: 192)

        switch speaker {
        case 0:
            vector[0] = 1.0
            vector[10] = 0.5
        case 1:
            vector[1] = 1.0
            vector[20] = 0.5
        case 2:
            vector[2] = 1.0
            vector[30] = 0.5
        case 3:
            vector[3] = 1.0
            vector[40] = 0.5
        case 4:
            vector[4] = 1.0
            vector[50] = 0.5
        default:
            vector[speaker % 192] = 1.0
        }

        // Normalize
        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        return vector.map { $0 / magnitude }
    }

    private func addNoise(to vector: [Float], magnitude: Float) -> [Float] {
        return vector.map { value in
            value + Float.random(in: -magnitude...magnitude)
        }
    }

    private func mostCommonElement(in array: [Int]) -> Int? {
        let counts = array.reduce(into: [:]) { counts, element in
            counts[element, default: 0] += 1
        }
        return counts.max(by: { $0.value < $1.value })?.key
    }
}
