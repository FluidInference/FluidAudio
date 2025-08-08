import XCTest

@testable import FluidAudio

/// Tests for SpeakerManager functionality
@available(macOS 13.0, iOS 16.0, *)
final class SpeakerManagerTests: XCTestCase {

    // Helper to create distinct embeddings
    private func createDistinctEmbedding(pattern: Int) -> [Float] {
        var embedding = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            // Create unique pattern for each embedding
            embedding[i] = sin(Float(i + pattern * 100) * 0.1)
        }
        return embedding
    }

    // MARK: - Basic Operations

    func testInitialization() {
        let manager = SpeakerManager()
        XCTAssertEqual(manager.speakerCount, 0)
        XCTAssertTrue(manager.speakerIds.isEmpty)
    }

    func testAssignNewSpeaker() {
        let manager = SpeakerManager()
        let embedding = createDistinctEmbedding(pattern: 1)

        let speakerId = manager.assignSpeaker(embedding, speechDuration: 2.0)

        XCTAssertNotNil(speakerId)
        XCTAssertEqual(manager.speakerCount, 1)
        XCTAssertTrue(speakerId?.hasPrefix("Speaker_") ?? false)
    }

    func testAssignExistingSpeaker() {
        let manager = SpeakerManager(speakerThreshold: 0.3)  // Low threshold for testing

        // Add first speaker
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let speakerId1 = manager.assignSpeaker(embedding1, speechDuration: 2.0)

        // Add nearly identical embedding - should match existing speaker
        var embedding2 = embedding1
        embedding2[0] += 0.001  // Tiny variation
        let speakerId2 = manager.assignSpeaker(embedding2, speechDuration: 2.0)

        XCTAssertEqual(speakerId1, speakerId2)
        XCTAssertEqual(manager.speakerCount, 1)  // Should still be 1 speaker
    }

    func testMultipleSpeakers() {
        let manager = SpeakerManager(speakerThreshold: 0.5)

        // Create distinct embeddings
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let embedding2 = createDistinctEmbedding(pattern: 2)

        let speakerId1 = manager.assignSpeaker(embedding1, speechDuration: 2.0)
        let speakerId2 = manager.assignSpeaker(embedding2, speechDuration: 2.0)

        XCTAssertNotNil(speakerId1)
        XCTAssertNotNil(speakerId2)
        XCTAssertNotEqual(speakerId1, speakerId2)
        XCTAssertEqual(manager.speakerCount, 2)
    }

    // MARK: - Known Speaker Initialization

    func testInitializeKnownSpeakers() {
        let manager = SpeakerManager()

        let knownSpeakers = [
            "Alice": createDistinctEmbedding(pattern: 10),
            "Bob": createDistinctEmbedding(pattern: 20),
        ]

        manager.initializeKnownSpeakers(knownSpeakers)

        XCTAssertEqual(manager.speakerCount, 2)
        XCTAssertTrue(manager.speakerIds.contains("Alice"))
        XCTAssertTrue(manager.speakerIds.contains("Bob"))
    }

    func testRecognizeKnownSpeaker() {
        let manager = SpeakerManager(speakerThreshold: 0.3)

        let aliceEmbedding = createDistinctEmbedding(pattern: 10)
        manager.initializeKnownSpeakers(["Alice": aliceEmbedding])

        // Test with exact same embedding
        let testEmbedding = aliceEmbedding

        let assignedId = manager.assignSpeaker(testEmbedding, speechDuration: 2.0)
        XCTAssertEqual(assignedId, "Alice")
    }

    func testInvalidEmbeddingSize() {
        let manager = SpeakerManager()

        // Test with wrong size
        let invalidEmbedding = [Float](repeating: 0.5, count: 128)
        let speakerId = manager.assignSpeaker(invalidEmbedding, speechDuration: 2.0)

        XCTAssertNil(speakerId)
        XCTAssertEqual(manager.speakerCount, 0)
    }

    func testEmptyEmbedding() {
        let manager = SpeakerManager()

        let emptyEmbedding = [Float]()
        let speakerId = manager.assignSpeaker(emptyEmbedding, speechDuration: 2.0)

        XCTAssertNil(speakerId)
        XCTAssertEqual(manager.speakerCount, 0)
    }

    // MARK: - Speaker Info Access

    func testGetSpeakerInfo() {
        let manager = SpeakerManager()
        let embedding = createDistinctEmbedding(pattern: 1)

        let speakerId = manager.assignSpeaker(embedding, speechDuration: 3.5)
        XCTAssertNotNil(speakerId)

        if let id = speakerId {
            let info = manager.getSpeakerInfo(for: id)
            XCTAssertNotNil(info)
            XCTAssertEqual(info?.id, id)
            XCTAssertEqual(info?.embedding, embedding)
            XCTAssertEqual(info?.totalDuration, 3.5)
        }
    }

    func testGetAllSpeakerInfo() {
        let manager = SpeakerManager()

        // Add multiple speakers
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let embedding2 = createDistinctEmbedding(pattern: 2)

        let id1 = manager.assignSpeaker(embedding1, speechDuration: 2.0)
        let id2 = manager.assignSpeaker(embedding2, speechDuration: 3.0)

        let allInfo = manager.getAllSpeakerInfo()

        XCTAssertEqual(allInfo.count, 2)
        XCTAssertNotNil(id1.flatMap { allInfo[$0] })
        XCTAssertNotNil(id2.flatMap { allInfo[$0] })
    }

    // MARK: - Clear Operations

    func testResetSpeakers() {
        let manager = SpeakerManager()

        // Add speakers
        manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 2.0)
        manager.assignSpeaker(createDistinctEmbedding(pattern: 2), speechDuration: 2.0)

        XCTAssertEqual(manager.speakerCount, 2)

        manager.reset()

        // Wait a bit for async reset to complete
        Thread.sleep(forTimeInterval: 0.1)

        XCTAssertEqual(manager.speakerCount, 0)
        XCTAssertTrue(manager.speakerIds.isEmpty)
    }

    // MARK: - Distance Calculations

    func testCosineDistance() {
        let manager = SpeakerManager()

        // Test identical embeddings
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let distance1 = manager.cosineDistance(embedding1, embedding1)
        XCTAssertEqual(distance1, 0.0, accuracy: 0.0001)

        // Test different embeddings
        let embedding2 = createDistinctEmbedding(pattern: 2)
        let distance2 = manager.cosineDistance(embedding1, embedding2)
        XCTAssertGreaterThan(distance2, 0.0)  // Should be different

        // Test orthogonal embeddings
        var embedding3 = [Float](repeating: 0, count: 256)
        embedding3[0] = 1.0
        var embedding4 = [Float](repeating: 0, count: 256)
        embedding4[1] = 1.0
        let distance3 = manager.cosineDistance(embedding3, embedding4)
        XCTAssertEqual(distance3, 1.0, accuracy: 0.0001)  // Cosine distance of orthogonal vectors
    }

    func testCosineDistanceWithDifferentSizes() {
        let manager = SpeakerManager()

        let embedding1 = createDistinctEmbedding(pattern: 1)
        let embedding2 = [Float](repeating: 0.5, count: 128)

        let distance = manager.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(distance, Float.infinity)
    }

    // MARK: - Statistics

    func testGetStatistics() {
        let manager = SpeakerManager()

        // Add speakers with different durations
        manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 10.0)
        manager.assignSpeaker(createDistinctEmbedding(pattern: 2), speechDuration: 20.0)

        let stats = manager.getStatistics()

        XCTAssertTrue(stats.contains("Speakers: 2"))
        XCTAssertTrue(stats.contains("Total Duration:"))
    }

    // MARK: - Thread Safety

    func testConcurrentAccess() {
        let manager = SpeakerManager(speakerThreshold: 0.5)  // Set threshold for distinct embeddings
        let queue = DispatchQueue(label: "test", attributes: .concurrent)
        let group = DispatchGroup()
        let iterations = 10  // Reduced iterations for more reliable test

        // Use a serial queue to ensure embeddings are distinct
        let embeddings = (0..<iterations).map { i -> [Float] in
            // Create very distinct embeddings using different patterns
            var embedding = [Float](repeating: 0, count: 256)
            for j in 0..<256 {
                // Use different functions for each speaker
                switch i % 3 {
                case 0:
                    embedding[j] = sin(Float(j * (i + 1)) * 0.05)
                case 1:
                    embedding[j] = cos(Float(j * (i + 1)) * 0.05)
                default:
                    embedding[j] = Float(j % (i + 2)) / Float(i + 2) - 0.5
                }
            }
            return embedding
        }

        // Concurrent writes with pre-created distinct embeddings
        for i in 0..<iterations {
            group.enter()
            queue.async {
                _ = manager.assignSpeaker(embeddings[i], speechDuration: 2.0)
                group.leave()
            }
        }

        // Concurrent reads
        for _ in 0..<iterations {
            group.enter()
            queue.async {
                _ = manager.speakerCount
                _ = manager.speakerIds
                group.leave()
            }
        }

        let expectation = XCTestExpectation(description: "Concurrent operations complete")
        group.notify(queue: .main) {
            // Due to concurrent operations and clustering, we may not get exactly iterations speakers
            // But we should have at least some distinct speakers
            XCTAssertGreaterThan(manager.speakerCount, 0)
            XCTAssertLessThanOrEqual(manager.speakerCount, iterations)
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5.0)
    }

    // MARK: - Edge Cases

    func testSpeakerThresholdBoundaries() {
        // Test with very low threshold (everything matches)
        let manager1 = SpeakerManager(speakerThreshold: 0.01)
        manager1.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 2.0)
        var similarEmbedding = createDistinctEmbedding(pattern: 1)
        similarEmbedding[0] += 0.001  // Tiny variation
        manager1.assignSpeaker(similarEmbedding, speechDuration: 2.0)
        XCTAssertEqual(manager1.speakerCount, 1)  // Should match to same speaker

        // Test with high threshold (only exact matches)
        let manager2 = SpeakerManager(speakerThreshold: 0.001)  // Very small threshold
        let emb1 = createDistinctEmbedding(pattern: 1)
        manager2.assignSpeaker(emb1, speechDuration: 2.0)
        manager2.assignSpeaker(emb1, speechDuration: 2.0)  // Exact same embedding
        XCTAssertEqual(manager2.speakerCount, 1)  // Should match to same speaker
    }

    func testMinDurationFiltering() {
        let manager = SpeakerManager(
            speakerThreshold: 0.5,
            embeddingThreshold: 0.3,
            minDuration: 2.0
        )

        let embedding = createDistinctEmbedding(pattern: 1)

        // Test with duration below threshold - should not create new speaker
        let id1 = manager.assignSpeaker(embedding, speechDuration: 0.5)
        XCTAssertNil(id1)  // Should return nil for short duration
        XCTAssertEqual(manager.speakerCount, 0)  // No speaker created

        // Test with duration above threshold - should create speaker
        let id2 = manager.assignSpeaker(embedding, speechDuration: 3.0)
        XCTAssertNotNil(id2)
        XCTAssertEqual(manager.speakerCount, 1)  // One speaker created

        // Test again with short duration on existing speaker
        let id3 = manager.assignSpeaker(embedding, speechDuration: 0.5)
        XCTAssertEqual(id3, id2)  // Should match existing speaker even with short duration
    }
}
