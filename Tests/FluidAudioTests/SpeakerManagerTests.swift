import XCTest

@testable import FluidAudio

/// Tests for SpeakerManager functionality
@available(macOS 13.0, iOS 16.0, *)
final class SpeakerManagerTests: XCTestCase {

    // MARK: - Basic Operations

    func testInitialization() {
        let manager = SpeakerManager()
        XCTAssertEqual(manager.speakerCount, 0)
        XCTAssertTrue(manager.speakerIds.isEmpty)
    }

    func testAssignNewSpeaker() {
        let manager = SpeakerManager()
        let embedding = [Float](repeating: 0.5, count: 256)

        let speakerId = manager.assignSpeaker(embedding, speechDuration: 2.0)

        XCTAssertNotNil(speakerId)
        XCTAssertEqual(manager.speakerCount, 1)
        XCTAssertTrue(speakerId?.hasPrefix("Speaker_") ?? false)
    }

    func testAssignExistingSpeaker() {
        let manager = SpeakerManager(speakerThreshold: 0.3)  // Low threshold for testing

        // Add first speaker
        let embedding1 = [Float](repeating: 0.5, count: 256)
        let speakerId1 = manager.assignSpeaker(embedding1, speechDuration: 2.0)

        // Add nearly identical embedding - should match existing speaker
        var embedding2 = embedding1
        embedding2[0] = 0.51  // Tiny variation
        let speakerId2 = manager.assignSpeaker(embedding2, speechDuration: 2.0)

        XCTAssertEqual(speakerId1, speakerId2)
        XCTAssertEqual(manager.speakerCount, 1)  // Should still be 1 speaker
    }

    func testMultipleSpeakers() {
        let manager = SpeakerManager(speakerThreshold: 0.5)

        // Create distinct embeddings
        let embedding1 = [Float](repeating: 0.1, count: 256)
        let embedding2 = [Float](repeating: 0.9, count: 256)

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
            "Alice": [Float](repeating: 0.3, count: 256),
            "Bob": [Float](repeating: 0.7, count: 256),
        ]

        manager.initializeKnownSpeakers(knownSpeakers)

        XCTAssertEqual(manager.speakerCount, 2)
        XCTAssertTrue(manager.speakerIds.contains("Alice"))
        XCTAssertTrue(manager.speakerIds.contains("Bob"))
    }

    func testRecognizeKnownSpeaker() {
        let manager = SpeakerManager(speakerThreshold: 0.3)

        let aliceEmbedding = [Float](repeating: 0.3, count: 256)
        manager.initializeKnownSpeakers(["Alice": aliceEmbedding])

        // Test with slightly modified embedding
        var testEmbedding = aliceEmbedding
        testEmbedding[0] = 0.31

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
        let embedding = [Float](repeating: 0.5, count: 256)

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
        let embedding1 = [Float](repeating: 0.3, count: 256)
        let embedding2 = [Float](repeating: 0.7, count: 256)

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
        manager.assignSpeaker([Float](repeating: 0.3, count: 256), speechDuration: 2.0)
        manager.assignSpeaker([Float](repeating: 0.7, count: 256), speechDuration: 2.0)

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
        let embedding1 = [Float](repeating: 0.5, count: 256)
        let distance1 = manager.cosineDistance(embedding1, embedding1)
        XCTAssertEqual(distance1, 0.0, accuracy: 0.0001)

        // Test opposite embeddings
        let embedding2 = [Float](repeating: -0.5, count: 256)
        let distance2 = manager.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(distance2, 2.0, accuracy: 0.0001)  // Cosine distance of opposite vectors

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

        let embedding1 = [Float](repeating: 0.5, count: 256)
        let embedding2 = [Float](repeating: 0.5, count: 128)

        let distance = manager.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(distance, Float.infinity)
    }

    // MARK: - Statistics

    func testGetStatistics() {
        let manager = SpeakerManager()

        // Add speakers with different durations
        manager.assignSpeaker([Float](repeating: 0.3, count: 256), speechDuration: 10.0)
        manager.assignSpeaker([Float](repeating: 0.7, count: 256), speechDuration: 20.0)

        let stats = manager.getStatistics()

        XCTAssertTrue(stats.contains("Speakers: 2"))
        XCTAssertTrue(stats.contains("Total Duration:"))
    }

    // MARK: - Thread Safety

    func testConcurrentAccess() {
        let manager = SpeakerManager()
        let queue = DispatchQueue(label: "test", attributes: .concurrent)
        let group = DispatchGroup()
        let iterations = 100

        // Concurrent writes
        for i in 0..<iterations {
            group.enter()
            queue.async {
                let embedding = [Float](repeating: Float(i) / Float(iterations), count: 256)
                _ = manager.assignSpeaker(embedding, speechDuration: 1.0)
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
            // All speakers should be distinct due to different embeddings
            XCTAssertEqual(manager.speakerCount, iterations)
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5.0)
    }

    // MARK: - Edge Cases

    func testSpeakerThresholdBoundaries() {
        // Test with very low threshold (everything matches)
        let manager1 = SpeakerManager(speakerThreshold: 0.01)
        manager1.assignSpeaker([Float](repeating: 0.3, count: 256), speechDuration: 1.0)
        manager1.assignSpeaker([Float](repeating: 0.7, count: 256), speechDuration: 1.0)
        XCTAssertEqual(manager1.speakerCount, 1)  // Should match to same speaker

        // Test with very high threshold (nothing matches)
        let manager2 = SpeakerManager(speakerThreshold: 1.99)
        manager2.assignSpeaker([Float](repeating: 0.5, count: 256), speechDuration: 1.0)
        var similarEmbedding = [Float](repeating: 0.5, count: 256)
        similarEmbedding[0] = 0.501
        manager2.assignSpeaker(similarEmbedding, speechDuration: 1.0)
        XCTAssertEqual(manager2.speakerCount, 2)  // Should create different speakers
    }

    func testMinDurationFiltering() {
        let manager = SpeakerManager(
            speakerThreshold: 0.5,
            embeddingThreshold: 0.3,
            minDuration: 2.0
        )

        let embedding = [Float](repeating: 0.5, count: 256)

        // Test with duration below threshold
        let id1 = manager.assignSpeaker(embedding, speechDuration: 0.5)
        XCTAssertNotNil(id1)  // Still assigns but may not update

        // Test with duration above threshold
        let id2 = manager.assignSpeaker(embedding, speechDuration: 3.0)
        XCTAssertNotNil(id2)
        XCTAssertEqual(id1, id2)  // Should be same speaker
    }
}
