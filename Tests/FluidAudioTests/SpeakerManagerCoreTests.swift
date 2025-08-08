import XCTest

@testable import FluidAudio

/// Core tests for SpeakerManager functionality
@available(macOS 13.0, iOS 16.0, *)
final class SpeakerManagerCoreTests: XCTestCase {

    // Helper to create distinct embeddings
    private func createDistinctEmbedding(pattern: Int) -> [Float] {
        var embedding = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            // Create unique pattern for each embedding
            embedding[i] = sin(Float(i + pattern * 100) * 0.1)
        }
        return embedding
    }

    func testBasicInitialization() {
        let manager = SpeakerManager()
        XCTAssertEqual(manager.speakerCount, 0)
        XCTAssertTrue(manager.speakerIds.isEmpty)
    }

    func testSingleSpeakerAssignment() {
        let manager = SpeakerManager()
        let embedding = createDistinctEmbedding(pattern: 1)

        let speakerId = manager.assignSpeaker(embedding, speechDuration: 2.0)

        XCTAssertNotNil(speakerId)
        XCTAssertEqual(manager.speakerCount, 1)
        XCTAssertTrue(speakerId?.hasPrefix("Speaker_") ?? false)
    }

    func testMultipleDifferentSpeakers() {
        let manager = SpeakerManager()

        // Create very different embeddings
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let embedding2 = createDistinctEmbedding(pattern: 2)

        let id1 = manager.assignSpeaker(embedding1, speechDuration: 2.0)
        let id2 = manager.assignSpeaker(embedding2, speechDuration: 2.0)

        XCTAssertNotNil(id1)
        XCTAssertNotNil(id2)
        XCTAssertNotEqual(id1, id2, "Different embeddings should get different IDs")
        XCTAssertEqual(manager.speakerCount, 2)
    }

    func testSameSpeakerRecognition() {
        let manager = SpeakerManager(speakerThreshold: 0.3)  // Lower threshold for testing

        let embedding = createDistinctEmbedding(pattern: 1)

        // Assign first time
        let id1 = manager.assignSpeaker(embedding, speechDuration: 2.0)

        // Assign again with exact same embedding
        let id2 = manager.assignSpeaker(embedding, speechDuration: 2.0)

        XCTAssertEqual(id1, id2, "Same embedding should get same ID")
        XCTAssertEqual(manager.speakerCount, 1)
    }

    func testKnownSpeakerInitialization() {
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

    func testKnownSpeakerRecognition() {
        let manager = SpeakerManager(speakerThreshold: 0.3)

        let aliceEmbedding = createDistinctEmbedding(pattern: 10)
        manager.initializeKnownSpeakers(["Alice": aliceEmbedding])

        // Test with exact same embedding
        let assignedId = manager.assignSpeaker(aliceEmbedding, speechDuration: 2.0)
        XCTAssertEqual(assignedId, "Alice", "Should recognize known speaker")
    }

    func testGetSpeakerInfo() {
        let manager = SpeakerManager()
        let embedding = createDistinctEmbedding(pattern: 1)

        let speakerId = manager.assignSpeaker(embedding, speechDuration: 3.5)

        if let id = speakerId {
            let info = manager.getSpeakerInfo(for: id)
            XCTAssertNotNil(info)
            XCTAssertEqual(info?.id, id)
            XCTAssertEqual(info?.totalDuration, 3.5)
        }
    }

    func testGetAllSpeakerInfo() {
        let manager = SpeakerManager()

        let id1 = manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 2.0)
        let id2 = manager.assignSpeaker(createDistinctEmbedding(pattern: 2), speechDuration: 3.0)

        let allInfo = manager.getAllSpeakerInfo()

        XCTAssertEqual(allInfo.count, 2)
        XCTAssertNotNil(id1.flatMap { allInfo[$0] })
        XCTAssertNotNil(id2.flatMap { allInfo[$0] })
    }

    func testResetFunctionality() {
        let manager = SpeakerManager()

        // Add speakers
        manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 2.0)
        manager.assignSpeaker(createDistinctEmbedding(pattern: 2), speechDuration: 2.0)

        XCTAssertEqual(manager.speakerCount, 2)

        manager.reset()

        // Wait for async operation
        Thread.sleep(forTimeInterval: 0.1)

        XCTAssertEqual(manager.speakerCount, 0)
        XCTAssertTrue(manager.speakerIds.isEmpty)
    }

    func testInvalidEmbedding() {
        let manager = SpeakerManager()

        // Wrong size
        let wrongSize = [Float](repeating: 0.5, count: 128)
        let id1 = manager.assignSpeaker(wrongSize, speechDuration: 2.0)
        XCTAssertNil(id1)

        // Empty
        let empty = [Float]()
        let id2 = manager.assignSpeaker(empty, speechDuration: 2.0)
        XCTAssertNil(id2)

        XCTAssertEqual(manager.speakerCount, 0)
    }

    func testCosineDistanceCalculation() {
        let manager = SpeakerManager()

        let embedding1 = createDistinctEmbedding(pattern: 1)

        // Distance to self should be 0
        let selfDistance = manager.cosineDistance(embedding1, embedding1)
        XCTAssertEqual(selfDistance, 0.0, accuracy: 0.001)

        // Distance to different embedding should be > 0
        let embedding2 = createDistinctEmbedding(pattern: 2)
        let distance = manager.cosineDistance(embedding1, embedding2)
        XCTAssertGreaterThan(distance, 0.0)
    }

    func testStatisticsGeneration() {
        let manager = SpeakerManager()

        manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 10.0)
        manager.assignSpeaker(createDistinctEmbedding(pattern: 2), speechDuration: 20.0)

        let stats = manager.getStatistics()

        // Check that stats contain expected information
        XCTAssertTrue(stats.contains("2"), "Should show 2 speakers")
        XCTAssertTrue(stats.contains("30"), "Should show total duration of 30s")
    }
}
