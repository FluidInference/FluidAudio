import XCTest

@testable import FluidAudio

/// Tests for Speaker class functionality
final class SpeakerTests: XCTestCase {

    // Helper to create distinct embeddings
    private func createDistinctEmbedding(pattern: Int) -> [Float] {
        var embedding = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            embedding[i] = sin(Float(i + pattern * 100) * 0.1)
        }
        return embedding
    }

    // MARK: - Basic Speaker Tests

    func testSpeakerInitialization() {
        let embedding = createDistinctEmbedding(pattern: 1)
        let speaker = Speaker(
            id: "test1",
            name: "Alice",
            currentEmbedding: embedding,
            duration: 5.0
        )

        XCTAssertEqual(speaker.id, "test1")
        XCTAssertEqual(speaker.name, "Alice")
        let expectedEmbedding = VDSPOperations.l2Normalize(embedding)
        for (value, expected) in zip(speaker.currentEmbedding, expectedEmbedding) {
            XCTAssertEqual(value, expected, accuracy: 0.0001)
        }
        XCTAssertEqual(speaker.duration, 5.0)
        XCTAssertEqual(speaker.updateCount, 1)
        XCTAssertTrue(speaker.rawEmbeddings.isEmpty)
    }

    func testSpeakerDefaultName() {
        let embedding = createDistinctEmbedding(pattern: 1)
        let speaker = Speaker(
            id: "1",
            currentEmbedding: embedding
        )

        // Default name should be the ID
        XCTAssertEqual(speaker.name, "1")
    }

    // MARK: - Speaker Enrollment Workflow

    func testSpeakerEnrollmentWithName() {
        // Simulate the "My name is Alice" enrollment scenario
        let manager = SpeakerManager()
        let aliceEmbedding = createDistinctEmbedding(pattern: 10)

        // Step 1: User speaks and we get their embedding
        let speaker = manager.assignSpeaker(aliceEmbedding, speechDuration: 3.0)
        XCTAssertNotNil(speaker)

        // Step 2: Transcription detects "My name is Alice"
        // Step 3: Update the speaker's name from default to actual name
        speaker?.name = "Alice"

        XCTAssertEqual(speaker?.name, "Alice")
        XCTAssertNotEqual(speaker?.name, speaker?.id)  // Name should be different from ID

        // Step 4: Future audio from same speaker should be identified as "Alice"
        let futureSpeaker = manager.assignSpeaker(aliceEmbedding, speechDuration: 2.0)
        XCTAssertEqual(futureSpeaker?.id, speaker?.id)
        XCTAssertEqual(futureSpeaker?.name, "Alice")
    }

    // MARK: - Main Embedding Update Tests

    func testUpdateMainEmbedding() {
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let speaker = Speaker(id: "test", currentEmbedding: embedding1)

        let embedding2 = createDistinctEmbedding(pattern: 2)
        speaker.updateMainEmbedding(
            duration: 3.0,
            embedding: embedding2,
            segmentId: UUID(),
            alpha: 0.9
        )

        // Check that embedding was updated (not equal to either original)
        XCTAssertNotEqual(speaker.currentEmbedding, embedding1)
        XCTAssertNotEqual(speaker.currentEmbedding, embedding2)

        // Check that raw embedding was added
        XCTAssertEqual(speaker.rawEmbeddings.count, 1)
        XCTAssertEqual(speaker.updateCount, 2)
        XCTAssertEqual(speaker.duration, 3.0)
    }

    func testUpdateMainEmbeddingWithAlpha() {
        let embedding1 = [Float](repeating: 1.0, count: 256)
        let speaker = Speaker(id: "test", currentEmbedding: embedding1)

        // Use a valid embedding with non-zero magnitude
        let embedding2 = [Float](repeating: 0.5, count: 256)
        speaker.updateMainEmbedding(
            duration: 3.0,
            embedding: embedding2,
            segmentId: UUID(),
            alpha: 0.8  // 80% old, 20% new
        )

        // The speaker stores embeddings in L2-normalized form. With a single raw embedding,
        // the recalculated main embedding should equal the normalized segment embedding.
        let expectedEmbedding = VDSPOperations.l2Normalize(embedding2)
        for (value, expected) in zip(speaker.currentEmbedding, expectedEmbedding) {
            XCTAssertEqual(value, expected, accuracy: 0.001)
        }

        // Verify that the raw embedding was added
        XCTAssertEqual(speaker.rawEmbeddings.count, 1)
        XCTAssertEqual(speaker.updateCount, 2)
    }

    func testUpdateMainEmbeddingLowMagnitude() {
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let speaker = Speaker(id: "test", currentEmbedding: embedding1)

        // Create an embedding with magnitude below 0.1 threshold
        let lowMagnitudeEmbedding = [Float](repeating: 0.001, count: 256)  // Very low magnitude

        // Update with low magnitude embedding - should not update
        speaker.updateMainEmbedding(
            duration: 3.0,
            embedding: lowMagnitudeEmbedding,
            segmentId: UUID(),
            alpha: 0.9
        )

        // Embedding should not have been updated (magnitude too low)
        let expectedEmbedding = VDSPOperations.l2Normalize(embedding1)
        for (value, expected) in zip(speaker.currentEmbedding, expectedEmbedding) {
            XCTAssertEqual(value, expected, accuracy: 0.0001)
        }
        XCTAssertEqual(speaker.rawEmbeddings.count, 0)  // No raw embedding added
        XCTAssertEqual(speaker.updateCount, 1)  // No update
        XCTAssertEqual(speaker.duration, 0.0)  // Duration NOT updated due to early return
    }

    // MARK: - Raw Embedding Management Tests

    func testAddRawEmbedding() {
        let speaker = Speaker(id: "test", currentEmbedding: createDistinctEmbedding(pattern: 1))

        let rawEmb1 = RawEmbedding(embedding: createDistinctEmbedding(pattern: 2))
        let rawEmb2 = RawEmbedding(embedding: createDistinctEmbedding(pattern: 3))

        speaker.addRawEmbedding(rawEmb1)
        speaker.addRawEmbedding(rawEmb2)

        XCTAssertEqual(speaker.rawEmbeddings.count, 2)
    }

    func testRawEmbeddingFIFO() {
        let speaker = Speaker(id: "test", currentEmbedding: createDistinctEmbedding(pattern: 1))

        // Add 60 raw embeddings (more than the 50 limit)
        for i in 0..<60 {
            let rawEmb = RawEmbedding(embedding: createDistinctEmbedding(pattern: i))
            speaker.addRawEmbedding(rawEmb)
        }

        // Should only keep the last 50
        XCTAssertEqual(speaker.rawEmbeddings.count, 50)

        // First embedding should be from pattern 10 (0-9 were removed)
        let firstEmbedding = speaker.rawEmbeddings.first?.embedding
        let expectedFirst = VDSPOperations.l2Normalize(createDistinctEmbedding(pattern: 10))
        if let firstValue = firstEmbedding?[0] {
            XCTAssertEqual(firstValue, expectedFirst[0], accuracy: 0.001)
        }
    }

    func testRemoveRawEmbedding() {
        let speaker = Speaker(id: "test", currentEmbedding: createDistinctEmbedding(pattern: 1))

        let rawEmb1 = RawEmbedding(embedding: createDistinctEmbedding(pattern: 2))
        let rawEmb2 = RawEmbedding(embedding: createDistinctEmbedding(pattern: 3))

        speaker.addRawEmbedding(rawEmb1)
        speaker.addRawEmbedding(rawEmb2)

        XCTAssertEqual(speaker.rawEmbeddings.count, 2)

        // Remove the first one
        let removed = speaker.removeRawEmbedding(segmentId: rawEmb1.segmentId)
        XCTAssertNotNil(removed)
        XCTAssertEqual(speaker.rawEmbeddings.count, 1)

        // Try to remove non-existent
        let notRemoved = speaker.removeRawEmbedding(segmentId: UUID())
        XCTAssertNil(notRemoved)
        XCTAssertEqual(speaker.rawEmbeddings.count, 1)
    }

    // MARK: - Recalculate Main Embedding Tests

    func testRecalculateMainEmbedding() {
        let speaker = Speaker(id: "test", currentEmbedding: [Float](repeating: 0, count: 256))

        // Add some raw embeddings
        let emb1 = [Float](repeating: 1.0, count: 256)
        let emb2 = [Float](repeating: 2.0, count: 256)
        let emb3 = [Float](repeating: 3.0, count: 256)

        speaker.addRawEmbedding(RawEmbedding(embedding: emb1))
        speaker.addRawEmbedding(RawEmbedding(embedding: emb2))
        speaker.addRawEmbedding(RawEmbedding(embedding: emb3))

        speaker.recalculateMainEmbedding()

        // Raw embeddings are stored normalized; recalculating should keep the unit-normalized vector.
        let expectedEmbedding = VDSPOperations.l2Normalize([Float](repeating: 1.0, count: 256))
        for (value, expected) in zip(speaker.currentEmbedding, expectedEmbedding) {
            XCTAssertEqual(value, expected, accuracy: 0.0001)
        }
    }

    func testRecalculateMainEmbeddingEmpty() {
        let original = createDistinctEmbedding(pattern: 1)
        let speaker = Speaker(id: "test", currentEmbedding: original)

        // No raw embeddings
        speaker.recalculateMainEmbedding()

        // Should keep the previously normalized embedding
        let expectedEmbedding = VDSPOperations.l2Normalize(original)
        for (value, expected) in zip(speaker.currentEmbedding, expectedEmbedding) {
            XCTAssertEqual(value, expected, accuracy: 0.0001)
        }
    }

    // MARK: - Speaker Merging Tests

    func testMergeSpeakers() {
        let speaker1 = Speaker(
            id: "speaker1",
            name: "Alice",
            currentEmbedding: createDistinctEmbedding(pattern: 1),
            duration: 10.0
        )
        speaker1.updateCount = 5

        let speaker2 = Speaker(
            id: "speaker2",
            name: "Bob",
            currentEmbedding: createDistinctEmbedding(pattern: 2),
            duration: 20.0
        )
        speaker2.updateCount = 3

        // Add raw embeddings to both
        speaker1.addRawEmbedding(RawEmbedding(embedding: createDistinctEmbedding(pattern: 3)))
        speaker2.addRawEmbedding(RawEmbedding(embedding: createDistinctEmbedding(pattern: 4)))
        speaker2.addRawEmbedding(RawEmbedding(embedding: createDistinctEmbedding(pattern: 5)))

        speaker1.mergeWith(speaker2)

        // Check merged properties
        XCTAssertEqual(speaker1.duration, 30.0)  // 10 + 20
        XCTAssertEqual(speaker1.updateCount, 8)  // 5 + 3
        XCTAssertEqual(speaker1.rawEmbeddings.count, 3)  // 1 + 2
        XCTAssertEqual(speaker1.name, "Alice")  // Keeps original name by default
    }

    func testMergeSpeakersWithCustomName() {
        let speaker1 = Speaker(id: "speaker1", name: "Alice", currentEmbedding: createDistinctEmbedding(pattern: 1))
        let speaker2 = Speaker(id: "speaker2", name: "Bob", currentEmbedding: createDistinctEmbedding(pattern: 2))

        speaker1.mergeWith(speaker2, keepName: "Charlie")

        XCTAssertEqual(speaker1.name, "Charlie")
    }

    // MARK: - SendableSpeaker Tests

    func testToSendable() {
        let speaker = Speaker(
            id: "test1",
            name: "Alice",
            currentEmbedding: createDistinctEmbedding(pattern: 1),
            duration: 15.0
        )
        speaker.updateCount = 3

        let sendable = speaker.toSendable()

        // SendableSpeaker has Int id, Speaker has String id
        XCTAssertEqual(sendable.name, speaker.name)
        XCTAssertEqual(sendable.duration, speaker.duration)
        XCTAssertEqual(sendable.mainEmbedding, speaker.currentEmbedding)
        XCTAssertEqual(sendable.label, speaker.name)  // Since name is not empty
    }

    func testSendableSpeakerLabel() {
        // Test with name
        let speaker1 = SendableSpeaker(
            id: 1,
            name: "Alice",
            duration: 10.0,
            mainEmbedding: createDistinctEmbedding(pattern: 1),
            createdAt: Date(),
            updatedAt: Date()
        )
        XCTAssertEqual(speaker1.label, "Alice")

        // Test without name
        let speaker2 = SendableSpeaker(
            id: 2,
            name: "",
            duration: 10.0,
            mainEmbedding: createDistinctEmbedding(pattern: 2),
            createdAt: Date(),
            updatedAt: Date()
        )
        XCTAssertEqual(speaker2.label, "Speaker #2")
    }

    // MARK: - Codable Tests

    func testSpeakerCodable() throws {
        let speaker = Speaker(
            id: "test1",
            name: "Alice",
            currentEmbedding: createDistinctEmbedding(pattern: 1),
            duration: 15.0
        )
        speaker.addRawEmbedding(RawEmbedding(embedding: createDistinctEmbedding(pattern: 2)))

        // Encode
        let encoder = JSONEncoder()
        let data = try encoder.encode(speaker)

        // Decode
        let decoder = JSONDecoder()
        let decodedSpeaker = try decoder.decode(Speaker.self, from: data)

        XCTAssertEqual(decodedSpeaker.id, speaker.id)
        XCTAssertEqual(decodedSpeaker.name, speaker.name)
        XCTAssertEqual(decodedSpeaker.currentEmbedding, speaker.currentEmbedding)
        XCTAssertEqual(decodedSpeaker.duration, speaker.duration)
        XCTAssertEqual(decodedSpeaker.rawEmbeddings.count, 1)
    }

    // MARK: - Hashable Tests

    func testSpeakerHashable() {
        let speaker1 = Speaker(id: "test1", currentEmbedding: createDistinctEmbedding(pattern: 1))
        let speaker2 = Speaker(id: "test2", currentEmbedding: createDistinctEmbedding(pattern: 2))
        let speaker3 = Speaker(id: "test1", currentEmbedding: createDistinctEmbedding(pattern: 3))

        var set = Set<Speaker>()
        set.insert(speaker1)
        set.insert(speaker2)
        set.insert(speaker3)  // Same ID as speaker1

        // Should only have 2 speakers (speaker3 has same ID as speaker1)
        XCTAssertEqual(set.count, 2)
    }

    func testSpeakerEquatable() {
        let speaker1 = Speaker(id: "test1", currentEmbedding: createDistinctEmbedding(pattern: 1))
        let speaker2 = Speaker(id: "test2", currentEmbedding: createDistinctEmbedding(pattern: 2))
        let speaker3 = Speaker(id: "test1", currentEmbedding: createDistinctEmbedding(pattern: 3))

        XCTAssertEqual(speaker1, speaker3)  // Same ID
        XCTAssertNotEqual(speaker1, speaker2)  // Different ID
    }
}
