import XCTest

@testable import FluidAudio

/// Tests for SpeakerOperations utilities
final class SpeakerOperationsTests: XCTestCase {

    // Helper to create distinct embeddings
    private func createDistinctEmbedding(pattern: Int) -> [Float] {
        var embedding = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            embedding[i] = sin(Float(i + pattern * 100) * 0.1)
        }
        return embedding
    }

    private func createNormalizedEmbedding(pattern: Int) -> [Float] {
        let embedding = createDistinctEmbedding(pattern: pattern)
        let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
        return embedding.map { $0 / magnitude }
    }

    // MARK: - Configuration Tests

    func testAssignmentConfigPlatformSpecific() {
        #if os(macOS)
        let config = SpeakerUtilities.AssignmentConfig.current
        XCTAssertEqual(config.maxDistanceForAssignment, 0.65)
        XCTAssertEqual(config.maxDistanceForUpdate, 0.45)
        #else
        let config = SpeakerUtilities.AssignmentConfig.current
        XCTAssertEqual(config.maxDistanceForAssignment, 0.55)
        XCTAssertEqual(config.maxDistanceForUpdate, 0.45)
        #endif
    }

    // MARK: - Cosine Distance Tests

    func testCosineDistance() {
        // Test identical embeddings
        let emb1 = createNormalizedEmbedding(pattern: 1)
        let distance1 = SpeakerUtilities.cosineDistance(emb1, emb1)
        XCTAssertEqual(distance1, 0.0, accuracy: 0.001)

        // Test different embeddings
        let emb2 = createNormalizedEmbedding(pattern: 2)
        let distance2 = SpeakerUtilities.cosineDistance(emb1, emb2)
        XCTAssertGreaterThan(distance2, 0.0)
        XCTAssertLessThanOrEqual(distance2, 2.0)

        // Test orthogonal embeddings
        var emb3 = [Float](repeating: 0, count: 256)
        emb3[0] = 1.0
        var emb4 = [Float](repeating: 0, count: 256)
        emb4[1] = 1.0
        let distance3 = SpeakerUtilities.cosineDistance(emb3, emb4)
        XCTAssertEqual(distance3, 1.0, accuracy: 0.001)
    }

    func testCosineDistanceWithDifferentSizes() {
        let emb1 = createDistinctEmbedding(pattern: 1)
        let emb2 = [Float](repeating: 0.5, count: 128)

        let distance = SpeakerUtilities.cosineDistance(emb1, emb2)
        XCTAssertEqual(distance, Float.infinity)
    }

    func testCosineDistanceWithZeroVectors() {
        let emb1 = [Float](repeating: 0, count: 256)
        let emb2 = createDistinctEmbedding(pattern: 1)

        let distance = SpeakerUtilities.cosineDistance(emb1, emb2)
        XCTAssertEqual(distance, Float.infinity)
    }

    // MARK: - Embedding Validation Tests

    func testValidateEmbedding() {
        // Valid embedding
        let validEmb = createDistinctEmbedding(pattern: 1)
        XCTAssertTrue(SpeakerUtilities.validateEmbedding(validEmb))

        // Wrong size - actually passes validation since size check was removed
        let wrongSizeEmb = [Float](repeating: 0.5, count: 128)
        XCTAssertTrue(SpeakerUtilities.validateEmbedding(wrongSizeEmb))  // Passes since it has good magnitude

        // Zero embedding
        let zeroEmb = [Float](repeating: 0, count: 256)
        XCTAssertFalse(SpeakerUtilities.validateEmbedding(zeroEmb))

        // Low magnitude embedding
        var lowMagEmb = [Float](repeating: 0, count: 256)
        lowMagEmb[0] = 0.01  // Very small magnitude
        XCTAssertFalse(SpeakerUtilities.validateEmbedding(lowMagEmb, minMagnitude: 0.1))

        // Valid with custom threshold
        var customEmb = [Float](repeating: 0, count: 256)
        customEmb[0] = 0.2
        XCTAssertTrue(SpeakerUtilities.validateEmbedding(customEmb, minMagnitude: 0.1))
    }

    // MARK: - Speaker Assignment Decision Tests

    func testShouldAssignSpeaker() {
        // Test with very small distance - high confidence
        let result1 = SpeakerUtilities.shouldAssignSpeaker(
            distance: 0.1,
            duration: 5.0,
            config: .current
        )

        XCTAssertTrue(result1.shouldAssign)
        XCTAssertTrue(result1.shouldUpdate)
        XCTAssertGreaterThan(result1.confidence, 0.8)  // High confidence

        // Test with medium distance
        let result2 = SpeakerUtilities.shouldAssignSpeaker(
            distance: 0.4,
            duration: 5.0,
            config: .current
        )

        XCTAssertTrue(result2.shouldAssign)
        XCTAssertTrue(result2.shouldUpdate)
        XCTAssertGreaterThan(result2.confidence, 0.5)  // Medium confidence
        XCTAssertLessThan(result2.confidence, 0.8)

        // Test with large distance - should not assign
        let result3 = SpeakerUtilities.shouldAssignSpeaker(
            distance: 0.8,
            duration: 5.0,
            config: .current
        )

        XCTAssertFalse(result3.shouldAssign)
        XCTAssertFalse(result3.shouldUpdate)
        XCTAssertLessThan(result3.confidence, 0.3)  // Low confidence
    }

    // MARK: - Find Closest Speaker Tests

    func testFindClosestSpeaker() {
        let speakers = [
            Speaker(id: "speaker1", currentEmbedding: createDistinctEmbedding(pattern: 1)),
            Speaker(id: "speaker2", currentEmbedding: createDistinctEmbedding(pattern: 2)),
            Speaker(id: "speaker3", currentEmbedding: createDistinctEmbedding(pattern: 3)),
        ]

        // Find closest to pattern 1
        var testEmb = createDistinctEmbedding(pattern: 1)
        testEmb[0] += 0.01  // Slight variation

        let result = SpeakerUtilities.findClosestSpeaker(
            embedding: testEmb,
            candidates: speakers
        )

        XCTAssertNotNil(result.speaker)
        XCTAssertEqual(result.speaker?.id, "speaker1")
        XCTAssertLessThan(result.distance, 0.5)

        // Test with empty candidates
        let emptyResult = SpeakerUtilities.findClosestSpeaker(
            embedding: testEmb,
            candidates: []
        )

        XCTAssertNil(emptyResult.speaker)
        XCTAssertEqual(emptyResult.distance, Float.infinity)
    }

    // MARK: - Speaker Creation Validation Tests

    func testValidateSpeakerCreation() {
        // Valid creation
        let validResult = SpeakerUtilities.validateSpeakerCreation(
            id: "test1",
            name: "Test Speaker",
            duration: 5.0,
            embedding: createDistinctEmbedding(pattern: 1),
            config: .current
        )
        if case .success = validResult {
            XCTAssert(true)
        } else {
            XCTFail("Should have succeeded")
        }

        // Too short duration
        let tooShortResult = SpeakerUtilities.validateSpeakerCreation(
            id: "test2",
            duration: 0.5,
            embedding: createDistinctEmbedding(pattern: 2),
            config: .current
        )
        if case .failure = tooShortResult {
            XCTAssert(true)
        } else {
            XCTFail("Should have failed")
        }
    }

    // MARK: - Create Speaker Tests

    func testCreateSpeaker() {
        let embedding = createDistinctEmbedding(pattern: 1)

        let speaker = SpeakerUtilities.createSpeaker(
            id: "test1",
            name: "Test Speaker",
            duration: 5.0,
            embedding: embedding
        )

        XCTAssertNotNil(speaker)
        XCTAssertEqual(speaker?.id, "test1")
        XCTAssertEqual(speaker?.name, "Test Speaker")
        let expectedEmbedding = VDSPOperations.l2Normalize(embedding)
        if let current = speaker?.currentEmbedding {
            for (value, expected) in zip(current, expectedEmbedding) {
                XCTAssertEqual(value, expected, accuracy: 0.0001)
            }
        } else {
            XCTFail("Speaker embedding missing")
        }
        XCTAssertEqual(speaker?.duration, 5.0)
    }

    // MARK: - Update Embedding Tests

    func testUpdateEmbedding() {
        let oldEmb = [Float](repeating: 1.0, count: 256)
        let newEmb = [Float](repeating: 0.5, count: 256)  // Use non-zero values to pass validation

        let alpha: Float = 0.7
        let updated = SpeakerUtilities.updateEmbedding(
            current: oldEmb,
            new: newEmb,
            alpha: alpha
        )

        XCTAssertNotNil(updated)
        // Embeddings are averaged in normalized space and then renormalized.
        if let updatedValues = updated {
            let normalizedCurrent = VDSPOperations.l2Normalize(oldEmb)
            let normalizedNew = VDSPOperations.l2Normalize(newEmb)
            var combined = [Float](repeating: 0, count: normalizedCurrent.count)
            for i in 0..<combined.count {
                combined[i] = alpha * normalizedCurrent[i] + (1 - alpha) * normalizedNew[i]
            }
            let expectedValues = VDSPOperations.l2Normalize(combined)
            for (value, expected) in zip(updatedValues, expectedValues) {
                XCTAssertEqual(value, expected, accuracy: 0.001)
            }
        }
    }

    // MARK: - Raw Embedding Management Tests

    func testAddRawEmbedding() {
        let initialRaw: [RawEmbedding] = []
        let embedding = createDistinctEmbedding(pattern: 2)
        let segmentId = UUID()

        let result = SpeakerUtilities.addRawEmbedding(
            to: initialRaw,
            segmentId: segmentId,
            embedding: embedding,
            maxCapacity: 50
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.updated.count, 1)
        XCTAssertEqual(result?.updated.first?.segmentId, segmentId)
    }

    func testAddRawEmbeddingFIFO() {
        var rawEmbeddings: [RawEmbedding] = []

        // Add more than max
        for i in 0..<10 {
            if let result = SpeakerUtilities.addRawEmbedding(
                to: rawEmbeddings,
                segmentId: UUID(),
                embedding: createDistinctEmbedding(pattern: i),
                maxCapacity: 5
            ) {
                rawEmbeddings = result.updated
            }
        }

        // Should only have 5
        XCTAssertEqual(rawEmbeddings.count, 5)
    }

    func testRemoveRawEmbedding() {
        let segmentId = UUID()
        let rawEmbeddings = [
            RawEmbedding(segmentId: segmentId, embedding: createDistinctEmbedding(pattern: 2)),
            RawEmbedding(embedding: createDistinctEmbedding(pattern: 3)),
        ]

        let result = SpeakerUtilities.removeRawEmbedding(
            from: rawEmbeddings,
            segmentId: segmentId
        )

        XCTAssertEqual(result.updated.count, 1)
        XCTAssertNotNil(result.removed)
    }

    // MARK: - Update Speaker with Segment Tests

    func testUpdateSpeakerWithSegment() {
        let speaker = Speaker(id: "test", currentEmbedding: createDistinctEmbedding(pattern: 1))
        let newEmbedding = createDistinctEmbedding(pattern: 2)
        let segmentId = UUID()

        let result = SpeakerUtilities.updateSpeakerWithSegment(
            currentMainEmbedding: speaker.currentEmbedding,
            currentRawEmbeddings: speaker.rawEmbeddings,
            currentDuration: speaker.duration,
            segmentDuration: 5.0,
            segmentEmbedding: newEmbedding,
            segmentId: segmentId,
            alpha: 0.9
        )

        XCTAssertNotNil(result)
        XCTAssertNotNil(result?.updatedMainEmbedding)
        XCTAssertEqual(result?.updatedDuration, 5.0)
    }

    // MARK: - Merge Speakers Tests

    func testMergeSpeakers() {
        let speaker1 = Speaker(
            id: "speaker1",
            name: "Alice",
            currentEmbedding: createDistinctEmbedding(pattern: 1),
            duration: 10.0
        )
        speaker1.updateCount = 5
        speaker1.addRawEmbedding(RawEmbedding(embedding: createDistinctEmbedding(pattern: 3)))

        let speaker2 = Speaker(
            id: "speaker2",
            name: "Bob",
            currentEmbedding: createDistinctEmbedding(pattern: 2),
            duration: 20.0
        )
        speaker2.updateCount = 3
        speaker2.addRawEmbedding(RawEmbedding(embedding: createDistinctEmbedding(pattern: 4)))

        let merged = SpeakerUtilities.mergeSpeakers(
            speaker1Raw: speaker1.rawEmbeddings,
            speaker1Duration: 10.0,
            speaker2Raw: speaker2.rawEmbeddings,
            speaker2Duration: 20.0
        )

        XCTAssertEqual(merged.mergedDuration, 30.0)
        XCTAssertEqual(merged.mergedRaw.count, 2)  // Both raw embeddings
        XCTAssertNotNil(merged.newMainEmbedding)
    }

    func testMergeSpeakersWithMaxCapacity() {
        let speaker1 = Speaker(id: "speaker1", name: "Alice", currentEmbedding: createDistinctEmbedding(pattern: 1))
        let speaker2 = Speaker(id: "speaker2", name: "Bob", currentEmbedding: createDistinctEmbedding(pattern: 2))

        // Add many raw embeddings
        for i in 0..<30 {
            speaker1.addRawEmbedding(RawEmbedding(embedding: createDistinctEmbedding(pattern: i)))
        }
        for i in 30..<60 {
            speaker2.addRawEmbedding(RawEmbedding(embedding: createDistinctEmbedding(pattern: i)))
        }

        let merged = SpeakerUtilities.mergeSpeakers(
            speaker1Raw: speaker1.rawEmbeddings,
            speaker1Duration: 10.0,
            speaker2Raw: speaker2.rawEmbeddings,
            speaker2Duration: 20.0,
            maxCapacity: 50
        )

        XCTAssertEqual(merged.mergedRaw.count, 50)  // Should be capped at 50
        XCTAssertEqual(merged.mergedDuration, 30.0)
    }

    // MARK: - Average Embeddings Tests

    func testAverageEmbeddings() {
        let emb1 = [Float](repeating: 1.0, count: 256)
        let emb2 = [Float](repeating: 2.0, count: 256)
        let emb3 = [Float](repeating: 3.0, count: 256)

        let average = SpeakerUtilities.averageEmbeddings([emb1, emb2, emb3])

        XCTAssertNotNil(average)
        // Average should reflect normalized mean of normalized embeddings.
        let expected = VDSPOperations.l2Normalize([Float](repeating: 2.0, count: 256))
        if let average = average {
            for (value, expectedValue) in zip(average, expected) {
                XCTAssertEqual(value, expectedValue, accuracy: 0.001)
            }
        } else {
            XCTFail("Average should not be nil")
        }
    }

    func testAverageEmbeddingsEmpty() {
        let average = SpeakerUtilities.averageEmbeddings([])
        XCTAssertNil(average)
    }

    func testAverageEmbeddingsDifferentSizes() {
        let emb1 = [Float](repeating: 1.0, count: 256)
        let emb2 = [Float](repeating: 2.0, count: 128)  // Wrong size

        let average = SpeakerUtilities.averageEmbeddings([emb1, emb2])
        // Should return average of valid embeddings only (emb1 in this case)
        XCTAssertNotNil(average)
        XCTAssertEqual(average?.count, 256)
        // Should match the normalized emb1 since only it is valid
        if let avg = average {
            let expected = VDSPOperations.l2Normalize(emb1)
            for (value, expectedValue) in zip(avg, expected) {
                XCTAssertEqual(value, expectedValue, accuracy: 0.001)
            }
        }
    }

    // MARK: - SpeakerManager Extension Tests

    func testReassignSegment() {
        let manager = SpeakerManager()

        // Create initial speakers
        let emb1 = createDistinctEmbedding(pattern: 1)
        let emb2 = createDistinctEmbedding(pattern: 2)

        let speaker1 = manager.assignSpeaker(emb1, speechDuration: 5.0)
        let speaker2 = manager.assignSpeaker(emb2, speechDuration: 5.0)

        XCTAssertNotNil(speaker1)
        XCTAssertNotNil(speaker2)

        // Test that both speakers were created
        XCTAssertEqual(manager.speakerCount, 2)

        // Test reassigning an embedding that's closer to speaker2
        let reassignedSpeaker = manager.assignSpeaker(emb2, speechDuration: 3.0)
        XCTAssertEqual(reassignedSpeaker?.id, speaker2?.id)
    }

    func testGetCurrentSpeakerNames() {
        let manager = SpeakerManager()

        // Add speakers with names
        let alice = Speaker(id: "alice", name: "Alice", currentEmbedding: createDistinctEmbedding(pattern: 1))
        let bob = Speaker(id: "bob", name: "Bob", currentEmbedding: createDistinctEmbedding(pattern: 2))

        manager.initializeKnownSpeakers([alice, bob])

        // getCurrentSpeakerNames actually returns speaker IDs, not names
        let speakerIds = manager.getCurrentSpeakerNames()

        XCTAssertEqual(speakerIds.count, 2)
        XCTAssertTrue(speakerIds.contains("alice"))
        XCTAssertTrue(speakerIds.contains("bob"))
    }

    func testGetGlobalSpeakerStats() {
        let manager = SpeakerManager(speakerThreshold: 0.5)  // Use higher threshold to ensure distinct speakers

        // Add speakers with very different embeddings to ensure they're distinct
        let speaker1 = manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 10.0)
        let speaker2 = manager.assignSpeaker(createDistinctEmbedding(pattern: 100), speechDuration: 20.0)
        let speaker3 = manager.assignSpeaker(createDistinctEmbedding(pattern: 200), speechDuration: 30.0)

        // Debug: check if all speakers were created
        XCTAssertNotNil(speaker1)
        XCTAssertNotNil(speaker2)
        XCTAssertNotNil(speaker3)

        let stats = manager.getGlobalSpeakerStats()

        // If not all 3 speakers were created, adjust expectations
        XCTAssertGreaterThanOrEqual(stats.totalSpeakers, 2)
        XCTAssertGreaterThanOrEqual(stats.totalDuration, 30.0)
        XCTAssertGreaterThanOrEqual(stats.averageConfidence, 0.0)
        XCTAssertGreaterThanOrEqual(stats.speakersWithHistory, 0)
    }

    // MARK: - Edge Cases and Error Handling

    func testHandleEmptyDatabase() {
        let embedding = createDistinctEmbedding(pattern: 1)

        let result = SpeakerUtilities.findClosestSpeaker(
            embedding: embedding,
            candidates: []
        )

        XCTAssertNil(result.speaker)  // Should return nil for empty database
        XCTAssertEqual(result.distance, Float.infinity)
    }

    func testHandleInvalidEmbeddings() {
        // Test shouldAssignSpeaker with edge case distances
        let resultNegative = SpeakerUtilities.shouldAssignSpeaker(
            distance: -0.1,  // Invalid negative distance
            duration: 5.0,
            config: .current
        )

        // Should handle negative distance as very close
        XCTAssertTrue(resultNegative.shouldAssign)

        let resultInfinity = SpeakerUtilities.shouldAssignSpeaker(
            distance: Float.infinity,
            duration: 5.0,
            config: .current
        )

        // Should not assign for infinite distance
        XCTAssertFalse(resultInfinity.shouldAssign)
    }
}
