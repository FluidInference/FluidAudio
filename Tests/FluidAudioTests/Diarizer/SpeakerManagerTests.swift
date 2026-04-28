import XCTest

@testable import FluidAudio

/// Tests for SpeakerManager functionality
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

    private func normalizedEmbedding(pattern: Int) -> [Float] {
        VDSPOperations.l2Normalize(createDistinctEmbedding(pattern: pattern))
    }

    // MARK: - Basic Operations

    func testInitialization() async {
        let manager = SpeakerManager()
        let speakerCount = await manager.speakerCount
        let speakerIds = await manager.speakerIds
        XCTAssertEqual(speakerCount, 0)
        XCTAssertTrue(speakerIds.isEmpty)
    }

    func testAssignNewSpeaker() async {
        let manager = SpeakerManager()
        let embedding = createDistinctEmbedding(pattern: 1)

        let speaker = await manager.assignSpeaker(embedding, speechDuration: 2.0)

        XCTAssertNotNil(speaker)
        let speakerCount = await manager.speakerCount
        XCTAssertEqual(speakerCount, 1)
        // ID should be numeric (starting from 1)
        XCTAssertEqual(speaker?.id, "1")
    }

    func testAssignExistingSpeaker() async {
        let manager = SpeakerManager(speakerThreshold: 0.3)  // Low threshold for testing

        // Add first speaker
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let speaker1 = await manager.assignSpeaker(embedding1, speechDuration: 2.0)

        // Add nearly identical embedding - should match existing speaker
        var embedding2 = embedding1
        embedding2[0] += 0.001  // Tiny variation
        let speaker2 = await manager.assignSpeaker(embedding2, speechDuration: 2.0)

        XCTAssertEqual(speaker1?.id, speaker2?.id)
        let speakerCount = await manager.speakerCount
        XCTAssertEqual(speakerCount, 1)  // Should still be 1 speaker
    }

    func testMultipleSpeakers() async {
        let manager = SpeakerManager(speakerThreshold: 0.5)

        // Create distinct embeddings
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let embedding2 = createDistinctEmbedding(pattern: 2)

        let speaker1 = await manager.assignSpeaker(embedding1, speechDuration: 2.0)
        let speaker2 = await manager.assignSpeaker(embedding2, speechDuration: 2.0)

        XCTAssertNotNil(speaker1)
        XCTAssertNotNil(speaker2)
        XCTAssertNotEqual(speaker1?.id, speaker2?.id)
        let speakerCount = await manager.speakerCount
        XCTAssertEqual(speakerCount, 2)
    }

    // MARK: - Known Speaker Initialization

    func testInitializeKnownSpeakers() async {
        let manager = SpeakerManager()

        let knownSpeakers = [
            Speaker(
                id: "Alice",
                name: "Alice",
                currentEmbedding: createDistinctEmbedding(pattern: 10),
                duration: 0,
                createdAt: Date(),
                updatedAt: Date()
            ),
            Speaker(
                id: "Bob",
                name: "Bob",
                currentEmbedding: createDistinctEmbedding(pattern: 20),
                duration: 0,
                createdAt: Date(),
                updatedAt: Date()
            ),
        ]

        await manager.initializeKnownSpeakers(knownSpeakers)

        let speakerCount = await manager.speakerCount
        let speakerIds = await manager.speakerIds
        XCTAssertEqual(speakerCount, 2)
        XCTAssertTrue(speakerIds.contains("Alice"))
        XCTAssertTrue(speakerIds.contains("Bob"))
    }

    func testRecognizeKnownSpeaker() async {
        let manager = SpeakerManager(speakerThreshold: 0.3)

        let aliceEmbedding = createDistinctEmbedding(pattern: 10)
        let aliceSpeaker = Speaker(
            id: "Alice",
            name: "Alice",
            currentEmbedding: aliceEmbedding,
            duration: 0,
            createdAt: Date(),
            updatedAt: Date()
        )
        await manager.initializeKnownSpeakers([aliceSpeaker])

        // Test with exact same embedding
        let testEmbedding = aliceEmbedding

        let assignedSpeaker = await manager.assignSpeaker(testEmbedding, speechDuration: 2.0)
        XCTAssertEqual(assignedSpeaker?.id, "Alice")
    }

    func testInitializeKnownSpeakersPreservesPermanentByDefault() async {
        let manager = SpeakerManager()

        let original = Speaker(
            id: "Alice",
            name: "Original",
            currentEmbedding: createDistinctEmbedding(pattern: 10),
            duration: 4.0
        )
        await manager.initializeKnownSpeakers([original])
        await manager.makeSpeakerPermanent("Alice")

        let replacement = Speaker(
            id: "Alice",
            name: "Replacement",
            currentEmbedding: createDistinctEmbedding(pattern: 20),
            duration: 8.0
        )

        await manager.initializeKnownSpeakers([replacement], mode: .overwrite, preserveIfPermanent: true)

        let stored = await manager.getSpeaker(for: "Alice")
        XCTAssertEqual(stored?.name, "Original")
        XCTAssertEqual(stored?.duration, 4.0)
    }

    func testInitializeKnownSpeakersOverwriteCanReplacePermanentWhenAllowed() async {
        let manager = SpeakerManager()

        let original = Speaker(
            id: "Alice",
            name: "Original",
            currentEmbedding: createDistinctEmbedding(pattern: 10),
            duration: 4.0,
            isPermanent: true
        )
        await manager.initializeKnownSpeakers([original])

        let replacement = Speaker(
            id: "Alice",
            name: "Replacement",
            currentEmbedding: createDistinctEmbedding(pattern: 20),
            duration: 10.0
        )

        await manager.initializeKnownSpeakers([replacement], mode: .overwrite, preserveIfPermanent: false)

        let stored = await manager.getSpeaker(for: "Alice")
        XCTAssertEqual(stored?.name, "Replacement")
        XCTAssertEqual(stored?.duration, 10.0)
    }

    func testInitializeKnownSpeakersMergeCombinesDurations() async {
        let manager = SpeakerManager()

        let base = Speaker(
            id: "Alice",
            name: "Alice",
            currentEmbedding: createDistinctEmbedding(pattern: 10),
            duration: 2.0
        )
        let incoming = Speaker(
            id: "Alice",
            name: "Alice",
            currentEmbedding: createDistinctEmbedding(pattern: 11),
            duration: 3.0
        )

        await manager.initializeKnownSpeakers([base])
        await manager.initializeKnownSpeakers([incoming], mode: .merge)

        let stored = await manager.getSpeaker(for: "Alice")
        XCTAssertEqual(stored?.duration, 5.0)
    }

    func testInvalidEmbeddingSize() async {
        let manager = SpeakerManager()

        // Test with wrong size
        let invalidEmbedding = [Float](repeating: 0.5, count: 128)
        let speaker = await manager.assignSpeaker(invalidEmbedding, speechDuration: 2.0)

        XCTAssertNil(speaker)
        let speakerCount = await manager.speakerCount
        XCTAssertEqual(speakerCount, 0)
    }

    func testEmptyEmbedding() async {
        let manager = SpeakerManager()

        let emptyEmbedding = [Float]()
        let speaker = await manager.assignSpeaker(emptyEmbedding, speechDuration: 2.0)

        XCTAssertNil(speaker)
        let speakerCount = await manager.speakerCount
        XCTAssertEqual(speakerCount, 0)
    }

    // MARK: - Speaker Info Access

    func testGetSpeakerInfo() async {
        let manager = SpeakerManager()
        let embedding = createDistinctEmbedding(pattern: 1)

        let speaker = await manager.assignSpeaker(embedding, speechDuration: 3.5)
        XCTAssertNotNil(speaker)

        if let id = speaker?.id {
            let info = await manager.getSpeaker(for: id)
            XCTAssertNotNil(info)
            XCTAssertEqual(info?.id, id)
            let normalizedExpected = VDSPOperations.l2Normalize(embedding)
            XCTAssertEqual(info?.currentEmbedding, normalizedExpected)
            XCTAssertEqual(info?.duration, 3.5)
        }
    }

    func testPublicSpeakerInfoMembers() async {
        // This test verifies that all SpeakerInfo members are public as requested in PR #63
        let manager = SpeakerManager()
        let embedding = createDistinctEmbedding(pattern: 1)

        let speaker = await manager.assignSpeaker(embedding, speechDuration: 5.0)
        XCTAssertNotNil(speaker)

        if let id = speaker?.id, let info = await manager.getSpeaker(for: id) {
            // Test that all public properties are accessible
            let publicId = info.id
            let publicEmbedding = info.currentEmbedding
            let publicDuration = info.duration
            let publicUpdatedAt = info.updatedAt
            let publicUpdateCount = info.updateCount

            // Verify the values
            XCTAssertEqual(publicId, id)
            let normalizedExpected = VDSPOperations.l2Normalize(embedding)
            XCTAssertEqual(publicEmbedding, normalizedExpected)
            XCTAssertEqual(publicDuration, 5.0)
            XCTAssertNotNil(publicUpdatedAt)
            XCTAssertEqual(publicUpdateCount, 1)
        }
    }

    func testGetAllSpeakerInfo() async {
        let manager = SpeakerManager()

        // Add multiple speakers
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let embedding2 = createDistinctEmbedding(pattern: 2)

        let speaker1 = await manager.assignSpeaker(embedding1, speechDuration: 2.0)
        let speaker2 = await manager.assignSpeaker(embedding2, speechDuration: 3.0)

        let allInfo = await manager.getAllSpeakers()

        XCTAssertEqual(allInfo.count, 2)
        if let id1 = speaker1?.id {
            XCTAssertNotNil(allInfo[id1])
        }
        if let id2 = speaker2?.id {
            XCTAssertNotNil(allInfo[id2])
        }
    }

    // MARK: - Lookup Helpers

    func testFindSpeakerAndMatchingSpeakers() async {
        let manager = SpeakerManager(speakerThreshold: 0.8)

        await manager.upsertSpeaker(id: "A", currentEmbedding: normalizedEmbedding(pattern: 1), duration: 5.0)
        await manager.upsertSpeaker(id: "B", currentEmbedding: normalizedEmbedding(pattern: 2), duration: 5.0)

        let (matchId, distance) = await manager.findSpeaker(with: normalizedEmbedding(pattern: 1))
        XCTAssertEqual(matchId, "A")
        XCTAssertEqual(distance, 0.0, accuracy: 0.0001)

        var orthogonalEmbedding0 = [Float](repeating: 0, count: 256)
        var orthogonalEmbedding1 = [Float](repeating: 0, count: 256)
        orthogonalEmbedding0[0] = 1
        orthogonalEmbedding1[1] = 1
        await manager.upsertSpeaker(id: "C", currentEmbedding: orthogonalEmbedding0, duration: 5.0)
        let (missingId, missingDistance) = await manager.findSpeaker(
            with: orthogonalEmbedding1,
            speakerThreshold: 0.5
        )
        XCTAssertNil(missingId)
        XCTAssertEqual(missingDistance, .infinity)

        let combined = zip(normalizedEmbedding(pattern: 1), normalizedEmbedding(pattern: 2)).map { ($0 + $1) / 2 }
        let matches = await manager.findMatchingSpeakers(
            with: VDSPOperations.l2Normalize(combined),
            speakerThreshold: 2.0
        )

        XCTAssertEqual(matches.count, 3)
        XCTAssertLessThanOrEqual(matches[0].distance, matches[1].distance)
        XCTAssertEqual(Set(matches.map(\.id)), Set(["A", "B", "C"]))
    }

    func testFindSpeakersWhereFiltersByPredicate() async {
        let manager = SpeakerManager()
        await manager.upsertSpeaker(id: "short", currentEmbedding: normalizedEmbedding(pattern: 10), duration: 1.0)
        await manager.upsertSpeaker(id: "long", currentEmbedding: normalizedEmbedding(pattern: 20), duration: 8.0)

        let filtered = await manager.findSpeakers { $0.duration > 5.0 }
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered.first, "long")
    }

    // MARK: - Clear Operations

    func testResetSpeakers() async {
        let manager = SpeakerManager()

        // Add speakers
        _ = await manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 2.0)
        _ = await manager.assignSpeaker(createDistinctEmbedding(pattern: 2), speechDuration: 2.0)

        let countBefore = await manager.speakerCount
        XCTAssertEqual(countBefore, 2)

        await manager.reset()

        let countAfter = await manager.speakerCount
        let idsAfter = await manager.speakerIds
        XCTAssertEqual(countAfter, 0)
        XCTAssertTrue(idsAfter.isEmpty)
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

    func testGetStatistics() async {
        let manager = SpeakerManager()

        // Add speakers with different durations
        _ = await manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 10.0)
        _ = await manager.assignSpeaker(createDistinctEmbedding(pattern: 2), speechDuration: 20.0)

        // getStatistics method was removed - test speaker count instead
        let speakerCount = await manager.speakerCount
        XCTAssertEqual(speakerCount, 2)

        // Verify speakers were added with correct info
        let allInfo = await manager.getAllSpeakers()
        XCTAssertEqual(allInfo.count, 2)
    }

    // MARK: - Thread Safety

    func testConcurrentAccess() async {
        let manager = SpeakerManager(speakerThreshold: 0.5)  // Set threshold for distinct embeddings
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

        // Concurrent writes and reads via structured concurrency
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<iterations {
                let embedding = embeddings[i]
                group.addTask {
                    _ = await manager.assignSpeaker(embedding, speechDuration: 2.0)
                }
            }

            for _ in 0..<iterations {
                group.addTask {
                    _ = await manager.speakerCount
                    _ = await manager.speakerIds
                }
            }
        }

        // Due to concurrent operations and clustering, we may not get exactly iterations speakers
        // But we should have at least some distinct speakers
        let finalCount = await manager.speakerCount
        XCTAssertGreaterThan(finalCount, 0)
        XCTAssertLessThanOrEqual(finalCount, iterations)
    }

    // MARK: - Upsert Tests

    func testUpsertNewSpeaker() async {
        let manager = SpeakerManager()
        let embedding = createDistinctEmbedding(pattern: 1)

        // Upsert a new speaker
        await manager.upsertSpeaker(
            id: "TestSpeaker1",
            currentEmbedding: embedding,
            duration: 5.0
        )

        let speakerCount = await manager.speakerCount
        XCTAssertEqual(speakerCount, 1)

        let info = await manager.getSpeaker(for: "TestSpeaker1")
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.id, "TestSpeaker1")
        let normalizedExpected = VDSPOperations.l2Normalize(embedding)
        XCTAssertEqual(info?.currentEmbedding, normalizedExpected)
        XCTAssertEqual(info?.duration, 5.0)
        XCTAssertEqual(info?.updateCount, 1)
    }

    func testUpsertExistingSpeaker() async {
        let manager = SpeakerManager()
        let embedding1 = createDistinctEmbedding(pattern: 1)
        let embedding2 = createDistinctEmbedding(pattern: 2)

        // Insert initial speaker
        await manager.upsertSpeaker(
            id: "TestSpeaker1",
            currentEmbedding: embedding1,
            duration: 5.0
        )

        let originalInfo = await manager.getSpeaker(for: "TestSpeaker1")
        let originalCreatedAt = originalInfo?.createdAt

        // Wait a bit to ensure different timestamp
        try? await Task.sleep(nanoseconds: 10_000_000)

        // Update the same speaker
        await manager.upsertSpeaker(
            id: "TestSpeaker1",
            currentEmbedding: embedding2,
            duration: 10.0,
            updateCount: 5
        )

        let speakerCount = await manager.speakerCount
        XCTAssertEqual(speakerCount, 1)  // Should still be 1 speaker

        let updatedInfo = await manager.getSpeaker(for: "TestSpeaker1")
        XCTAssertNotNil(updatedInfo)
        XCTAssertEqual(updatedInfo?.id, "TestSpeaker1")
        XCTAssertEqual(updatedInfo?.currentEmbedding, embedding2)
        XCTAssertEqual(updatedInfo?.duration, 10.0)
        XCTAssertEqual(updatedInfo?.updateCount, 5)
        // CreatedAt should remain the same
        XCTAssertEqual(updatedInfo?.createdAt, originalCreatedAt)
        // UpdatedAt should be different
        XCTAssertNotEqual(updatedInfo?.updatedAt, originalCreatedAt)
    }

    func testUpsertWithSpeakerObject() async {
        let manager = SpeakerManager()
        let embedding = createDistinctEmbedding(pattern: 1)

        var speaker = Speaker(
            id: "Alice",
            name: "Alice",
            currentEmbedding: embedding,
            duration: 7.5
        )

        // Add some raw embeddings
        let rawEmbedding = RawEmbedding(embedding: embedding)
        speaker.addRawEmbedding(rawEmbedding)

        await manager.upsertSpeaker(speaker)

        let speakerCount = await manager.speakerCount
        XCTAssertEqual(speakerCount, 1)

        let info = await manager.getSpeaker(for: "Alice")
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.id, "Alice")
        let normalizedExpected = VDSPOperations.l2Normalize(embedding)
        XCTAssertEqual(info?.currentEmbedding, normalizedExpected)
        XCTAssertEqual(info?.duration, 7.5)
        XCTAssertEqual(info?.rawEmbeddings.count, 1)
    }

    // MARK: - Permanence & Merge Operations

    func testMakeAndRevokePermanentSpeakers() async throws {
        let manager = SpeakerManager()
        let speaker = await manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 2.5)
        let id = try XCTUnwrap(speaker?.id)

        await manager.makeSpeakerPermanent(id)
        let permanentIds = await manager.permanentSpeakerIds
        XCTAssertTrue(permanentIds.contains(id))

        await manager.removeSpeaker(id)
        let stillHas = await manager.hasSpeaker(id)
        XCTAssertTrue(stillHas)

        await manager.revokePermanence(from: id)
        await manager.removeSpeaker(id)
        let removedNow = await manager.hasSpeaker(id)
        XCTAssertFalse(removedNow)
    }

    func testMergeSpeakerRespectsPermanentFlag() async throws {
        let manager = SpeakerManager()
        let speaker1 = await manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 3.0)
        let speaker2 = await manager.assignSpeaker(createDistinctEmbedding(pattern: 2), speechDuration: 4.0)

        let id1 = try XCTUnwrap(speaker1?.id)
        let id2 = try XCTUnwrap(speaker2?.id)

        await manager.makeSpeakerPermanent(id1)
        await manager.mergeSpeaker(id1, into: id2)
        let has1 = await manager.hasSpeaker(id1)
        let has2 = await manager.hasSpeaker(id2)
        XCTAssertTrue(has1)
        XCTAssertTrue(has2)

        await manager.mergeSpeaker(id1, into: id2, mergedName: "Merged Speaker", stopIfPermanent: false)
        let hasAfterMerge1 = await manager.hasSpeaker(id1)
        XCTAssertFalse(hasAfterMerge1)
        let mergedOpt = await manager.getSpeaker(for: id2)
        let merged = try XCTUnwrap(mergedOpt)
        XCTAssertEqual(merged.name, "Merged Speaker")
        let finalCount = await manager.speakerCount
        XCTAssertEqual(finalCount, 1)
        XCTAssertGreaterThan(merged.duration, 4.0)
    }

    func testFindMergeablePairsRespectsPermanentExclusion() async {
        let manager = SpeakerManager(speakerThreshold: 0.3)
        let base = normalizedEmbedding(pattern: 1)
        var close = base
        close[0] += 0.001
        close = VDSPOperations.l2Normalize(close)
        let far = normalizedEmbedding(pattern: 80)

        await manager.upsertSpeaker(id: "A", currentEmbedding: base, duration: 5.0)
        await manager.upsertSpeaker(id: "B", currentEmbedding: close, duration: 5.0)
        await manager.upsertSpeaker(id: "C", currentEmbedding: far, duration: 5.0)

        let pairs = await manager.findMergeablePairs(speakerThreshold: 0.2)
        XCTAssertEqual(pairs.count, 1)
        XCTAssertEqual(Set([pairs[0].speakerToMerge, pairs[0].destination]), Set(["A", "B"]))

        await manager.makeSpeakerPermanent("A")
        await manager.makeSpeakerPermanent("B")

        let filtered = await manager.findMergeablePairs(speakerThreshold: 0.2, excludeIfBothPermanent: true)
        XCTAssertTrue(filtered.isEmpty)

        let unfiltered = await manager.findMergeablePairs(speakerThreshold: 0.2, excludeIfBothPermanent: false)
        XCTAssertEqual(unfiltered.count, 1)
        XCTAssertEqual(Set([unfiltered[0].speakerToMerge, unfiltered[0].destination]), Set(["A", "B"]))
    }

    // MARK: - Removal & Reset

    func testRemoveSpeakersInactiveAndPredicateVariants() async {
        let manager = SpeakerManager()
        let now = Date()
        await manager.upsertSpeaker(
            id: "old",
            currentEmbedding: normalizedEmbedding(pattern: 3),
            duration: 2.0,
            updatedAt: now.addingTimeInterval(-120)
        )
        await manager.upsertSpeaker(
            id: "recent",
            currentEmbedding: normalizedEmbedding(pattern: 4),
            duration: 2.0,
            updatedAt: now
        )

        await manager.removeSpeakersInactive(since: now.addingTimeInterval(-60))
        let hasOld = await manager.hasSpeaker("old")
        let hasRecent = await manager.hasSpeaker("recent")
        XCTAssertFalse(hasOld)
        XCTAssertTrue(hasRecent)

        await manager.makeSpeakerPermanent("recent")
        await manager.removeSpeakers { $0.duration <= 2.0 }
        let hasRecentAfterKeep = await manager.hasSpeaker("recent")
        XCTAssertTrue(hasRecentAfterKeep)

        await manager.removeSpeakers(where: { $0.duration <= 2.0 }, keepIfPermanent: false)
        let hasRecentAfterForce = await manager.hasSpeaker("recent")
        XCTAssertFalse(hasRecentAfterForce)
    }

    func testResetKeepsPermanentSpeakers() async throws {
        let manager = SpeakerManager()
        let speaker1 = await manager.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 2.0)
        let speaker2 = await manager.assignSpeaker(createDistinctEmbedding(pattern: 2), speechDuration: 2.0)

        let id1 = try XCTUnwrap(speaker1?.id)
        let id2 = try XCTUnwrap(speaker2?.id)

        await manager.makeSpeakerPermanent(id1)
        await manager.reset(keepIfPermanent: true)

        let has1 = await manager.hasSpeaker(id1)
        let has2 = await manager.hasSpeaker(id2)
        let ids = await manager.speakerIds
        XCTAssertTrue(has1)
        XCTAssertFalse(has2)
        XCTAssertEqual(ids, [id1])
    }

    // MARK: - Embedding Update Tests

    func testEmbeddingUpdateWithinAssignSpeaker() async {
        let manager = SpeakerManager(
            speakerThreshold: 0.3,
            embeddingThreshold: 0.2,
            minEmbeddingUpdateDuration: 2.0
        )

        // Create initial speaker
        let emb1 = createDistinctEmbedding(pattern: 1)
        let speaker1 = await manager.assignSpeaker(emb1, speechDuration: 3.0)
        XCTAssertNotNil(speaker1)

        // Get initial state
        let initialInfo = await manager.getSpeaker(for: speaker1!.id)
        let initialUpdateCount = initialInfo?.updateCount ?? 0

        // Assign similar embedding with sufficient duration - should update
        var emb2 = emb1
        emb2[0] += 0.01  // Very similar
        let speaker2 = await manager.assignSpeaker(emb2, speechDuration: 3.0)

        XCTAssertEqual(speaker2?.id, speaker1?.id)  // Same speaker

        // Check that embedding was updated
        let updatedInfo = await manager.getSpeaker(for: speaker1!.id)
        XCTAssertGreaterThan(updatedInfo?.updateCount ?? 0, initialUpdateCount)
        XCTAssertNotEqual(updatedInfo?.currentEmbedding, emb1)  // Embedding changed
    }

    func testNoEmbeddingUpdateForShortDuration() async {
        let manager = SpeakerManager(
            speakerThreshold: 0.3,
            embeddingThreshold: 0.2,
            minEmbeddingUpdateDuration: 2.0
        )

        // Create initial speaker
        let emb1 = createDistinctEmbedding(pattern: 1)
        let speaker1 = await manager.assignSpeaker(emb1, speechDuration: 3.0)
        XCTAssertNotNil(speaker1)

        // Get initial state
        let initialInfo = await manager.getSpeaker(for: speaker1!.id)
        let initialUpdateCount = initialInfo?.updateCount ?? 0

        // Assign similar embedding with short duration - WILL update embedding now (duration check removed)
        var emb2 = emb1
        emb2[0] += 0.01
        let speaker2 = await manager.assignSpeaker(emb2, speechDuration: 0.5)

        XCTAssertEqual(speaker2?.id, speaker1?.id)  // Same speaker

        // Check that embedding WAS updated (since duration check was removed)
        let updatedInfo = await manager.getSpeaker(for: speaker1!.id)
        XCTAssertGreaterThan(updatedInfo?.updateCount ?? 0, initialUpdateCount)  // Updated
        XCTAssertNotEqual(updatedInfo?.currentEmbedding, emb1)  // Embedding changed
        XCTAssertGreaterThan(updatedInfo?.duration ?? 0, 3.0)  // Duration still increased
    }

    func testRawEmbeddingFIFOInManager() async {
        let manager = SpeakerManager(
            speakerThreshold: 0.3,
            embeddingThreshold: 0.2,
            minEmbeddingUpdateDuration: 2.0
        )

        // Create initial speaker
        let emb1 = createDistinctEmbedding(pattern: 1)
        let speaker = await manager.assignSpeaker(emb1, speechDuration: 3.0)
        XCTAssertNotNil(speaker)

        // Add many embeddings to trigger FIFO (max 50)
        for i in 0..<60 {
            var emb = emb1
            emb[0] += Float(i) * 0.001  // Slight variations
            _ = await manager.assignSpeaker(emb, speechDuration: 2.5)
        }

        // Check that raw embeddings are limited to 50
        let info = await manager.getSpeaker(for: speaker!.id)
        XCTAssertLessThanOrEqual(info?.rawEmbeddings.count ?? 0, 50)
    }

    // MARK: - Edge Cases

    func testSpeakerThresholdBoundaries() async {
        // Test with very low threshold (everything matches)
        let manager1 = SpeakerManager(speakerThreshold: 0.01)
        _ = await manager1.assignSpeaker(createDistinctEmbedding(pattern: 1), speechDuration: 2.0)
        var similarEmbedding = createDistinctEmbedding(pattern: 1)
        similarEmbedding[0] += 0.001  // Tiny variation
        _ = await manager1.assignSpeaker(similarEmbedding, speechDuration: 2.0)
        let count1 = await manager1.speakerCount
        XCTAssertEqual(count1, 1)  // Should match to same speaker

        // Test with high threshold (only exact matches)
        let manager2 = SpeakerManager(speakerThreshold: 0.001)  // Very small threshold
        let emb1 = createDistinctEmbedding(pattern: 1)
        _ = await manager2.assignSpeaker(emb1, speechDuration: 2.0)
        _ = await manager2.assignSpeaker(emb1, speechDuration: 2.0)  // Exact same embedding
        let count2 = await manager2.speakerCount
        XCTAssertEqual(count2, 1)  // Should match to same speaker
    }

    func testMinDurationFiltering() async {
        let manager = SpeakerManager(
            speakerThreshold: 0.5,
            embeddingThreshold: 0.3,
            minSpeechDuration: 2.0
        )

        let embedding = createDistinctEmbedding(pattern: 1)

        // Test with duration below threshold - should not create new speaker
        let speaker1 = await manager.assignSpeaker(embedding, speechDuration: 0.5)
        XCTAssertNil(speaker1)  // Should return nil for short duration
        let count0 = await manager.speakerCount
        XCTAssertEqual(count0, 0)  // No speaker created

        // Test with duration above threshold - should create speaker
        let speaker2 = await manager.assignSpeaker(embedding, speechDuration: 3.0)
        XCTAssertNotNil(speaker2)
        let count1 = await manager.speakerCount
        XCTAssertEqual(count1, 1)  // One speaker created

        // Test again with short duration on existing speaker
        let speaker3 = await manager.assignSpeaker(embedding, speechDuration: 0.5)
        XCTAssertEqual(speaker3?.id, speaker2?.id)  // Should match existing speaker even with short duration
    }
}
