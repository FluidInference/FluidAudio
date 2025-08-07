import XCTest

@testable import FluidAudio

/// Tests for speaker database operations without requiring models
@available(macOS 13.0, iOS 16.0, *)
final class SpeakerDatabaseTests: XCTestCase {

    func testSpeakerCreation() {
        // Test that Speaker can be created with various configurations
        let speaker1 = Speaker(name: "Alice", embedding: [Float](repeating: 0.1, count: 256))
        XCTAssertEqual(speaker1.name, "Alice")
        XCTAssertEqual(speaker1.embedding.count, 256)
        XCTAssertNotNil(speaker1.id)

        // Test with timing information
        let speaker2 = Speaker(
            name: "Bob",
            embedding: [Float](repeating: 0.2, count: 256),
            startTime: 10.0,
            endTime: 15.0,
            confidence: 0.95
        )
        XCTAssertEqual(speaker2.duration, 5.0)
        XCTAssertEqual(speaker2.confidence, 0.95)
    }

    func testSpeakerCodable() throws {
        // Test that Speaker can be encoded and decoded
        let original = Speaker(
            name: "Test Speaker",
            embedding: [Float](repeating: 0.5, count: 256),
            startTime: 1.0,
            endTime: 2.0
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(Speaker.self, from: data)

        XCTAssertEqual(decoded.name, original.name)
        XCTAssertEqual(decoded.embedding, original.embedding)
        XCTAssertEqual(decoded.startTime, original.startTime)
        XCTAssertEqual(decoded.endTime, original.endTime)
    }

    func testSpeakerArrayOperations() throws {
        // Test operations on arrays of speakers
        var speakers = [Speaker]()

        // Add multiple speakers
        for i in 0..<5 {
            speakers.append(
                Speaker(
                    name: "Speaker \(i)",
                    embedding: [Float](repeating: Float(i) * 0.1, count: 256)
                ))
        }

        XCTAssertEqual(speakers.count, 5)

        // Test filtering
        let filtered = speakers.filter { $0.name.contains("2") }
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered.first?.name, "Speaker 2")

        // Test sorting by name
        let sorted = speakers.sorted { $0.name < $1.name }
        XCTAssertEqual(sorted.first?.name, "Speaker 0")
        XCTAssertEqual(sorted.last?.name, "Speaker 4")

        // Test JSON export/import of array
        let data = try JSONEncoder().encode(speakers)
        let imported = try JSONDecoder().decode([Speaker].self, from: data)
        XCTAssertEqual(imported.count, speakers.count)
    }

    func testSpeakerDictionary() {
        // Test using speakers in a dictionary (simulating database)
        var database = [String: Speaker]()

        let speaker1 = Speaker(name: "Alice", embedding: [Float](repeating: 0.1, count: 256))
        let speaker2 = Speaker(name: "Bob", embedding: [Float](repeating: 0.2, count: 256))

        database[speaker1.id] = speaker1
        database[speaker2.id] = speaker2

        // Test retrieval
        XCTAssertNotNil(database[speaker1.id])
        XCTAssertEqual(database[speaker1.id]?.name, "Alice")

        // Test update
        database[speaker1.id]?.name = "Alice Updated"
        XCTAssertEqual(database[speaker1.id]?.name, "Alice Updated")

        // Test deletion
        database.removeValue(forKey: speaker2.id)
        XCTAssertNil(database[speaker2.id])
        XCTAssertEqual(database.count, 1)
    }

    func testSpeakerStatistics() {
        // Test speaker statistics tracking
        var speaker = Speaker(name: "Test", embedding: [Float](repeating: 0.1, count: 256))

        // Initial state
        XCTAssertEqual(speaker.totalDuration, 0)
        XCTAssertNotNil(speaker.lastSeen)

        // Update duration
        speaker.totalDuration += 10.5
        speaker.totalDuration += 5.3
        XCTAssertEqual(speaker.totalDuration, 15.8, accuracy: 0.01)

        // Update last seen
        let newDate = Date()
        speaker.lastSeen = newDate
        XCTAssertEqual(speaker.lastSeen, newDate)
    }

    func testConcurrentDatabaseAccess() {
        // Test thread-safe access to speaker database
        let queue = DispatchQueue(label: "test.db", attributes: .concurrent)
        var database = [String: Speaker]()
        let lock = NSLock()

        let expectation = XCTestExpectation(description: "Concurrent operations complete")
        let group = DispatchGroup()

        // Add speakers concurrently
        for i in 0..<100 {
            group.enter()
            queue.async {
                let speaker = Speaker(
                    name: "Speaker \(i)",
                    embedding: [Float](repeating: Float(i) * 0.01, count: 256)
                )
                lock.lock()
                database[speaker.id] = speaker
                lock.unlock()
                group.leave()
            }
        }

        group.notify(queue: .main) {
            XCTAssertEqual(database.count, 100, "Should have all 100 speakers")
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5.0)
    }

    // MARK: - Embedding Array Tests

    func testEmbeddingValidation() {
        // Test various embedding sizes and values
        let validEmbedding = [Float](repeating: 0.5, count: 256)
        let speaker1 = Speaker(name: "Valid", embedding: validEmbedding)
        XCTAssertEqual(speaker1.embedding.count, 256)

        // Test empty embedding
        let emptyEmbedding = [Float]()
        let speaker2 = Speaker(name: "Empty", embedding: emptyEmbedding)
        XCTAssertTrue(speaker2.embedding.isEmpty)

        // Test large embedding
        let largeEmbedding = [Float](repeating: 0.1, count: 1024)
        let speaker3 = Speaker(name: "Large", embedding: largeEmbedding)
        XCTAssertEqual(speaker3.embedding.count, 1024)

        // Test embedding with extreme values
        var extremeEmbedding = [Float](repeating: 0, count: 256)
        extremeEmbedding[0] = Float.infinity
        extremeEmbedding[1] = -Float.infinity
        extremeEmbedding[2] = Float.nan
        let speaker4 = Speaker(name: "Extreme", embedding: extremeEmbedding)
        XCTAssertTrue(speaker4.embedding[0].isInfinite)
        XCTAssertTrue(speaker4.embedding[1].isInfinite)
        XCTAssertTrue(speaker4.embedding[2].isNaN)
    }

    func testEmbeddingComparison() {
        // Test comparing embeddings for equality
        let embedding1 = [Float](repeating: 0.5, count: 256)
        let embedding2 = [Float](repeating: 0.5, count: 256)
        let embedding3 = [Float](repeating: 0.3, count: 256)

        let speaker1 = Speaker(name: "A", embedding: embedding1)
        let speaker2 = Speaker(name: "B", embedding: embedding2)
        let speaker3 = Speaker(name: "C", embedding: embedding3)

        // Same embeddings
        XCTAssertEqual(speaker1.embedding, speaker2.embedding)

        // Different embeddings
        XCTAssertNotEqual(speaker1.embedding, speaker3.embedding)
    }

    func testEmbeddingNormalization() {
        // Test normalizing embeddings (common preprocessing step)
        var embedding = [Float]()
        for _ in 0..<256 {
            embedding.append(Float.random(in: -1...1))
        }

        // Calculate L2 norm
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })

        // Normalize
        let normalized = embedding.map { $0 / norm }

        // Verify normalization
        let newNorm = sqrt(normalized.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(newNorm, 1.0, accuracy: 0.0001)

        let speaker = Speaker(name: "Normalized", embedding: normalized)
        XCTAssertEqual(speaker.embedding.count, 256)
    }

    func testEmbeddingDistance() {
        // Test calculating distances between embeddings
        let embedding1 = [Float](repeating: 1.0, count: 256)
        let embedding2 = [Float](repeating: 1.0, count: 256)
        let embedding3 = [Float](repeating: -1.0, count: 256)

        // Euclidean distance
        func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
            guard a.count == b.count else { return Float.infinity }
            var sum: Float = 0
            for i in 0..<a.count {
                let diff = a[i] - b[i]
                sum += diff * diff
            }
            return sqrt(sum)
        }

        // Same embeddings should have distance 0
        let dist1 = euclideanDistance(embedding1, embedding2)
        XCTAssertEqual(dist1, 0.0, accuracy: 0.0001)

        // Opposite embeddings should have maximum distance
        let dist2 = euclideanDistance(embedding1, embedding3)
        XCTAssertGreaterThan(dist2, 0)
    }

    func testEmbeddingGrouping() {
        // Test grouping speakers by similar embeddings
        var speakers = [Speaker]()

        // Create groups of similar speakers
        for group in 0..<3 {
            let baseValue = Float(group) * 0.3
            for member in 0..<3 {
                let variance = Float(member) * 0.01
                let embedding = [Float](repeating: baseValue + variance, count: 256)
                speakers.append(
                    Speaker(
                        name: "Group\(group)_Member\(member)",
                        embedding: embedding
                    ))
            }
        }

        XCTAssertEqual(speakers.count, 9)

        // Group by first embedding value (simple clustering)
        let grouped = Dictionary(grouping: speakers) { speaker in
            Int(speaker.embedding[0] * 10)  // Group by rounded value
        }

        XCTAssertEqual(grouped.count, 3, "Should have 3 groups")
        for (_, group) in grouped {
            XCTAssertEqual(group.count, 3, "Each group should have 3 members")
        }
    }

    func testEmbeddingMemoryEfficiency() {
        // Test memory usage with large numbers of embeddings
        var speakers = [Speaker]()
        let embeddingSize = 256
        let speakerCount = 1000

        for i in 0..<speakerCount {
            // Create unique embeddings
            var embedding = [Float](repeating: 0, count: embeddingSize)
            for j in 0..<embeddingSize {
                embedding[j] = Float(i * embeddingSize + j) / Float(speakerCount * embeddingSize)
            }
            speakers.append(Speaker(name: "Speaker\(i)", embedding: embedding))
        }

        XCTAssertEqual(speakers.count, speakerCount)

        // Verify all embeddings are unique
        let uniqueEmbeddings = Set(speakers.map { $0.embedding.first ?? 0 })
        XCTAssertEqual(uniqueEmbeddings.count, speakerCount)
    }

    func testEmbeddingTransformations() {
        // Test common embedding transformations
        var original = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            original[i] = Float.random(in: -1...1)
        }

        // Test mean centering
        let mean = original.reduce(0, +) / Float(original.count)
        let centered = original.map { $0 - mean }
        let newMean = centered.reduce(0, +) / Float(centered.count)
        XCTAssertEqual(newMean, 0, accuracy: 0.0001)

        // Test scaling
        let scaled = original.map { $0 * 2.0 }
        XCTAssertEqual(scaled[0], original[0] * 2.0, accuracy: 0.0001)

        // Test clipping
        let clipped = original.map { max(-0.5, min(0.5, $0)) }
        XCTAssertTrue(clipped.allSatisfy { $0 >= -0.5 && $0 <= 0.5 })

        // Create speakers with transformed embeddings
        let speaker1 = Speaker(name: "Centered", embedding: centered)
        let speaker2 = Speaker(name: "Scaled", embedding: scaled)
        let speaker3 = Speaker(name: "Clipped", embedding: clipped)

        XCTAssertEqual(speaker1.embedding.count, 256)
        XCTAssertEqual(speaker2.embedding.count, 256)
        XCTAssertEqual(speaker3.embedding.count, 256)
    }

    func testEmbeddingSerialization() throws {
        // Test that embeddings survive serialization correctly
        var embedding = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            embedding[i] = Float.random(in: -1...1)
        }

        let speaker = Speaker(name: "Test", embedding: embedding)

        // Test JSON serialization
        let jsonData = try JSONEncoder().encode(speaker)
        let decoded = try JSONDecoder().decode(Speaker.self, from: jsonData)

        // Verify embeddings are identical
        for i in 0..<256 {
            XCTAssertEqual(decoded.embedding[i], speaker.embedding[i], accuracy: 0.00001)
        }

        // Test with PropertyList
        let plistEncoder = PropertyListEncoder()
        let plistData = try plistEncoder.encode(speaker)

        let plistDecoder = PropertyListDecoder()
        let plistDecoded = try plistDecoder.decode(Speaker.self, from: plistData)

        XCTAssertEqual(plistDecoded.embedding, speaker.embedding)
    }

    // MARK: - Known Speaker Initialization Tests

    func testKnownSpeakerInitialization() {
        // Test initializing SpeakerManager with known speakers
        let manager = SpeakerManager()

        // Create known speakers with distinct embeddings
        var aliceEmbedding = [Float](repeating: 0, count: 256)
        var bobEmbedding = [Float](repeating: 0, count: 256)

        // Make embeddings distinct
        for i in 0..<256 {
            aliceEmbedding[i] = Float(i) / 256.0
            bobEmbedding[i] = Float(255 - i) / 256.0
        }

        let knownSpeakers = [
            "Alice": aliceEmbedding,
            "Bob": bobEmbedding,
        ]

        // Initialize with known speakers
        manager.initializeKnownSpeakers(knownSpeakers)

        // Test that known speakers are recognized
        // When we assign an embedding very close to Alice's
        var testEmbedding = aliceEmbedding
        testEmbedding[0] += 0.001  // Tiny variation

        let assignedId = manager.assignSpeaker(
            testEmbedding,
            speechDuration: 2.0
        )

        XCTAssertEqual(assignedId, "Alice", "Should recognize Alice")

        // Test Bob is recognized
        let bobId = manager.assignSpeaker(
            bobEmbedding,
            speechDuration: 2.0
        )

        XCTAssertEqual(bobId, "Bob", "Should recognize Bob")
    }

    func testNewSpeakerCreation() {
        // Test that new speakers are created for unknown embeddings
        let manager = SpeakerManager()

        // Create a distinct embedding for Alice
        var aliceEmbedding = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            aliceEmbedding[i] = sin(Float(i) * 0.1)
        }

        let knownSpeakers = [
            "Alice": aliceEmbedding
        ]

        // Initialize with known speakers
        manager.initializeKnownSpeakers(knownSpeakers)

        // Create different unknown speakers with very distinct embeddings
        var unknown1 = [Float](repeating: 0, count: 256)
        var unknown2 = [Float](repeating: 0, count: 256)
        for i in 0..<256 {
            unknown1[i] = cos(Float(i) * 0.2)  // Different frequency pattern
            unknown2[i] = Float(i % 2 == 0 ? 1 : -1)  // Alternating pattern
        }

        // Should create new speakers for unknown embeddings
        let id1 = manager.assignSpeaker(unknown1, speechDuration: 2.0)
        XCTAssertNotNil(id1, "Should create first new speaker")
        XCTAssertTrue(id1?.hasPrefix("Speaker_") ?? false, "Should be a new speaker ID")

        let id2 = manager.assignSpeaker(unknown2, speechDuration: 2.0)
        XCTAssertNotNil(id2, "Should create second new speaker")
        XCTAssertTrue(id2?.hasPrefix("Speaker_") ?? false, "Should be a new speaker ID")
        XCTAssertNotEqual(id1, id2, "Different embeddings should get different IDs")
    }

    func testKnownSpeakerRecognition() {
        // Test that known speakers are properly recognized
        let manager = SpeakerManager()

        // Create distinct embeddings for each speaker
        var aliceEmbedding = [Float](repeating: 0, count: 256)
        var bobEmbedding = [Float](repeating: 0, count: 256)

        // Make embeddings distinct using different patterns
        for i in 0..<256 {
            aliceEmbedding[i] = sin(Float(i) * 0.1)
            bobEmbedding[i] = cos(Float(i) * 0.1)
        }

        let knownSpeakers = [
            "Alice": aliceEmbedding,
            "Bob": bobEmbedding,
        ]

        // Initialize with known speakers
        manager.initializeKnownSpeakers(knownSpeakers)

        // Known speaker should be recognized (with slight variation)
        var testAlice = aliceEmbedding
        testAlice[0] += 0.01  // Small variation

        let aliceId = manager.assignSpeaker(
            testAlice,
            speechDuration: 2.0
        )
        XCTAssertEqual(aliceId, "Alice", "Should recognize known speaker")

        // Unknown speaker should get a new ID
        let unknownId = manager.assignSpeaker(
            [Float](repeating: 0.9, count: 256),
            speechDuration: 2.0
        )
        XCTAssertNotNil(unknownId, "Should create new speaker for unknown embedding")
        XCTAssertTrue(unknownId?.hasPrefix("Speaker_") ?? false, "Unknown should get new speaker ID")
    }

    func testKnownSpeakerPersistence() {
        // Test that known speakers persist across operations
        let manager = SpeakerManager()

        // Create and initialize known speakers
        var speakers = [String: [Float]]()
        for i in 0..<5 {
            var embedding = [Float](repeating: 0, count: 256)
            embedding[i] = 1.0  // Make each unique
            speakers["Person\(i)"] = embedding
        }

        manager.initializeKnownSpeakers(speakers)

        // Process multiple assignments
        for i in 0..<5 {
            var embedding = [Float](repeating: 0, count: 256)
            embedding[i] = 0.99  // Close to original

            let id = manager.assignSpeaker(embedding, speechDuration: 1.0)
            XCTAssertEqual(id, "Person\(i)", "Should consistently identify Person\(i)")
        }

        // Verify statistics show correct speaker count
        let stats = manager.getStatistics()
        XCTAssertTrue(stats.contains("Speakers: 5"), "Should have 5 speakers")
    }

    func testInvalidEmbeddingSize() {
        // Test that invalid embedding sizes are handled
        let manager = SpeakerManager()

        let invalidSpeakers = [
            "Invalid1": [Float](repeating: 0.1, count: 128),  // Wrong size
            "Invalid2": [Float](),  // Empty
            "Valid": [Float](repeating: 0.2, count: 256),  // Correct size
        ]

        manager.initializeKnownSpeakers(invalidSpeakers)

        // Only valid speaker should be initialized
        let validId = manager.assignSpeaker(
            [Float](repeating: 0.2, count: 256),
            speechDuration: 1.0
        )
        XCTAssertEqual(validId, "Valid", "Should only have valid speaker")

        // Invalid embeddings should not match anything
        let invalidId = manager.assignSpeaker(
            [Float](repeating: 0.1, count: 256),
            speechDuration: 1.0
        )
        XCTAssertNotEqual(invalidId, "Invalid1", "Invalid speakers should not be in database")
    }
}
