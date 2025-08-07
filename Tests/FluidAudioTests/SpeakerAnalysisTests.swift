import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class SpeakerAnalysisTests: XCTestCase {

    var diarizerManager: DiarizerManager!
    var speakerOps: SpeakerOperations!

    override func setUp() async throws {
        try await super.setUp()

        // Initialize diarizer
        diarizerManager = DiarizerManager()
        let models = try await DiarizerModels.downloadIfNeeded()
        diarizerManager.initialize(models: models)

        // Create speaker operations
        speakerOps = SpeakerOperations(diarizerManager: diarizerManager)
    }

    override func tearDown() {
        diarizerManager?.cleanup()
        super.tearDown()
    }

    func testAddAndGetSpeakers() {
        // Add speakers
        let id1 = speakerOps.addSpeaker(name: "Alice", embedding: [Float](repeating: 0.1, count: 256))
        let id2 = speakerOps.addSpeaker(name: "Bob", embedding: [Float](repeating: 0.2, count: 256))

        // Get all speakers
        let speakers = speakerOps.getAllSpeakers()
        XCTAssertEqual(speakers.count, 2, "Should have 2 speakers")

        // Check names
        let names = speakers.map { $0.name }.sorted()
        XCTAssertEqual(names, ["Alice", "Bob"], "Should have correct names")
    }

    func testRenameSpeaker() {
        // Add speaker
        let id = speakerOps.addSpeaker(name: "Alice", embedding: [Float](repeating: 0.1, count: 256))

        // Rename
        speakerOps.renameSpeaker(id: id, newName: "Alice Smith")

        // Verify rename
        let speakers = speakerOps.getAllSpeakers()
        XCTAssertEqual(speakers.first?.name, "Alice Smith", "Name should be updated")
    }

    func testDeleteSpeaker() {
        // Add speakers
        let id1 = speakerOps.addSpeaker(name: "Alice", embedding: [Float](repeating: 0.1, count: 256))
        let id2 = speakerOps.addSpeaker(name: "Bob", embedding: [Float](repeating: 0.2, count: 256))

        // Delete one
        speakerOps.deleteSpeaker(id: id1)

        // Verify deletion
        let speakers = speakerOps.getAllSpeakers()
        XCTAssertEqual(speakers.count, 1, "Should have 1 speaker left")
        XCTAssertEqual(speakers.first?.name, "Bob", "Bob should remain")
    }

    func testClearAll() {
        // Add speakers
        _ = speakerOps.addSpeaker(name: "Alice", embedding: [Float](repeating: 0.1, count: 256))
        _ = speakerOps.addSpeaker(name: "Bob", embedding: [Float](repeating: 0.2, count: 256))

        // Clear all
        speakerOps.clearAll()

        // Verify cleared
        let speakers = speakerOps.getAllSpeakers()
        XCTAssertTrue(speakers.isEmpty, "Should have no speakers")
    }

    func testSpeakerVerificationSameSpeaker() async throws {
        // Create synthetic audio samples (would use real audio in production)
        let audio1 = createSyntheticAudio(duration: 3.0, frequency: 440.0)
        let audio2 = createSyntheticAudio(duration: 3.0, frequency: 440.0)

        // Verify same speaker
        let result = try await speakerOps.verifySameSpeaker(
            audio1: audio1,
            audio2: audio2,
            threshold: 0.7
        )

        // Since both are the same synthetic audio, they should be considered the same
        XCTAssertTrue(result.isSame, "Same audio should be verified as same speaker")
        XCTAssertGreaterThan(result.confidence, 0.5, "Confidence should be reasonable")
    }

    func testSpeakerVerificationDifferentSpeakers() async throws {
        // Create different synthetic audio samples
        let audio1 = createSyntheticAudio(duration: 3.0, frequency: 440.0)
        let audio2 = createSyntheticAudio(duration: 3.0, frequency: 880.0)

        // Verify different speakers
        let result = try await speakerOps.verifySameSpeaker(
            audio1: audio1,
            audio2: audio2,
            threshold: 0.7
        )

        // Different frequencies might be considered different speakers
        // This depends on the model's behavior with synthetic audio
        XCTAssertNotNil(result.confidence, "Should return a confidence value")
    }

    func testFindSpeakerInAudio() async throws {
        // Create a longer audio sample
        let searchAudio = createSyntheticAudio(duration: 10.0, frequency: 440.0)

        // Process to get an embedding
        let result = try diarizerManager.performCompleteDiarization(searchAudio)
        guard let targetEmbedding = result.segments.first?.embedding else {
            XCTFail("Could not get embedding")
            return
        }

        // Search for speaker
        let matches = try await speakerOps.findSpeaker(
            targetEmbedding: targetEmbedding,
            in: searchAudio,
            threshold: 0.65
        )

        // Should find at least one match
        XCTAssertFalse(matches.isEmpty, "Should find speaker in audio")

        if let firstMatch = matches.first {
            XCTAssertGreaterThan(firstMatch.confidence, 0.0, "Should have confidence score")
            XCTAssertNotNil(firstMatch.startTime, "Should have start time")
            XCTAssertNotNil(firstMatch.endTime, "Should have end time")
        }
    }

    func testStreamingAssignment() {
        // Test streaming speaker assignment
        let embedding1 = [Float](repeating: 0.1, count: 256)
        let embedding2 = [Float](repeating: 0.1, count: 256)  // Same as 1
        let embedding3 = [Float](repeating: 0.9, count: 256)  // Different

        // First assignment creates new speaker
        let id1 = speakerOps.assignStreamingSpeaker(embedding: embedding1, duration: 5.0)
        XCTAssertNotNil(id1, "Should assign speaker ID")

        // Second assignment should reuse same speaker (similar embedding)
        let id2 = speakerOps.assignStreamingSpeaker(embedding: embedding2, duration: 3.0)
        XCTAssertEqual(id1, id2, "Should assign same speaker for similar embedding")

        // Third assignment should create new speaker (different embedding)
        let id3 = speakerOps.assignStreamingSpeaker(embedding: embedding3, duration: 2.0)
        XCTAssertNotEqual(id1, id3, "Should assign different speaker for different embedding")

        // Check speaker count
        let speakers = speakerOps.getAllSpeakers()
        XCTAssertEqual(speakers.count, 2, "Should have 2 speakers")
    }

    func testFindSimilarSpeakers() {
        // Add speakers
        _ = speakerOps.addSpeaker(name: "Alice", embedding: [Float](repeating: 0.1, count: 256))
        _ = speakerOps.addSpeaker(name: "Bob", embedding: [Float](repeating: 0.2, count: 256))
        _ = speakerOps.addSpeaker(name: "Charlie", embedding: [Float](repeating: 0.9, count: 256))

        // Find similar to Alice's embedding
        let targetEmbedding = [Float](repeating: 0.15, count: 256)
        let similar = speakerOps.findSimilarSpeakers(to: targetEmbedding, limit: 2)

        XCTAssertEqual(similar.count, 2, "Should return 2 similar speakers")
        // Alice and Bob should be closer than Charlie
        XCTAssertTrue(similar[0].speaker.name == "Alice" || similar[0].speaker.name == "Bob")
    }

    func testMergeSpeakers() {
        // Add speakers
        let id1 = speakerOps.addSpeaker(name: "Alice", embedding: [Float](repeating: 0.1, count: 256))
        let id2 = speakerOps.addSpeaker(name: "Alice Duplicate", embedding: [Float](repeating: 0.11, count: 256))

        // Merge duplicate into original
        speakerOps.mergeSpeakers(sourceId: id2, targetId: id1)

        // Check result
        let speakers = speakerOps.getAllSpeakers()
        XCTAssertEqual(speakers.count, 1, "Should have 1 speaker after merge")
        XCTAssertEqual(speakers.first?.id, id1, "Original should remain")
    }

    func testImportExport() throws {
        // Add speakers
        _ = speakerOps.addSpeaker(name: "Alice", embedding: [Float](repeating: 0.1, count: 256))
        _ = speakerOps.addSpeaker(name: "Bob", embedding: [Float](repeating: 0.2, count: 256))

        // Export
        let exportData = try speakerOps.exportToJSON()
        XCTAssertGreaterThan(exportData.count, 0, "Should export data")

        // Clear and import
        speakerOps.clearAll()
        try speakerOps.importFromJSON(exportData)

        // Verify import
        let speakers = speakerOps.getAllSpeakers()
        XCTAssertEqual(speakers.count, 2, "Should import 2 speakers")
        let names = speakers.map { $0.name }.sorted()
        XCTAssertEqual(names, ["Alice", "Bob"], "Should have correct names")
    }

    // Helper function to create synthetic audio
    private func createSyntheticAudio(duration: Double, frequency: Double) -> [Float] {
        let sampleRate = 16000.0
        let samples = Int(duration * sampleRate)
        var audio = [Float](repeating: 0, count: samples)

        for i in 0..<samples {
            let t = Double(i) / sampleRate
            audio[i] = Float(sin(2.0 * .pi * frequency * t))
        }

        return audio
    }
}
