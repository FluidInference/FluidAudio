import CoreML
import Foundation
import OSLog

// MARK: - Speaker Schema

/// Single speaker representation that can be either a profile or a segment
@available(macOS 13.0, iOS 16.0, *)
public struct Speaker: Identifiable, Codable {
    public let id: String
    public var name: String
    public var embedding: [Float]

    // Optional timing (when used as a segment)
    public var startTime: Double?
    public var endTime: Double?
    public var confidence: Float = 1.0

    // Statistics (when used as a profile)
    public var totalDuration: Double = 0
    public var lastSeen: Date = Date()

    public var duration: Double? {
        guard let start = startTime, let end = endTime else { return nil }
        return end - start
    }

    public init(
        id: String? = nil,
        name: String? = nil,
        embedding: [Float],
        startTime: Double? = nil,
        endTime: Double? = nil,
        confidence: Float = 1.0
    ) {
        self.id = id ?? "Speaker_\(UUID().uuidString.prefix(8))"
        self.name = name ?? self.id
        self.embedding = embedding
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
    }
}

// MARK: - Speaker Operations

/// All speaker-related operations in one place
@available(macOS 13.0, iOS 16.0, *)
public class SpeakerOperations {

    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "SpeakerOps")
    private let diarizerManager: DiarizerManager

    // Simple speaker database
    private var speakers: [String: Speaker] = [:]
    private let queue = DispatchQueue(label: "speaker.ops", attributes: .concurrent)

    public init(diarizerManager: DiarizerManager) {
        self.diarizerManager = diarizerManager
    }

    // MARK: - Basic Operations

    /// Add a speaker
    public func addSpeaker(name: String, embedding: [Float]) -> String {
        let speaker = Speaker(name: name, embedding: embedding)

        queue.sync(flags: .barrier) {
            speakers[speaker.id] = speaker
        }

        logger.info("Added speaker: \(name)")
        return speaker.id
    }

    /// Rename a speaker
    public func renameSpeaker(id: String, newName: String) {
        queue.sync(flags: .barrier) {
            speakers[id]?.name = newName
        }
        logger.info("Renamed speaker \(id) to \(newName)")
    }

    /// Delete a speaker
    public func deleteSpeaker(id: String) {
        queue.sync(flags: .barrier) {
            speakers.removeValue(forKey: id)
        }
        logger.info("Deleted speaker \(id)")
    }

    /// Get all speakers
    public func getAllSpeakers() -> [Speaker] {
        queue.sync {
            Array(speakers.values)
        }
    }

    /// Clear all speakers
    public func clearAll() {
        queue.sync(flags: .barrier) {
            speakers.removeAll()
        }
        logger.info("Cleared all speakers")
    }

    // MARK: - Verification

    /// Check if two audio samples are from the same speaker
    public func verifySameSpeaker(
        audio1: [Float],
        audio2: [Float],
        threshold: Float = 0.7
    ) async throws -> (isSame: Bool, confidence: Float) {

        // Get embeddings from audio
        let result1 = try diarizerManager.performCompleteDiarization(audio1)
        let result2 = try diarizerManager.performCompleteDiarization(audio2)

        guard let embedding1 = result1.segments.first?.embedding,
            let embedding2 = result2.segments.first?.embedding
        else {
            throw DiarizerError.embeddingExtractionFailed
        }

        // Calculate distance
        let distance = diarizerManager.cosineDistance(embedding1, embedding2)
        let isSame = distance < threshold
        let confidence = isSame ? (1.0 - distance) : distance

        return (isSame, confidence)
    }

    // MARK: - Search

    /// Find a speaker in audio
    public func findSpeaker(
        targetEmbedding: [Float],
        in audio: [Float],
        threshold: Float = 0.65
    ) async throws -> [Speaker] {

        // Process audio
        let result = try diarizerManager.performCompleteDiarization(audio)

        // Group segments by speaker
        var speakerGroups: [String: [TimedSpeakerSegment]] = [:]
        for segment in result.segments {
            speakerGroups[segment.speakerId, default: []].append(segment)
        }

        // Find matching speaker
        var matches: [Speaker] = []

        for (speakerId, segments) in speakerGroups {
            // Get average embedding for this speaker
            guard let firstEmbedding = segments.first?.embedding else { continue }

            let distance = diarizerManager.cosineDistance(targetEmbedding, firstEmbedding)

            if distance < threshold {
                // Convert to speaker segments
                for segment in segments {
                    matches.append(
                        Speaker(
                            id: speakerId,
                            embedding: firstEmbedding,
                            startTime: Double(segment.startTimeSeconds),
                            endTime: Double(segment.endTimeSeconds),
                            confidence: 1.0 - distance
                        ))
                }
            }
        }

        return matches
    }

    // MARK: - Streaming Support

    /// Assign speaker for streaming (maintains consistent IDs across chunks)
    public func assignStreamingSpeaker(
        embedding: [Float],
        duration: Float,
        threshold: Float = 0.7
    ) -> String {

        return queue.sync(flags: .barrier) {
            // Find closest existing speaker
            var minDistance: Float = Float.infinity
            var closestId: String?

            for (id, speaker) in speakers {
                let distance = diarizerManager.cosineDistance(embedding, speaker.embedding)
                if distance < minDistance {
                    minDistance = distance
                    closestId = id
                }
            }

            // Assign or create
            if let id = closestId, minDistance < threshold {
                // Update existing speaker
                speakers[id]?.totalDuration += Double(duration)
                speakers[id]?.lastSeen = Date()
                return id
            } else {
                // Create new speaker
                let speaker = Speaker(embedding: embedding)
                speakers[speaker.id] = speaker
                return speaker.id
            }
        }
    }

    // MARK: - Import/Export

    /// Export speakers to JSON
    public func exportToJSON() throws -> Data {
        let speakerList = getAllSpeakers()
        return try JSONEncoder().encode(speakerList)
    }

    /// Import speakers from JSON
    public func importFromJSON(_ data: Data) throws {
        let imported = try JSONDecoder().decode([Speaker].self, from: data)

        queue.sync(flags: .barrier) {
            for speaker in imported {
                speakers[speaker.id] = speaker
            }
        }

        logger.info("Imported \(imported.count) speakers")
    }

    // MARK: - Utilities

    /// Find similar speakers to an embedding
    public func findSimilarSpeakers(
        to embedding: [Float],
        limit: Int = 5
    ) -> [(speaker: Speaker, distance: Float)] {

        return queue.sync {
            var results: [(Speaker, Float)] = []

            for speaker in speakers.values {
                let distance = diarizerManager.cosineDistance(embedding, speaker.embedding)
                results.append((speaker, distance))
            }

            return
                results
                .sorted { $0.1 < $1.1 }
                .prefix(limit)
                .map { ($0.0, $0.1) }
        }
    }

    /// Merge two speakers
    public func mergeSpeakers(sourceId: String, targetId: String) {
        queue.sync(flags: .barrier) {
            guard let source = speakers[sourceId],
                var target = speakers[targetId]
            else { return }

            // Merge duration
            target.totalDuration += source.totalDuration

            // Update target
            speakers[targetId] = target

            // Remove source
            speakers.removeValue(forKey: sourceId)
        }

        logger.info("Merged \(sourceId) into \(targetId)")
    }
}
