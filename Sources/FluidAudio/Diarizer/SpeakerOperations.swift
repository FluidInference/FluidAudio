import CoreML
import Foundation
import OSLog

/// Single speaker representation that can be either a profile or a segment
@available(macOS 13.0, iOS 16.0, *)
public struct Speaker: Identifiable, Codable {
    public let id: String
    public var name: String
    public var embedding: [Float]

    public var startTime: Double?
    public var endTime: Double?
    public var confidence: Float = 1.0
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

// MARK: - SpeakerManager Extensions

@available(macOS 13.0, iOS 16.0, *)
extension SpeakerManager {

    private static let opsLogger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "SpeakerOps")

    /// Export current speakers as Speaker objects for persistence or external processing.
    ///
    /// Use cases:
    /// - Saving speaker profiles to disk for later sessions
    /// - Transferring speaker data between different components
    /// - Creating speaker databases for known participants
    ///
    /// Example:
    /// ```swift
    /// let speakers = speakerManager.exportAsSpeakers()
    /// let jsonData = try JSONEncoder().encode(speakers)
    /// try jsonData.write(to: profilesURL)
    /// ```
    public func exportAsSpeakers() -> [Speaker] {
        var speakers: [Speaker] = []
        for speakerId in speakerIds {
            if let info = getSpeakerInfo(for: speakerId) {
                var speaker = Speaker(
                    id: speakerId,
                    name: speakerId,
                    embedding: info.embedding
                )
                speaker.totalDuration = Double(info.totalDuration)
                speaker.lastSeen = info.lastSeen
                speakers.append(speaker)
            }
        }
        return speakers
    }

    /// Import speakers from Speaker objects to restore previous sessions.
    ///
    /// Use cases:
    /// - Loading known speaker profiles at startup
    /// - Transferring speakers between different DiarizerManager instances
    /// - Implementing speaker enrollment systems
    ///
    /// Example:
    /// ```swift
    /// let jsonData = try Data(contentsOf: profilesURL)
    /// let speakers = try JSONDecoder().decode([Speaker].self, from: jsonData)
    /// speakerManager.importFromSpeakers(speakers)
    /// ```
    public func importFromSpeakers(_ speakers: [Speaker]) {
        var profiles: [String: [Float]] = [:]
        for speaker in speakers {
            profiles[speaker.id] = speaker.embedding
        }
        initializeKnownSpeakers(profiles)
        Self.opsLogger.info("Imported \(speakers.count) speakers")
    }

    /// Verify if two embeddings are from the same speaker.
    ///
    /// Use cases:
    /// - Speaker verification for security applications
    /// - Confirming speaker identity in multi-session recordings
    /// - Quality assurance for speaker segmentation
    ///
    /// - Returns: Tuple with (isSame: whether speakers match, confidence: 0.0-1.0)
    public func verifySameSpeaker(
        embedding1: [Float],
        embedding2: [Float],
        threshold: Float = 0.7
    ) -> (isSame: Bool, confidence: Float) {
        let distance = cosineDistance(embedding1, embedding2)
        let isSame = distance < threshold
        let confidence = 1.0 - distance
        return (isSame, confidence)
    }

    /// Find speakers in segments that match the target embedding.
    ///
    /// Use cases:
    /// - Searching for a specific speaker in a recording
    /// - Identifying when a known speaker spoke
    /// - Cross-session speaker tracking
    ///
    /// - Parameters:
    ///   - targetEmbedding: The speaker embedding to search for
    ///   - segments: Array of segments to search through
    ///   - threshold: Similarity threshold (0.0-1.0, lower = more similar)
    public func findSpeaker(
        targetEmbedding: [Float],
        in segments: [TimedSpeakerSegment],
        threshold: Float = 0.65
    ) -> [Speaker] {
        var speakerGroups: [String: [TimedSpeakerSegment]] = [:]
        for segment in segments {
            speakerGroups[segment.speakerId, default: []].append(segment)
        }

        var matches: [Speaker] = []

        for (speakerId, segments) in speakerGroups {
            guard let firstEmbedding = segments.first?.embedding else { continue }

            let distance = cosineDistance(targetEmbedding, firstEmbedding)

            if distance < threshold {
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

    /// Export speakers to JSON data for persistence.
    ///
    /// Use cases:
    /// - Saving speaker profiles to disk
    /// - Sending speaker data over network
    /// - Creating backups of speaker databases
    public func exportToJSON() throws -> Data {
        let speakers = exportAsSpeakers()
        return try JSONEncoder().encode(speakers)
    }

    /// Import speakers from JSON data.
    ///
    /// Use cases:
    /// - Restoring speaker profiles from disk
    /// - Receiving speaker data from network
    /// - Loading pre-trained speaker profiles
    public func importFromJSON(_ data: Data) throws {
        let imported = try JSONDecoder().decode([Speaker].self, from: data)
        importFromSpeakers(imported)
    }

    /// Find similar speakers to a target embedding, ranked by similarity.
    ///
    /// Use cases:
    /// - Speaker identification from a database
    /// - Finding the most likely speaker match
    /// - Speaker clustering analysis
    ///
    /// - Parameters:
    ///   - embedding: The target speaker embedding
    ///   - limit: Maximum number of results to return
    /// - Returns: Array of (speaker, distance) tuples sorted by similarity
    public func findSimilarSpeakers(
        to embedding: [Float],
        limit: Int = 5
    ) -> [(speaker: Speaker, distance: Float)] {
        var results: [(Speaker, Float)] = []

        for speakerId in speakerIds {
            if let info = getSpeakerInfo(for: speakerId) {
                let distance = cosineDistance(embedding, info.embedding)
                var speaker = Speaker(
                    id: speakerId,
                    name: speakerId,
                    embedding: info.embedding
                )
                speaker.totalDuration = Double(info.totalDuration)
                speaker.lastSeen = info.lastSeen
                results.append((speaker, distance))
            }
        }

        return
            results
            .sorted { $0.1 < $1.1 }
            .prefix(limit)
            .map { ($0.0, $0.1) }
    }
}
