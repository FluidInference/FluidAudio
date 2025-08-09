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

    /// Export current speakers as Speaker objects
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

    /// Import speakers from Speaker objects
    public func importFromSpeakers(_ speakers: [Speaker]) {
        var profiles: [String: [Float]] = [:]
        for speaker in speakers {
            profiles[speaker.id] = speaker.embedding
        }
        initializeKnownSpeakers(profiles)
        Self.opsLogger.info("Imported \(speakers.count) speakers")
    }

    /// Verify if two embeddings are from the same speaker
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

    /// Find speakers in segments that match the target embedding
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

    /// Export speakers to JSON data
    public func exportToJSON() throws -> Data {
        let speakers = exportAsSpeakers()
        return try JSONEncoder().encode(speakers)
    }

    /// Import speakers from JSON data
    public func importFromJSON(_ data: Data) throws {
        let imported = try JSONDecoder().decode([Speaker].self, from: data)
        importFromSpeakers(imported)
    }

    /// Find similar speakers to a target embedding
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
