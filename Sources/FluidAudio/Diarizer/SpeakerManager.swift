import Foundation
import OSLog

/// In-memory speaker database for streaming diarization
/// Tracks speakers across chunks and maintains consistent IDs
@available(macOS 13.0, iOS 16.0, *)
public class SpeakerManager {
    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "SpeakerManager")

    // Speaker database: ID -> representative embedding
    private var speakerDatabase: [String: SpeakerInfo] = [:]
    private var nextSpeakerId = 1
    private let queue = DispatchQueue(label: "speaker.manager.queue", attributes: .concurrent)

    // Configuration
    public var speakerThreshold: Float
    public var embeddingThreshold: Float
    private let minDuration: Float

    public struct SpeakerInfo {
        let id: String
        var embedding: [Float]
        var totalDuration: Float
        var lastSeen: Date
        var updateCount: Int

        init(id: String, embedding: [Float], duration: Float) {
            self.id = id
            self.embedding = embedding
            self.totalDuration = duration
            self.lastSeen = Date()
            self.updateCount = 1
        }
    }

    public init(
        speakerThreshold: Float = 0.65,
        embeddingThreshold: Float = 0.45,
        minDuration: Float = 1.0
    ) {
        self.speakerThreshold = speakerThreshold
        self.embeddingThreshold = embeddingThreshold
        self.minDuration = minDuration
    }

    public func initializeKnownSpeakers(_ speakerIdToEmbedding: [String: [Float]]) {
        queue.sync(flags: .barrier) {
            // Clear existing database
            speakerDatabase.removeAll()

            // Add known speakers
            for (speakerId, embedding) in speakerIdToEmbedding {
                guard embedding.count == 256 else {
                    logger.warning("Skipping speaker \(speakerId) - invalid embedding size: \(embedding.count)")
                    continue
                }

                speakerDatabase[speakerId] = SpeakerInfo(
                    id: speakerId,
                    embedding: embedding,
                    duration: 0
                )
                logger.info("Initialized known speaker: \(speakerId)")
            }

            // Reset speaker ID counter
            self.nextSpeakerId = 1

            logger.info("Initialized with \(self.speakerDatabase.count) known speakers")
        }
    }

    public func assignSpeaker(
        _ embedding: [Float],
        speechDuration: Float,
        confidence: Float = 1.0
    ) -> String? {
        guard !embedding.isEmpty && embedding.count == 256 else {
            logger.error("Invalid embedding size: \(embedding.count)")
            return nil
        }

        return queue.sync(flags: .barrier) {
            var minDistance: Float = Float.infinity
            var closestSpeakerId: String?

            for (speakerId, speakerInfo) in speakerDatabase {
                let distance = cosineDistance(embedding, speakerInfo.embedding)
                if distance < minDistance {
                    minDistance = distance
                    closestSpeakerId = speakerId
                }
            }

            if let speakerId = closestSpeakerId, minDistance < speakerThreshold {
                // Match found - assign to existing speaker
                logger.debug("Matched to speaker \(speakerId) with distance \(minDistance)")

                if minDistance < embeddingThreshold
                    && speechDuration >= minDuration
                {
                    updateSpeakerEmbedding(
                        speakerId: speakerId, newEmbedding: embedding, duration: speechDuration)
                }

                // Update last seen time
                speakerDatabase[speakerId]?.lastSeen = Date()
                speakerDatabase[speakerId]?.totalDuration += speechDuration

                return speakerId
            } else if speechDuration >= minDuration {
                let newSpeakerId = "Speaker_\(nextSpeakerId)"
                nextSpeakerId += 1

                speakerDatabase[newSpeakerId] = SpeakerInfo(
                    id: newSpeakerId,
                    embedding: embedding,
                    duration: speechDuration
                )

                logger.info("Created new speaker \(newSpeakerId) (distance to closest: \(minDistance))")
                return newSpeakerId
            } else {
                logger.debug(
                    "Segment too short (\(speechDuration)s) to create new speaker, distance: \(minDistance)")
                return nil
            }
        }
    }

    private func updateSpeakerEmbedding(speakerId: String, newEmbedding: [Float], duration: Float) {
        guard var speakerInfo = speakerDatabase[speakerId] else { return }

        // Weighted average based on duration
        let totalWeight = speakerInfo.totalDuration + duration
        let oldWeight = speakerInfo.totalDuration / totalWeight
        let newWeight = duration / totalWeight

        // Update embedding with weighted average
        for i in 0..<speakerInfo.embedding.count {
            speakerInfo.embedding[i] = speakerInfo.embedding[i] * oldWeight + newEmbedding[i] * newWeight
        }

        speakerInfo.updateCount += 1
        speakerDatabase[speakerId] = speakerInfo

        logger.debug("Updated embedding for \(speakerId), update count: \(speakerInfo.updateCount)")
    }

    internal func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else {
            return Float.infinity
        }

        var dotProduct: Float = 0
        var magnitudeA: Float = 0
        var magnitudeB: Float = 0

        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            magnitudeA += a[i] * a[i]
            magnitudeB += b[i] * b[i]
        }

        magnitudeA = sqrt(magnitudeA)
        magnitudeB = sqrt(magnitudeB)

        guard magnitudeA > 0, magnitudeB > 0 else {
            return Float.infinity
        }

        let similarity = dotProduct / (magnitudeA * magnitudeB)
        return 1.0 - similarity
    }

    public var speakerCount: Int {
        queue.sync { speakerDatabase.count }
    }

    public var speakerIds: [String] {
        queue.sync { Array(speakerDatabase.keys).sorted() }
    }

    /// Get detailed speaker information
    public func getAllSpeakerInfo() -> [String: SpeakerInfo] {
        queue.sync {
            return speakerDatabase
        }
    }

    public func getSpeakerInfo(for speakerId: String) -> SpeakerInfo? {
        queue.sync { speakerDatabase[speakerId] }
    }

    public func reset() {
        queue.sync(flags: .barrier) {
            speakerDatabase.removeAll()
            nextSpeakerId = 1
            logger.info("Speaker database reset")
        }
    }

    public func pruneInactiveSpeakers(olderThan timeInterval: TimeInterval = 300) {
        queue.sync(flags: .barrier) {
            let cutoffDate = Date().addingTimeInterval(-timeInterval)
            let inactiveSpeakers = speakerDatabase.filter { $0.value.lastSeen < cutoffDate }

            for (speakerId, _) in inactiveSpeakers {
                speakerDatabase.removeValue(forKey: speakerId)
                logger.info("Pruned inactive speaker \(speakerId)")
            }
        }
    }

    public func getStatistics() -> String {
        queue.sync {
            let totalSpeakers = speakerDatabase.count
            let totalDuration = speakerDatabase.values.reduce(0) { $0 + $1.totalDuration }
            let avgUpdates = speakerDatabase.values.reduce(0) { $0 + $1.updateCount } / max(1, totalSpeakers)

            return """
                Speakers: \(totalSpeakers)
                Total Duration: \(String(format: "%.1f", totalDuration))s
                Avg Updates: \(avgUpdates)
                """
        }
    }
}
