import Foundation
import OSLog

/// In-memory speaker database for streaming diarization
/// Tracks speakers across chunks and maintains consistent IDs
@available(macOS 13.0, iOS 16.0, *)
public class StreamingSpeakerManager {
    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "StreamingSpeakerManager")

    // Speaker database: ID -> representative embedding
    private var speakerDatabase: [String: SpeakerInfo] = [:]
    private var nextSpeakerId = 1
    private let queue = DispatchQueue(label: "speaker.manager.queue", attributes: .concurrent)

    // Configuration
    public var assignmentThreshold: Float
    public var updateThreshold: Float
    private let minDurationForNewSpeaker: Float

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
        assignmentThreshold: Float = 0.65,  // Max distance for assignment
        updateThreshold: Float = 0.45,  // Max distance for updating embedding
        minDurationForNewSpeaker: Float = 1.0
    ) {
        self.assignmentThreshold = assignmentThreshold
        self.updateThreshold = updateThreshold
        self.minDurationForNewSpeaker = minDurationForNewSpeaker
    }

    /// Assign or create speaker for given embedding
    public func assignSpeaker(
        embedding: [Float],
        duration: Float,
        confidence: Float = 1.0
    ) -> String? {
        guard !embedding.isEmpty && embedding.count == 256 else {
            logger.error("Invalid embedding size: \(embedding.count)")
            return nil
        }

        return queue.sync(flags: .barrier) {
            // Find closest existing speaker
            var minDistance: Float = Float.infinity
            var closestSpeakerId: String?

            for (speakerId, speakerInfo) in speakerDatabase {
                let distance = cosineDistance(embedding, speakerInfo.embedding)
                if distance < minDistance {
                    minDistance = distance
                    closestSpeakerId = speakerId
                }
            }

            // Decision logic
            if let speakerId = closestSpeakerId, minDistance < assignmentThreshold {
                // Match found - assign to existing speaker
                logger.debug("Matched to speaker \(speakerId) with distance \(minDistance)")

                // Update embedding if very close match and sufficient duration
                if minDistance < updateThreshold && duration >= minDurationForNewSpeaker {
                    updateSpeakerEmbedding(speakerId: speakerId, newEmbedding: embedding, duration: duration)
                }

                // Update last seen time
                speakerDatabase[speakerId]?.lastSeen = Date()
                speakerDatabase[speakerId]?.totalDuration += duration

                return speakerId
            } else if duration >= minDurationForNewSpeaker {
                // No match and sufficient duration - create new speaker
                let newSpeakerId = "Speaker_\(nextSpeakerId)"
                nextSpeakerId += 1

                speakerDatabase[newSpeakerId] = SpeakerInfo(
                    id: newSpeakerId,
                    embedding: embedding,
                    duration: duration
                )

                logger.info("Created new speaker \(newSpeakerId) (distance to closest: \(minDistance))")
                return newSpeakerId
            } else {
                // Duration too short for new speaker, but no match found
                logger.debug("Segment too short (\(duration)s) to create new speaker, distance: \(minDistance)")
                return nil
            }
        }
    }

    /// Update speaker embedding with weighted average
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

    /// Calculate cosine distance between two embeddings
    private func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
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

    /// Get current speaker count
    public var speakerCount: Int {
        queue.sync { speakerDatabase.count }
    }

    /// Get all speaker IDs
    public var speakerIds: [String] {
        queue.sync { Array(speakerDatabase.keys).sorted() }
    }

    /// Get speaker info
    public func getSpeakerInfo(for speakerId: String) -> SpeakerInfo? {
        queue.sync { speakerDatabase[speakerId] }
    }

    /// Clear all speakers
    public func reset() {
        queue.sync(flags: .barrier) {
            speakerDatabase.removeAll()
            nextSpeakerId = 1
            logger.info("Speaker database reset")
        }
    }

    /// Remove speakers not seen recently (for long streams)
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

    /// Get statistics about the speaker database
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
