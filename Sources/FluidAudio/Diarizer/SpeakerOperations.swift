import Foundation
import OSLog

/// These functions provide additional capabilities beyond core diarization
@available(macOS 13.0, iOS 16.0, *)
extension SpeakerManager {

    /// Add a historical embedding to a speaker and update the main embedding.
    ///
    /// - Parameters:
    ///   - embedding: The historical embedding to add
    ///   - speakerId: The speaker ID to add the embedding to
    /// - Returns: Updated Speaker if successful, nil if speaker not found
    @discardableResult
    public func addHistoricalEmbedding(
        _ embedding: HistoricalEmbedding,
        to speakerId: String
    ) -> Speaker? {
        return queue.sync(flags: .barrier) {
            guard var speaker = speakerDatabase[speakerId] else {
                logger.warning("Speaker \(speakerId) not found")
                return nil
            }

            // Validate embedding quality
            let embeddingMagnitude = sqrt(embedding.embedding.map { $0 * $0 }.reduce(0, +))
            guard embeddingMagnitude > 0.1 else {
                logger.debug("Skipping low-quality embedding for speaker \(speakerId)")
                return nil
            }

            // FIFO management - max 50 embeddings
            if speaker.historicalEmbeddings.count >= 50 {
                speaker.historicalEmbeddings.removeFirst()
            }

            speaker.historicalEmbeddings.append(embedding)

            // Always update main embedding using exponential moving average
            let alpha: Float = 0.9
            for i in 0..<speaker.mainEmbedding.count {
                speaker.mainEmbedding[i] = alpha * speaker.mainEmbedding[i] + (1 - alpha) * embedding.embedding[i]
            }
            speaker.updateCount += 1
            speaker.updatedAt = Date()

            speakerDatabase[speakerId] = speaker

            logger.debug(
                "Added historical embedding to speaker \(speakerId), total: \(speaker.historicalEmbeddings.count)")

            // Return the updated Speaker object
            return Speaker(
                id: speaker.id,
                name: "Speaker \(speaker.id)",
                mainEmbedding: speaker.mainEmbedding,
                duration: speaker.duration,
                createdAt: speaker.createdAt,
                updatedAt: speaker.updatedAt
            )
        }
    }

    /// Remove a historical embedding from a speaker and recalculate the main embedding.
    ///
    /// Useful for speaker reassignment scenarios where a segment needs to be moved to a different speaker.
    ///
    /// - Parameters:
    ///   - segmentId: The segment ID of the historical embedding to remove
    ///   - speakerId: The speaker ID to remove the embedding from
    /// - Returns: The removed historical embedding if found, nil otherwise
    @discardableResult
    public func removeHistoricalEmbedding(
        segmentId: UUID,
        from speakerId: String
    ) -> HistoricalEmbedding? {
        return queue.sync(flags: .barrier) {
            guard var speaker = speakerDatabase[speakerId] else {
                logger.warning("Speaker \(speakerId) not found")
                return nil
            }

            let beforeCount = speaker.historicalEmbeddings.count

            if let index = speaker.historicalEmbeddings.firstIndex(where: { $0.segmentId == segmentId }) {
                let removed = speaker.historicalEmbeddings.remove(at: index)
                speakerDatabase[speakerId] = speaker

                logger.info(
                    "✅ Speaker \(speakerId): removed historical embedding (before: \(beforeCount), after: \(speaker.historicalEmbeddings.count))"
                )

                recalculateMainEmbedding(for: speakerId)

                return removed
            }

            logger.info("❌ Speaker \(speakerId): historical embedding not found for segment \(segmentId)")
            return nil
        }
    }

    /// Reassign a historical embedding from one speaker to another.
    ///
    /// This moves a segment's embedding from one speaker to another, updating both speakers'
    /// main embeddings. Useful for correcting misclassified segments.
    ///
    /// - Parameters:
    ///   - segmentId: The segment ID to reassign
    ///   - fromSpeakerId: The current speaker ID
    ///   - toSpeakerId: The target speaker ID
    /// - Returns: True if successful, false if segment not found or speakers don't exist
    @discardableResult
    public func reassignSegment(
        segmentId: UUID,
        from fromSpeakerId: String,
        to toSpeakerId: String
    ) -> Bool {
        // Remove from original speaker
        guard let embedding = removeHistoricalEmbedding(segmentId: segmentId, from: fromSpeakerId) else {
            logger.warning("Failed to remove segment \(segmentId) from speaker \(fromSpeakerId)")
            return false
        }

        // Add to new speaker
        let updatedSpeaker = addHistoricalEmbedding(embedding, to: toSpeakerId)

        if updatedSpeaker != nil {
            logger.info("✅ Reassigned segment \(segmentId) from \(fromSpeakerId) to \(toSpeakerId)")
        } else {
            logger.error("Failed to add segment \(segmentId) to speaker \(toSpeakerId)")
            // Try to restore to original speaker
            _ = addHistoricalEmbedding(embedding, to: fromSpeakerId)
        }

        return updatedSpeaker != nil
    }

    // MARK: - Speaker Recalculation

    /// Recalculate the main embedding from historical embeddings.
    ///
    /// This method averages all historical embeddings to create a new main embedding.
    /// Useful for recovering from corrupted main embeddings or updating after significant changes.
    ///
    /// - Parameter speakerId: The speaker ID to recalculate
    /// - Returns: Updated Speaker if successful, nil if speaker not found or no historical embeddings
    @discardableResult
    public func recalculateMainEmbedding(for speakerId: String) -> Speaker? {
        return queue.sync(flags: .barrier) {
            guard var speaker = speakerDatabase[speakerId] else {
                logger.warning("Speaker \(speakerId) not found")
                return nil
            }

            let embeddings = speaker.historicalEmbeddings

            guard !embeddings.isEmpty else {
                logger.error("No historical embeddings for speaker \(speakerId)")
                return nil
            }

            guard let firstEmbedding = embeddings.first, !firstEmbedding.embedding.isEmpty else {
                logger.error("First historical embedding is empty for speaker \(speakerId)")
                return nil
            }

            let beforeFirst10 = Array(speaker.mainEmbedding.prefix(10))
            logger.info("Recalculating main embedding for \(speakerId), before: \(beforeFirst10)")

            let embeddingSize = firstEmbedding.embedding.count

            // Calculate average of all historical embeddings
            var averageEmbedding = [Float](repeating: 0.0, count: embeddingSize)

            var validEmbeddingCount = 0
            for historical in embeddings {
                if historical.embedding.count != embeddingSize {
                    logger.warning(
                        "Skipping historical embedding with wrong size: \(historical.embedding.count) != \(embeddingSize)"
                    )
                    continue
                }

                // Add to average
                for i in 0..<embeddingSize {
                    averageEmbedding[i] += historical.embedding[i]
                }
                validEmbeddingCount += 1
            }

            guard validEmbeddingCount > 0 else {
                logger.error("No valid historical embeddings found for speaker \(speakerId)")
                return nil
            }

            // Divide by count to get average
            let count = Float(validEmbeddingCount)
            for i in 0..<embeddingSize {
                averageEmbedding[i] /= count
            }

            if averageEmbedding.isEmpty {
                logger.error("Resulting average embedding is empty for speaker \(speakerId)")
                return nil
            }

            speaker.mainEmbedding = averageEmbedding
            speaker.updatedAt = Date()
            speakerDatabase[speakerId] = speaker

            let afterFirst10 = Array(speaker.mainEmbedding.prefix(10))
            logger.info("Recalculated main embedding for \(speakerId), after: \(afterFirst10)")
            logger.info("Used \(validEmbeddingCount) historical embeddings for recalculation")

            // Return the updated Speaker object
            return Speaker(
                id: speaker.id,
                name: "Speaker \(speaker.id)",
                mainEmbedding: speaker.mainEmbedding,
                duration: speaker.duration,
                createdAt: speaker.createdAt,
                updatedAt: speaker.updatedAt
            )
        }
    }

    // MARK: - Speaker Query Operations

    /// Get the names/IDs of all current speakers.
    ///
    /// Returns a list of speaker identifiers from the in-memory database.
    /// Useful for getting a quick overview of detected speakers.
    ///
    /// - Returns: Array of speaker IDs/names currently in the database
    public func getCurrentSpeakerNames() -> [String] {
        return queue.sync {
            return Array(speakerDatabase.keys).sorted()
        }
    }

    /// Get global speaker statistics from the in-memory database.
    ///
    /// Returns comprehensive statistics about all speakers including count,
    /// total duration, and quality metrics.
    ///
    /// - Returns: Tuple containing (totalSpeakers, totalDuration, averageConfidence, speakersWithHistory)
    public func getGlobalSpeakerStats() -> (
        totalSpeakers: Int,
        totalDuration: Float,
        averageConfidence: Float,
        speakersWithHistory: Int
    ) {
        return queue.sync { () -> (Int, Float, Float, Int) in
            let speakers = Array(speakerDatabase.values)

            guard !speakers.isEmpty else {
                return (0, 0, 0, 0)
            }

            let totalDuration = speakers.reduce(0) { $0 + $1.duration }
            let totalUpdates = speakers.reduce(0) { $0 + $1.updateCount }
            let averageConfidence = Float(totalUpdates) / Float(speakers.count) / 10.0  // Normalize
            let speakersWithHistory = speakers.filter { !$0.historicalEmbeddings.isEmpty }.count

            logger.info(
                "Global stats - Speakers: \(speakers.count), Duration: \(String(format: "%.1f", totalDuration))s, Avg confidence: \(String(format: "%.2f", averageConfidence)), With history: \(speakersWithHistory)"
            )

            return (speakers.count, totalDuration, min(1.0, averageConfidence), speakersWithHistory)
        }
    }
}
