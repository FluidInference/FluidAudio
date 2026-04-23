import Accelerate
import Foundation
import OSLog

/// In-memory speaker database for streaming diarization
/// Tracks speakers across chunks and maintains consistent IDs
///
/// This is an `actor` to provide thread-safe access under Swift 6 strict
/// concurrency. Previous implementations used a `DispatchQueue` which could
/// trigger `unsafeForcedSync` warnings (and occasional libmalloc heap
/// corruption via concurrent `[Float]` COW mutations) when called from
/// async contexts such as VAD / diarization buffer callbacks.
public actor SpeakerManager {
    internal let logger = AppLogger(category: "SpeakerManager")

    // Constants
    public static let embeddingSize = 256  // Standard embedding dimension for speaker models

    // Speaker database: ID -> Speaker
    internal var speakerDatabase: [String: Speaker] = [:]
    private var nextSpeakerId = 1

    // Track the highest speaker ID to ensure uniqueness
    private var highestSpeakerId = 0

    public var speakerThreshold: Float  // Max distance for speaker assignment (default: 0.65)
    public var embeddingThreshold: Float  // Max distance for updating embeddings (default: 0.45)
    public var minSpeechDuration: Float  // Min duration to create speaker (default: 1.0)
    public var minEmbeddingUpdateDuration: Float  // Min duration to update embeddings (default: 2.0)

    public init(
        speakerThreshold: Float = 0.65,
        embeddingThreshold: Float = 0.45,
        minSpeechDuration: Float = 1.0,
        minEmbeddingUpdateDuration: Float = 2.0
    ) {
        self.speakerThreshold = speakerThreshold
        self.embeddingThreshold = embeddingThreshold
        self.minSpeechDuration = minSpeechDuration
        self.minEmbeddingUpdateDuration = minEmbeddingUpdateDuration
    }

    // MARK: - Threshold setters (actor-isolated)

    public func setSpeakerThreshold(_ value: Float) { self.speakerThreshold = value }
    public func setEmbeddingThreshold(_ value: Float) { self.embeddingThreshold = value }
    public func setMinSpeechDuration(_ value: Float) { self.minSpeechDuration = value }
    public func setMinEmbeddingUpdateDuration(_ value: Float) { self.minEmbeddingUpdateDuration = value }

    /// Add known speakers to the database
    /// - Parameters:
    ///   - speakers: Array of `Speaker`s to add
    ///   - mode: Mode for handling overlapping ID conflicts.
    ///   - preservePermanent: Whether to avoid overwriting/merging pre-existing permanent speakers
    public func initializeKnownSpeakers(
        _ speakers: [Speaker], mode: SpeakerInitializationMode = .skip, preserveIfPermanent: Bool = true
    ) {
        if mode == .reset {
            self.reset(keepIfPermanent: preserveIfPermanent)
        }

        var maxNumericId = 0

        for speaker in speakers {
            guard speaker.currentEmbedding.count == Self.embeddingSize else {
                logger.warning(
                    "Skipping speaker \(speaker.id) - invalid embedding size: \(speaker.currentEmbedding.count)")
                continue
            }

            // Check if the speaker ID is already initialized
            if let oldSpeaker = self.speakerDatabase[speaker.id] {
                // Handle duplicate speaker
                switch mode {
                case .reset, .overwrite:
                    if !(oldSpeaker.isPermanent && preserveIfPermanent) {
                        logger.warning("Speaker \(speaker.id) is already initialized. Overwriting old speaker.")
                        speakerDatabase[speaker.id] = speaker
                    } else {
                        logger.warning(
                            "Failed to overwrite Speaker \(speaker.id) because it is permanent. Skipping")
                        continue
                    }
                case .merge:
                    if !(oldSpeaker.isPermanent && preserveIfPermanent) {
                        logger.warning("Speaker \(speaker.id) is already initialized. Merging with old speaker.")
                        speakerDatabase[speaker.id]?.mergeWith(speaker, keepName: speaker.name)
                    } else {
                        logger.warning(
                            "Failed to merge Speaker \(speaker.id) into Speaker \(oldSpeaker.id) because the existing speaker is permanent. Skipping"
                        )
                        continue
                    }
                case .skip:
                    logger.warning("Speaker \(speaker.id) is already initialized. Skipping new speaker.")
                    continue
                }
            } else {
                speakerDatabase[speaker.id] = speaker
            }

            // Try to extract numeric ID if it's a pure number
            if let numericId = Int(speaker.id) {
                maxNumericId = max(maxNumericId, numericId)
            }

            logger.info(
                "Initialized known speaker: \(speaker.id) with \(speaker.rawEmbeddings.count) raw embeddings"
            )
        }

        self.highestSpeakerId = maxNumericId
        self.nextSpeakerId = maxNumericId + 1

        logger.info(
            "Initialized with \(self.speakerDatabase.count) known speakers, next ID will be: \(self.nextSpeakerId)"
        )
    }

    /// Match the embedding to the closest existing speaker if sufficiently similar or create a new one if not.
    /// - Parameters:
    ///    - embedding: 256D speaker embedding vector
    ///    - speechDuration: Duration of the speech segment during which this speaker was active
    ///    - confidence: Confidence in the embedding vector being correct
    ///    - speakerThreshold: The maximum cosine distance to an existing speaker to create a new one (uses the default threshold for this `SpeakerManager` object if none is provided)
    ///    - newName: Name to assign the speaker if a new one is created (default: `Speaker $id`)
    ///  - Returns: A `Speaker` object if a match was found or a new one was created. Returns `nil` if an error occurred.
    public func assignSpeaker(
        _ embedding: [Float],
        speechDuration: Float,
        confidence: Float = 1.0,
        speakerThreshold: Float? = nil,
        newName: String? = nil
    ) -> Speaker? {
        guard !embedding.isEmpty && embedding.count == Self.embeddingSize else {
            logger.error("Invalid embedding size: \(embedding.count)")
            return nil
        }

        let normalizedEmbedding = VDSPOperations.l2Normalize(embedding)
        let speakerThreshold = speakerThreshold ?? self.speakerThreshold

        let (closestSpeaker, distance) = findClosestSpeaker(to: normalizedEmbedding)

        if let speakerId = closestSpeaker, distance < speakerThreshold {
            updateExistingSpeaker(
                speakerId: speakerId,
                embedding: normalizedEmbedding,
                duration: speechDuration,
                distance: distance
            )

            return speakerDatabase[speakerId]
        }

        // Step 3: Create new speaker if duration is sufficient
        if speechDuration >= minSpeechDuration {
            let newSpeakerId = createNewSpeaker(
                embedding: normalizedEmbedding,
                duration: speechDuration,
                distanceToClosest: distance
            )

            return speakerDatabase[newSpeakerId]
        }

        // Step 4: Audio segment too short
        logger.debug("Audio segment too short (\(speechDuration)s) to create new speaker")
        return nil
    }

    /// Find the closest existing speaker to an embedding, up to a maximum cosine distance of `speakerThreshold`.
    /// - Parameters:
    ///    - embedding: 256D speaker embedding vector
    ///    - speakerThreshold: Maximum cosine distance to an existing speaker to create a new one (uses the default threshold for this `SpeakerManager` object if none is provided)
    ///  - Returns: ID of the match (if found) and the distance to that match.
    public func findSpeaker(with embedding: [Float], speakerThreshold: Float? = nil) -> (id: String?, distance: Float) {
        let (closestSpeakerId, minDistance) = findClosestSpeaker(to: embedding)
        let speakerThreshold = speakerThreshold ?? self.speakerThreshold
        if let closestSpeakerId, minDistance <= speakerThreshold {
            return (closestSpeakerId, minDistance)
        }
        return (nil, .infinity)
    }

    /// Find the closest existing speaker to an embedding, up to a maximum cosine distance of `speakerThreshold`.
    /// - Parameters:
    ///    - embedding: 256D speaker embedding vector
    ///    - speakerThreshold: Maximum cosine distance between `embedding` and another speaker for them to be a match (default: `self.speakerThreshold`)
    ///  - Returns: Array of the `maxCount` nearest speakers and the distances to them from `embedding`, sorted by ascending cosine distances (from closest to farthest).
    public func findMatchingSpeakers(
        with embedding: [Float], speakerThreshold: Float? = nil
    ) -> [(id: String, distance: Float)] {
        var matches: [(id: String, distance: Float)] = []
        let speakerThreshold = speakerThreshold ?? self.speakerThreshold

        for (speakerId, speaker) in speakerDatabase {
            let distance = cosineDistance(embedding, speaker.currentEmbedding)
            if distance <= speakerThreshold {
                matches.append((speakerId, distance))
            }
        }
        matches.sort { $0.distance < $1.distance }
        return matches
    }

    /// Find all speakers that meet a certain predicate
    /// - Parameter predicate: Condition the speakers must meet to be returned
    /// - Returns: A list of all Speaker IDs corresponding to Speakers that meet the predicate
    public func findSpeakers(where predicate: @Sendable (Speaker) -> Bool) -> [String] {
        return speakerDatabase.filter { predicate($0.value) }.map(\.key)
    }

    /// Mark a speaker as permanent
    /// - Parameter speakerId: ID of the speaker to mark as permanent
    public func makeSpeakerPermanent(_ speakerId: String) {
        guard speakerDatabase[speakerId] != nil else {
            logger.warning("Failed to mark speaker \(speakerId) as permanent (speaker not found).")
            return
        }
        logger.info("Marking speaker \(speakerId) as permanent.")
        speakerDatabase[speakerId]?.isPermanent = true
    }

    /// Remove a speaker's permanent marker
    /// - Parameter speakerId: ID of the speaker from which to remove the permanent marker
    public func revokePermanence(from speakerId: String) {
        guard speakerDatabase[speakerId] != nil else {
            logger.warning("Failed to revoke permanence from speaker \(speakerId) (speaker not found).")
            return
        }

        logger.info("Revoking permanence from speaker \(speakerId).")
        speakerDatabase[speakerId]?.isPermanent = false
    }

    /// Merge two speakers in the database.
    /// - Parameters:
    ///   - sourceId: ID of the `Speaker` being merged
    ///   - destinationId: ID of the `Speaker` that absorbs the other one
    ///   - mergedName: New name for the merged speaker (uses `destination`'s name if not provided)
    ///   - stopIfPermanent: Whether to stop merging if the source speaker is permanent
    public func mergeSpeaker(
        _ sourceId: String, into destinationId: String, mergedName: String? = nil, stopIfPermanent: Bool = true
    ) {
        // don't merge a speaker into itself
        guard sourceId != destinationId else {
            return
        }

        // ensure both speakers exist
        guard let speakerToMerge = speakerDatabase[sourceId],
            speakerDatabase[destinationId] != nil
        else {
            return
        }

        // don't merge permanent speakers into another one
        guard !(stopIfPermanent && speakerToMerge.isPermanent) else {
            return
        }

        // merge source into destination
        speakerDatabase[destinationId]?.mergeWith(speakerToMerge, keepName: mergedName)

        // remove source speaker
        speakerDatabase.removeValue(forKey: sourceId)
    }

    /// Find all pairs of speakers that can be merged
    /// - Parameters:
    ///    - speakerThreshold: Max cosine distance between speakers to let them be considered mergeable
    ///    - excludeIfBothPermanent: Whether to exclude speaker pairs where both speakers are permanent
    /// - Returns: Array of speaker ID pairs that belong to speakers that are similar enough to be merged
    public func findMergeablePairs(
        speakerThreshold: Float? = nil, excludeIfBothPermanent: Bool = true
    ) -> [(speakerToMerge: String, destination: String)] {
        let speakerThreshold = speakerThreshold ?? self.speakerThreshold
        var pairs: [(speakerToMerge: String, destination: String)] = []
        let ids = Array(speakerDatabase.keys)
        let count = speakerDatabase.count

        for i in (0..<count) {
            // get speaker 1
            guard let speaker1 = speakerDatabase[ids[i]] else {
                logger.error("ID \(ids[i]) not found in speakerDatabase")
                continue
            }

            for j in (i + 1)..<count {
                // get speaker 2
                guard let speaker2 = speakerDatabase[ids[j]] else {
                    logger.error("ID \(ids[j]) not found in speakerDatabase")
                    continue
                }

                // skip double permanent pairs
                if excludeIfBothPermanent && speaker1.isPermanent && speaker2.isPermanent {
                    logger.info(
                        "findMergeablePairs: Skipping \(speaker1.id) and \(speaker2.id) as both are permanent")
                    continue
                }

                // determine if they are similar enough
                let distance = cosineDistance(speaker1.currentEmbedding, speaker2.currentEmbedding)

                guard distance < speakerThreshold else {
                    continue
                }

                // prioritize putting speaker1 as the destination for consistency
                if !speaker2.isPermanent {
                    pairs.append((speakerToMerge: speaker2.id, destination: speaker1.id))
                } else {
                    pairs.append((speakerToMerge: speaker1.id, destination: speaker2.id))
                }
            }
        }

        return pairs
    }

    /// Remove a speaker from the database
    /// - Parameters:
    ///   - speakerID: ID of the speaker being removed
    ///   - keepIfPermanent: Whether to stop the removal if the speaker is marked as permanent
    public func removeSpeaker(_ speakerID: String, keepIfPermanent: Bool = true) {
        // determine if we should skip the removal due to permanence
        if keepIfPermanent, let speaker = self.speakerDatabase[speakerID], speaker.isPermanent {
            logger.warning("Failed to remove speaker: \(speakerID) (Speaker is permanent)")
            return
        }

        // attempt to remove the speaker
        if let _ = speakerDatabase.removeValue(forKey: speakerID) {
            logger.info("Removing speaker: \(speakerID)")
        } else {
            logger.warning("Failed to remove speaker: \(speakerID) (Speaker not found)")
        }
    }

    /// Remove all speakers that were inactive since a given `date`
    /// - Parameters:
    ///   - date: Speakers who have not been active after this date will be removed.
    ///   - keepIfPermanent: Whether to stop the removal if the speaker is marked as permanent
    public func removeSpeakersInactive(since date: Date, keepIfPermanent: Bool = true) {
        if keepIfPermanent {
            // don't remove permanent speakers
            for (speakerId, speaker) in speakerDatabase where speaker.updatedAt < date && !speaker.isPermanent {
                speakerDatabase.removeValue(forKey: speakerId)
                logger.info("Removing speaker \(speakerId) due to inactivity")
            }
        } else {
            // remove all inactive speakers
            for (speakerId, speaker) in speakerDatabase where speaker.updatedAt < date {
                speakerDatabase.removeValue(forKey: speakerId)
                logger.info("Removing speaker \(speakerId) due to inactivity")
            }
        }
    }

    /// Remove speakers that have been inactive for a given duration
    /// - Parameters:
    ///   - durationInactive: Minimum duration for which a speaker needs to be inactive to be removed
    ///   - keepIfPermanent: Whether to stop the removal if the speaker is marked as permanent
    public func removeSpeakersInactive(for durationInactive: TimeInterval, keepIfPermanent: Bool = true) {
        let date = Date().addingTimeInterval(-durationInactive)
        self.removeSpeakersInactive(since: date, keepIfPermanent: keepIfPermanent)
    }

    /// Remove speakers that meet a certain predicate
    /// - Parameters:
    ///   - predicate: The predicate to determine whether the speaker should be removed
    ///   - keepIfPermanent: Whether to stop the removal if the speaker is marked as permanent
    public func removeSpeakers(where predicate: @Sendable (Speaker) -> Bool, keepIfPermanent: Bool = true) {
        if keepIfPermanent {
            // don't remove permanent speakers
            for (speakerId, speaker) in speakerDatabase where predicate(speaker) && !speaker.isPermanent {
                speakerDatabase.removeValue(forKey: speakerId)
                logger.info("Removing speaker \(speakerId) based on predicate")
            }
        } else {
            for (speakerId, speaker) in speakerDatabase where predicate(speaker) {
                speakerDatabase.removeValue(forKey: speakerId)
                logger.info("Removing speaker \(speakerId) based on predicate")
            }
        }
    }

    /// Remove non-permanent speakers that meet a certain predicate
    /// - Parameters:
    ///   - predicate: Predicate to determine whether the speaker should be removed
    public func removeSpeakers(where predicate: @Sendable (Speaker) -> Bool) {
        removeSpeakers(where: predicate, keepIfPermanent: true)
    }

    /// Check if the speaker database has a speaker with a given ID.
    /// - Parameter speakerId: ID to check
    /// - Returns: `true` if a speaker is found, `false` if not
    public func hasSpeaker(_ speakerId: String) -> Bool {
        return speakerDatabase.keys.contains(speakerId)
    }

    private func findDistanceToClosestSpeaker(to embedding: [Float]) -> Float {
        return speakerDatabase.values.reduce(Float.infinity) {
            min($0, cosineDistance(embedding, $1.currentEmbedding))
        }
    }

    private func findClosestSpeaker(to embedding: [Float]) -> (speakerId: String?, distance: Float) {
        var minDistance: Float = Float.infinity
        var closestSpeakerId: String?

        for (speakerId, speaker) in speakerDatabase {
            let distance = cosineDistance(embedding, speaker.currentEmbedding)
            if distance < minDistance {
                minDistance = distance
                closestSpeakerId = speakerId
            }
        }

        return (closestSpeakerId, minDistance)
    }

    private func updateExistingSpeaker(
        speakerId: String,
        embedding: [Float],
        duration: Float,
        distance: Float
    ) {
        guard speakerDatabase[speakerId] != nil else {
            logger.error("Speaker \(speakerId) not found in database")
            return
        }

        // Update embedding if quality is good
        if distance < embeddingThreshold {
            var sumSquares: Float = 0
            vDSP_svesq(embedding, 1, &sumSquares, vDSP_Length(embedding.count))
            if sumSquares > 0.01 {
                speakerDatabase[speakerId]?.updateMainEmbedding(
                    duration: duration,
                    embedding: embedding,
                    segmentId: UUID(),
                    alpha: 0.9
                )
            }
        } else {
            // Just update duration if not updating embedding
            speakerDatabase[speakerId]?.duration += duration
            speakerDatabase[speakerId]?.updatedAt = Date()
        }
    }

    private func createNewSpeaker(
        embedding: [Float],
        duration: Float,
        distanceToClosest: Float,
        name: String? = nil,
        isPermanent: Bool = false
    ) -> String {
        let normalizedEmbedding = VDSPOperations.l2Normalize(embedding)
        let newSpeakerId = String(nextSpeakerId)
        let newSpeakerName = name ?? "Speaker \(newSpeakerId)"  // Default name with number if not provided
        nextSpeakerId += 1
        highestSpeakerId = max(highestSpeakerId, nextSpeakerId - 1)

        // Create new Speaker object
        var newSpeaker = Speaker(
            id: newSpeakerId,
            name: newSpeakerName,
            currentEmbedding: normalizedEmbedding,
            duration: duration,
            isPermanent: isPermanent
        )

        // Add initial raw embedding
        let initialRaw = RawEmbedding(segmentId: UUID(), embedding: normalizedEmbedding, timestamp: Date())
        newSpeaker.addRawEmbedding(initialRaw)

        speakerDatabase[newSpeakerId] = newSpeaker

        logger.info("Created new speaker \(newSpeakerId) (distance to closest: \(distanceToClosest))")
        return newSpeakerId
    }

    /// Internal cosine distance calculation that delegates to SpeakerUtilities
    /// Kept for backward compatibility with tests
    internal nonisolated func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        return SpeakerUtilities.cosineDistance(a, b)
    }

    public var speakerCount: Int {
        speakerDatabase.count
    }

    public var speakerIds: [String] {
        Array(speakerDatabase.keys).sorted()
    }

    public var permanentSpeakerIds: [String] {
        Array(speakerDatabase.filter(\.value.isPermanent).keys).sorted()
    }

    /// Get all speakers (for testing/debugging).
    public func getAllSpeakers() -> [String: Speaker] {
        return speakerDatabase
    }

    /// Get list of all speakers.
    public func getSpeakerList() -> [Speaker] {
        return [Speaker](speakerDatabase.values)
    }

    public func getSpeaker(for speakerId: String) -> Speaker? {
        speakerDatabase[speakerId]
    }

    /// - Parameter speaker: The Speaker object to upsert
    public func upsertSpeaker(_ speaker: Speaker) {
        upsertSpeaker(
            id: speaker.id,
            name: speaker.name,
            currentEmbedding: speaker.currentEmbedding,
            duration: speaker.duration,
            rawEmbeddings: speaker.rawEmbeddings,
            updateCount: speaker.updateCount,
            createdAt: speaker.createdAt,
            updatedAt: speaker.updatedAt,
            isPermanent: speaker.isPermanent
        )
    }

    /// Upsert a speaker - update if ID exists, insert if new
    ///
    /// - Parameters:
    ///   - id: The speaker ID
    ///   - currentEmbedding: The current embedding for the speaker
    ///   - duration: The total duration of speech
    ///   - rawEmbeddings: Raw embeddings for the speaker
    ///   - updateCount: Number of updates to this speaker
    ///   - createdAt: Creation timestamp
    ///   - updatedAt: Last update timestamp
    ///   - isPermanent: Whether the speaker should be protected from merges and removals by default
    public func upsertSpeaker(
        id: String,
        name: String? = nil,
        currentEmbedding: [Float],
        duration: Float,
        rawEmbeddings: [RawEmbedding] = [],
        updateCount: Int = 1,
        createdAt: Date? = nil,
        updatedAt: Date? = nil,
        isPermanent: Bool = false
    ) {
        let now = Date()

        if speakerDatabase[id] != nil {
            // Update existing speaker
            speakerDatabase[id]?.currentEmbedding = currentEmbedding
            speakerDatabase[id]?.duration = duration
            speakerDatabase[id]?.rawEmbeddings = rawEmbeddings
            speakerDatabase[id]?.updateCount = updateCount
            speakerDatabase[id]?.updatedAt = updatedAt ?? now
            if let name = name {
                speakerDatabase[id]?.name = name
            }
            if isPermanent {
                speakerDatabase[id]?.isPermanent = true
            }
            // Keep original createdAt

            logger.info("Updated existing speaker: \(id)")
        } else {
            // Insert new speaker
            var newSpeaker = Speaker(
                id: id,
                name: name ?? id,  // Default name is the ID
                currentEmbedding: currentEmbedding,
                duration: duration,
                createdAt: createdAt ?? now,
                updatedAt: updatedAt ?? now,
                isPermanent: isPermanent
            )

            newSpeaker.rawEmbeddings = rawEmbeddings
            newSpeaker.updateCount = updateCount

            speakerDatabase[id] = newSpeaker

            // Update tracking for numeric IDs
            if let numericId = Int(id) {
                highestSpeakerId = max(highestSpeakerId, numericId)
                nextSpeakerId = max(nextSpeakerId, numericId + 1)
            }

            logger.info("Inserted new speaker: \(id)")
        }
    }

    /// Reset the speaker database
    /// - Parameter keepIfPermanent: Whether to keep permanent speakers
    public func reset(keepIfPermanent: Bool = false) {
        if !keepIfPermanent {
            speakerDatabase.removeAll()
            nextSpeakerId = 1
            highestSpeakerId = 0
        } else {
            speakerDatabase = speakerDatabase.filter(\.value.isPermanent)
            // Recalculate nextSpeakerId and highestSpeakerId based on remaining permanent speakers
            var maxNumericId = 0
            for id in speakerDatabase.keys {
                if let numericId = Int(id) {
                    maxNumericId = max(maxNumericId, numericId)
                }
            }
            highestSpeakerId = maxNumericId
            nextSpeakerId = maxNumericId + 1
        }
        logger.info("Speaker database reset")
    }

    /// Mark all speakers as not permanent
    public func resetPermanentFlags() {
        for id in Array(speakerDatabase.keys) {
            speakerDatabase[id]?.isPermanent = false
        }
    }
}
