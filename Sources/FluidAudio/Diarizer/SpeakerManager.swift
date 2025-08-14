import Foundation
import OSLog

/// In-memory speaker database for streaming diarization
/// Tracks speakers across chunks and maintains consistent IDs
@available(macOS 13.0, iOS 16.0, *)
public class SpeakerManager {
    internal let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "SpeakerManager")

    // Constants
    public static let EMBEDDING_SIZE = 256  // Standard embedding dimension for speaker models

    // Speaker database: ID -> representative embedding
    internal var speakerDatabase: [String: SpeakerInfo] = [:]
    private var nextSpeakerId = 1
    internal let queue = DispatchQueue(label: "speaker.manager.queue", attributes: .concurrent)

    // Track the highest speaker ID to ensure uniqueness
    private var highestSpeakerId = 0

    public var speakerThreshold: Float  // Max distance for speaker assignment (default: 0.65)
    public var embeddingThreshold: Float  // Max distance for updating embeddings (default: 0.45)
    public var minSpeechDuration: Float  // Min duration to create speaker (default: 1.0)
    public var minEmbeddingUpdateDuration: Float  // Min duration to update embeddings (default: 2.0)

    public struct SpeakerInfo {
        public let id: String
        public var currentEmbedding: [Float]
        public var duration: Float
        public var createdAt: Date
        public var updatedAt: Date
        public var updateCount: Int
        public var rawEmbeddings: [RawEmbedding]

        public init(
            id: String, currentEmbedding: [Float], duration: Float, createdAt: Date? = nil, updatedAt: Date? = nil
        ) {
            let now = Date()
            self.id = id
            self.currentEmbedding = currentEmbedding
            self.duration = duration
            self.createdAt = createdAt ?? now
            self.updatedAt = updatedAt ?? now
            self.updateCount = 1
            self.rawEmbeddings = []
        }
    }

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

    public func initializeKnownSpeakers(_ speakers: [Speaker]) {
        queue.sync(flags: .barrier) {
            var maxNumericId = 0

            for speaker in speakers {
                guard speaker.currentEmbedding.count == Self.EMBEDDING_SIZE else {
                    logger.warning(
                        "Skipping speaker \(speaker.id) - invalid embedding size: \(speaker.currentEmbedding.count)")
                    continue
                }

                var speakerInfo = SpeakerInfo(
                    id: speaker.id,
                    currentEmbedding: speaker.currentEmbedding,
                    duration: speaker.duration,
                    createdAt: speaker.createdAt,
                    updatedAt: speaker.updatedAt
                )

                speakerInfo.rawEmbeddings = speaker.rawEmbeddings
                speakerInfo.updateCount = speaker.updateCount

                speakerDatabase[speaker.id] = speakerInfo

                if speaker.id.hasPrefix("Speaker_"),
                    let numericPart = speaker.id.split(separator: "_").last,
                    let numericId = Int(numericPart)
                {
                    maxNumericId = max(maxNumericId, numericId)
                }

                logger.info(
                    "Initialized known speaker: \(speaker.id) with \(speaker.rawEmbeddings.count) historical embeddings"
                )
            }

            self.highestSpeakerId = maxNumericId
            self.nextSpeakerId = maxNumericId + 1

            logger.info(
                "Initialized with \(self.speakerDatabase.count) known speakers, next ID will be: Speaker_\(self.nextSpeakerId)"
            )
        }
    }

    public func assignSpeaker(
        _ embedding: [Float],
        speechDuration: Float,
        confidence: Float = 1.0
    ) -> Speaker? {
        guard !embedding.isEmpty && embedding.count == Self.EMBEDDING_SIZE else {
            logger.error("Invalid embedding size: \(embedding.count)")
            return nil
        }

        return queue.sync(flags: .barrier) {
            let (closestSpeaker, distance) = findClosestSpeaker(to: embedding)

            if let speakerId = closestSpeaker, distance < speakerThreshold {
                updateExistingSpeaker(
                    speakerId: speakerId,
                    embedding: embedding,
                    duration: speechDuration,
                    distance: distance
                )

                if let speakerInfo = speakerDatabase[speakerId] {
                    return Speaker(
                        id: speakerInfo.id,
                        name: "Speaker \(speakerInfo.id)",
                        currentEmbedding: speakerInfo.currentEmbedding,
                        duration: speakerInfo.duration,
                        createdAt: speakerInfo.createdAt,
                        updatedAt: speakerInfo.updatedAt
                    )
                }
                return nil
            }

            // Step 3: Create new speaker if duration is sufficient
            if speechDuration >= minSpeechDuration {
                let newSpeakerId = createNewSpeaker(
                    embedding: embedding,
                    duration: speechDuration,
                    distanceToClosest: distance
                )

                // Return the full Speaker object
                if let speakerInfo = speakerDatabase[newSpeakerId] {
                    return Speaker(
                        id: speakerInfo.id,
                        name: "Speaker \(speakerInfo.id)",
                        currentEmbedding: speakerInfo.currentEmbedding,
                        duration: speakerInfo.duration,
                        createdAt: speakerInfo.createdAt,
                        updatedAt: speakerInfo.updatedAt
                    )
                }
                return nil
            }

            // Step 4: Audio segment too short
            logger.debug("Audio segment too short (\(speechDuration)s) to create new speaker")
            return nil
        }
    }

    private func findClosestSpeaker(to embedding: [Float]) -> (speakerId: String?, distance: Float) {
        var minDistance: Float = Float.infinity
        var closestSpeakerId: String?

        for (speakerId, speakerInfo) in speakerDatabase {
            let distance = cosineDistance(embedding, speakerInfo.currentEmbedding)
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
        guard var speakerInfo = speakerDatabase[speakerId] else {
            logger.error("Speaker \(speakerId) not found in database")
            return
        }

        // Update embedding if quality is good and duration meets threshold
        if distance < embeddingThreshold && duration >= minEmbeddingUpdateDuration {
            let embeddingMagnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
            if embeddingMagnitude > 0.1 {
                // Add to historical embeddings (with FIFO management - max 50)
                let historical = RawEmbedding(segmentId: UUID(), embedding: embedding)
                if speakerInfo.rawEmbeddings.count >= 50 {
                    speakerInfo.rawEmbeddings.removeFirst()
                }
                speakerInfo.rawEmbeddings.append(historical)

                // Update main embedding using exponential moving average
                let alpha: Float = 0.9
                for i in 0..<speakerInfo.currentEmbedding.count {
                    speakerInfo.currentEmbedding[i] =
                        alpha * speakerInfo.currentEmbedding[i] + (1 - alpha) * embedding[i]
                }

                speakerInfo.updateCount += 1
                logger.debug(
                    "Updated embedding for \(speakerId), update count: \(speakerInfo.updateCount), historical count: \(speakerInfo.rawEmbeddings.count)"
                )
            }
        }

        // Always update speaker stats regardless of embedding update
        speakerInfo.updatedAt = Date()
        speakerInfo.duration += duration
        speakerDatabase[speakerId] = speakerInfo
    }

    private func createNewSpeaker(
        embedding: [Float],
        duration: Float,
        distanceToClosest: Float
    ) -> String {
        let newSpeakerId = "Speaker_\(nextSpeakerId)"
        nextSpeakerId += 1
        highestSpeakerId = max(highestSpeakerId, nextSpeakerId - 1)

        // Create speaker with initial historical embedding
        let initialHistorical = RawEmbedding(segmentId: UUID(), embedding: embedding)
        var newSpeakerInfo = SpeakerInfo(
            id: newSpeakerId,
            currentEmbedding: embedding,
            duration: duration
        )
        newSpeakerInfo.rawEmbeddings = [initialHistorical]
        speakerDatabase[newSpeakerId] = newSpeakerInfo

        logger.info("Created new speaker \(newSpeakerId) (distance to closest: \(distanceToClosest))")
        return newSpeakerId
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

    /// Get all speaker info (for testing/debugging).
    public func getAllSpeakerInfo() -> [String: SpeakerInfo] {
        queue.sync {
            return speakerDatabase
        }
    }

    public func getSpeakerInfo(for speakerId: String) -> SpeakerInfo? {
        queue.sync { speakerDatabase[speakerId] }
    }

    /// Upsert a speaker from a Speaker object
    ///
    /// - Parameter speaker: The Speaker object to upsert
    public func upsertSpeaker(_ speaker: Speaker) {
        upsertSpeaker(
            id: speaker.id,
            currentEmbedding: speaker.currentEmbedding,
            duration: speaker.duration,
            rawEmbeddings: speaker.rawEmbeddings,
            updateCount: speaker.updateCount,
            createdAt: speaker.createdAt,
            updatedAt: speaker.updatedAt
        )
    }

    /// Upsert a speaker - update if exists, insert if new
    ///
    /// - Parameters:
    ///   - id: The speaker ID
    ///   - currentEmbedding: The current embedding for the speaker
    ///   - duration: The total duration of speech
    ///   - rawEmbeddings: Historical embeddings for the speaker
    ///   - updateCount: Number of updates to this speaker
    ///   - createdAt: Creation timestamp
    ///   - updatedAt: Last update timestamp
    public func upsertSpeaker(
        id: String,
        currentEmbedding: [Float],
        duration: Float,
        rawEmbeddings: [RawEmbedding] = [],
        updateCount: Int = 1,
        createdAt: Date? = nil,
        updatedAt: Date? = nil
    ) {
        queue.sync(flags: .barrier) {
            let now = Date()

            if var existingSpeaker = speakerDatabase[id] {
                // Update existing speaker
                existingSpeaker.currentEmbedding = currentEmbedding
                existingSpeaker.duration = duration
                existingSpeaker.rawEmbeddings = rawEmbeddings
                existingSpeaker.updateCount = updateCount
                existingSpeaker.updatedAt = updatedAt ?? now
                // Keep original createdAt

                speakerDatabase[id] = existingSpeaker
                logger.info("Updated existing speaker: \(id)")
            } else {
                // Insert new speaker
                let newSpeaker = SpeakerInfo(
                    id: id,
                    currentEmbedding: currentEmbedding,
                    duration: duration,
                    createdAt: createdAt ?? now,
                    updatedAt: updatedAt ?? now
                )

                var mutableSpeaker = newSpeaker
                mutableSpeaker.rawEmbeddings = rawEmbeddings
                mutableSpeaker.updateCount = updateCount

                speakerDatabase[id] = mutableSpeaker

                // Update tracking for numeric IDs
                if id.hasPrefix("Speaker_"),
                    let numericPart = id.split(separator: "_").last,
                    let numericId = Int(numericPart)
                {
                    highestSpeakerId = max(highestSpeakerId, numericId)
                    nextSpeakerId = max(nextSpeakerId, numericId + 1)
                }

                logger.info("Inserted new speaker: \(id)")
            }
        }
    }

    public func reset() {
        queue.sync(flags: .barrier) {
            speakerDatabase.removeAll()
            nextSpeakerId = 1
            highestSpeakerId = 0
            logger.info("Speaker database reset")
        }
    }
}
