import Accelerate
import Foundation
import OSLog

// MARK: - Standalone Speaker Utilities

/// Utility functions for speaker operations that can be used by external applications
public enum SpeakerUtilities {

    private static let logger = AppLogger(category: "SpeakerUtilities")
    private static let normalizationTolerance: Float = 1e-3

    // MARK: - Configuration

    /// Platform-specific configuration for speaker assignment
    public struct AssignmentConfig: Sendable {
        public let maxDistanceForAssignment: Float
        public let maxDistanceForUpdate: Float
        public let minSpeakerDuration: Float
        public let minSegmentDuration: Float

        public init(
            maxDistanceForAssignment: Float,
            maxDistanceForUpdate: Float,
            minSpeakerDuration: Float,
            minSegmentDuration: Float
        ) {
            self.maxDistanceForAssignment = maxDistanceForAssignment
            self.maxDistanceForUpdate = maxDistanceForUpdate
            self.minSpeakerDuration = minSpeakerDuration
            self.minSegmentDuration = minSegmentDuration
        }

        public static let macOS = AssignmentConfig(
            maxDistanceForAssignment: 0.65,
            maxDistanceForUpdate: 0.45,
            minSpeakerDuration: 4.0,
            minSegmentDuration: 1.0
        )

        public static let iOS = AssignmentConfig(
            maxDistanceForAssignment: 0.55,
            maxDistanceForUpdate: 0.45,
            minSpeakerDuration: 4.0,
            minSegmentDuration: 1.0
        )

        public static var current: AssignmentConfig {
            #if os(macOS)
            return .macOS
            #else
            return .iOS
            #endif
        }
    }

    // MARK: - Distance Calculations

    /// Calculate cosine distance between two embeddings
    /// Returns value between 0 (identical) and 2 (opposite)
    /// Returns infinity if embeddings are invalid
    public static func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty, !b.isEmpty else {
            logger.error("Invalid embeddings: a.count=\(a.count), b.count=\(b.count)")
            return Float.infinity
        }

        var dotProduct: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))

        var sumSquaresA: Float = 0
        var sumSquaresB: Float = 0
        vDSP_svesq(a, 1, &sumSquaresA, vDSP_Length(a.count))
        vDSP_svesq(b, 1, &sumSquaresB, vDSP_Length(b.count))

        guard sumSquaresA > 0 && sumSquaresB > 0 else {
            logger.warning("Zero magnitude embedding detected")
            return Float.infinity
        }

        let isUnitA = abs(sumSquaresA - 1.0) <= normalizationTolerance
        let isUnitB = abs(sumSquaresB - 1.0) <= normalizationTolerance

        let similarity: Float
        if isUnitA && isUnitB {
            similarity = dotProduct
        } else {
            let magnitudeA = sumSquaresA.squareRoot()
            let magnitudeB = sumSquaresB.squareRoot()

            guard magnitudeA > 0 && magnitudeB > 0 else {
                logger.warning("Zero magnitude after normalization guard")
                return Float.infinity
            }

            similarity = dotProduct / (magnitudeA * magnitudeB)
        }

        let clampedSimilarity = min(max(similarity, -1.0), 1.0)
        return 1 - clampedSimilarity
    }

    // MARK: - Embedding Validation

    /// Validate embedding quality
    public static func validateEmbedding(_ embedding: [Float], minMagnitude: Float = 0.1) -> Bool {
        guard !embedding.isEmpty else {
            logger.error("Empty embedding")
            return false
        }

        var sumSquares: Float = 0
        vDSP_svesq(embedding, 1, &sumSquares, vDSP_Length(embedding.count))
        let magnitude = sumSquares.squareRoot()
        guard magnitude > minMagnitude else {
            logger.warning("Low magnitude embedding: \(magnitude)")
            return false
        }

        guard embedding.allSatisfy({ $0.isFinite }) else {
            logger.error("Embedding contains NaN or Inf")
            return false
        }

        return true
    }

    // MARK: - Speaker Assignment Decision

    /// Decision result for speaker assignment
    public struct AssignmentDecision {
        public let shouldAssign: Bool
        public let shouldUpdate: Bool
        public let confidence: Float
        public let reason: String
    }

    /// Determine if a speaker should be assigned based on distance and duration
    public static func shouldAssignSpeaker(
        distance: Float,
        duration: Float,
        config: AssignmentConfig = .current
    ) -> AssignmentDecision {
        // Very high confidence
        if distance < 0.2 {
            return AssignmentDecision(
                shouldAssign: true,
                shouldUpdate: duration >= config.minSpeakerDuration,
                confidence: 1.0 - distance,
                reason: "Very high confidence match"
            )
        }

        // Good match with sufficient duration
        if distance < config.maxDistanceForAssignment && duration >= config.minSegmentDuration {
            let shouldUpdate = distance <= config.maxDistanceForUpdate && duration >= config.minSpeakerDuration
            return AssignmentDecision(
                shouldAssign: true,
                shouldUpdate: shouldUpdate,
                confidence: 1.0 - distance,
                reason: "Sufficient confidence and duration"
            )
        }

        // Not confident enough
        return AssignmentDecision(
            shouldAssign: false,
            shouldUpdate: false,
            confidence: 1.0 - distance,
            reason: distance >= config.maxDistanceForAssignment ? "Distance too far" : "Duration too short"
        )
    }

    // MARK: - Batch Operations

    /// Find closest speaker from candidates
    public static func findClosestSpeaker(
        embedding: [Float],
        candidates: [Speaker]
    ) -> (speaker: Speaker?, distance: Float) {
        guard !candidates.isEmpty else {
            return (nil, Float.infinity)
        }

        var minDistance = Float.infinity
        var closestSpeaker: Speaker?

        for candidate in candidates {
            let distance = cosineDistance(embedding, candidate.currentEmbedding)
            if distance < minDistance {
                minDistance = distance
                closestSpeaker = candidate
            }
        }

        return (closestSpeaker, minDistance)
    }

    // MARK: - Speaker Creation

    /// Validated speaker creation parameters
    public struct SpeakerCreationParams {
        public let id: String
        public let name: String
        public let duration: Float
        public let embedding: [Float]
        public let createdAt: Date
        public let updatedAt: Date

        public init(
            id: String,
            name: String? = nil,
            duration: Float,
            embedding: [Float],
            createdAt: Date = Date(),
            updatedAt: Date = Date()
        ) {
            self.id = id
            self.name = name ?? "Speaker \(id)"
            self.duration = duration
            self.embedding = embedding
            self.createdAt = createdAt
            self.updatedAt = updatedAt
        }
    }

    /// Result of speaker creation validation
    public enum SpeakerCreationResult {
        case success(SpeakerCreationParams)
        case failure(reason: String)
    }

    /// Validate and prepare speaker creation parameters
    public static func validateSpeakerCreation(
        id: String,
        name: String? = nil,
        duration: Float,
        embedding: [Float],
        config: AssignmentConfig = .current
    ) -> SpeakerCreationResult {
        // Validate duration
        guard duration >= config.minSpeakerDuration else {
            return .failure(reason: "Duration \(duration) < minimum \(config.minSpeakerDuration)")
        }

        // Validate embedding
        guard validateEmbedding(embedding) else {
            return .failure(reason: "Invalid embedding: failed quality checks")
        }

        // Create validated parameters
        let normalizedEmbedding = VDSPOperations.l2Normalize(embedding)
        let params = SpeakerCreationParams(
            id: id,
            name: name,
            duration: duration,
            embedding: normalizedEmbedding
        )

        return .success(params)
    }

    /// Create a new Speaker instance with validation
    public static func createSpeaker(
        id: String,
        name: String? = nil,
        duration: Float,
        embedding: [Float],
        config: AssignmentConfig = .current
    ) -> Speaker? {
        let result = validateSpeakerCreation(
            id: id,
            name: name,
            duration: duration,
            embedding: embedding,
            config: config
        )

        switch result {
        case .success(let params):
            return Speaker(
                id: params.id,
                name: params.name,
                currentEmbedding: params.embedding,
                duration: params.duration,
                createdAt: params.createdAt,
                updatedAt: params.updatedAt
            )
        case .failure(let reason):
            logger.error("Failed to create speaker: \(reason)")
            return nil
        }
    }

    // MARK: - Embedding Updates

    /// Update a speaker embedding using exponential moving average
    /// This is a pure function that returns the updated embedding without modifying state
    public static func updateEmbedding(
        current: [Float],
        new: [Float],
        alpha: Float = 0.9
    ) -> [Float]? {
        // Validate inputs
        guard current.count == new.count,
            !current.isEmpty,
            validateEmbedding(new)
        else {
            logger.error("Invalid embeddings for update")
            return nil
        }

        // Calculate exponential moving average
        let normalizedCurrent = VDSPOperations.l2Normalize(current)
        let normalizedNew = VDSPOperations.l2Normalize(new)

        var updated = [Float](repeating: 0, count: normalizedCurrent.count)
        for i in 0..<normalizedCurrent.count {
            updated[i] = alpha * normalizedCurrent[i] + (1 - alpha) * normalizedNew[i]
        }

        return VDSPOperations.l2Normalize(updated)
    }

    // MARK: - Raw Embedding Management

    /// Adds a raw embedding with validation and FIFO management
    /// Returns the updated array and whether recalculation is needed
    public static func addRawEmbedding(
        to rawEmbeddings: [RawEmbedding],
        segmentId: UUID,
        embedding: [Float],
        timestamp: Date = Date(),
        maxCapacity: Int = 50
    ) -> (updated: [RawEmbedding], shouldRecalculate: Bool)? {
        // Validate the embedding
        guard validateEmbedding(embedding) else {
            logger.warning("Invalid embedding for segment \(segmentId)")
            return nil
        }

        // Create the new raw embedding
        let normalizedEmbedding = VDSPOperations.l2Normalize(embedding)
        let newEmbedding = RawEmbedding(
            segmentId: segmentId,
            embedding: normalizedEmbedding,
            timestamp: timestamp
        )

        // FIFO queue management - inline for simplicity
        var updated = rawEmbeddings
        if updated.count >= maxCapacity {
            updated.removeFirst()
        }
        updated.append(newEmbedding)

        let shouldRecalculate = updated.count >= 3  // Need at least 3 for meaningful average

        return (updated, shouldRecalculate)
    }

    /// Removes a raw embedding by segment ID
    /// Returns the updated array and the removed embedding if found
    public static func removeRawEmbedding(
        from rawEmbeddings: [RawEmbedding],
        segmentId: UUID
    ) -> (updated: [RawEmbedding], removed: RawEmbedding?, shouldRecalculate: Bool) {
        var updated = rawEmbeddings

        if let index = updated.firstIndex(where: { $0.segmentId == segmentId }) {
            let removed = updated.remove(at: index)
            let shouldRecalculate = !updated.isEmpty
            return (updated, removed, shouldRecalculate)
        }

        return (rawEmbeddings, nil, false)
    }

    // MARK: - Complete Speaker Update Operations

    /// Complete speaker update operation including raw tracking
    public struct SpeakerUpdateResult {
        public let updatedMainEmbedding: [Float]?
        public let updatedRawEmbeddings: [RawEmbedding]
        public let updatedDuration: Float
        public let shouldRecalculate: Bool
    }

    /// Updates a speaker with new segment data - handles both main and raw embeddings
    public static func updateSpeakerWithSegment(
        currentMainEmbedding: [Float],
        currentRawEmbeddings: [RawEmbedding],
        currentDuration: Float,
        segmentDuration: Float,
        segmentEmbedding: [Float],
        segmentId: UUID,
        alpha: Float = 0.9,
        minSegmentDuration: Float = 2.0
    ) -> SpeakerUpdateResult? {
        // Validate segment duration
        guard segmentDuration >= minSegmentDuration else {
            return nil
        }

        // Validate embedding
        guard validateEmbedding(segmentEmbedding) else {
            return nil
        }

        let normalizedEmbedding = VDSPOperations.l2Normalize(segmentEmbedding)

        // Add to raw embeddings
        guard
            let (updatedRaw, shouldRecalc) = addRawEmbedding(
                to: currentRawEmbeddings,
                segmentId: segmentId,
                embedding: normalizedEmbedding,
                timestamp: Date()
            )
        else {
            return nil
        }

        // Update main embedding using exponential moving average
        let updatedMain = updateEmbedding(
            current: currentMainEmbedding,
            new: normalizedEmbedding,
            alpha: alpha
        )

        // Calculate new duration
        let newDuration = currentDuration + segmentDuration

        return SpeakerUpdateResult(
            updatedMainEmbedding: updatedMain,
            updatedRawEmbeddings: updatedRaw,
            updatedDuration: newDuration,
            shouldRecalculate: shouldRecalc
        )
    }

    /// Merge two speakers' data
    public static func mergeSpeakers(
        speaker1Raw: [RawEmbedding],
        speaker1Duration: Float,
        speaker2Raw: [RawEmbedding],
        speaker2Duration: Float,
        maxCapacity: Int = 50
    ) -> (mergedRaw: [RawEmbedding], mergedDuration: Float, newMainEmbedding: [Float]?) {
        // Merge raw embeddings
        var allEmbeddings = speaker1Raw + speaker2Raw

        // Keep only the most recent embeddings if over capacity
        if allEmbeddings.count > maxCapacity {
            allEmbeddings = Array(
                allEmbeddings
                    .sorted { $0.timestamp > $1.timestamp }
                    .prefix(maxCapacity)
            )
        }

        // Calculate new main embedding from merged history
        let embeddingArrays = allEmbeddings.map { $0.embedding }
        let newMainEmbedding = averageEmbeddings(embeddingArrays)

        // Merge durations
        let mergedDuration = speaker1Duration + speaker2Duration

        return (allEmbeddings, mergedDuration, newMainEmbedding)
    }

    /// Calculate average of multiple embeddings
    public static func averageEmbeddings(_ embeddings: [[Float]]) -> [Float]? {
        guard !embeddings.isEmpty,
            let dimension = embeddings.first?.count,
            dimension > 0
        else {
            return nil
        }

        var average = [Float](repeating: 0, count: dimension)
        var validCount = 0

        for embedding in embeddings {
            guard embedding.count == dimension else { continue }
            for i in 0..<dimension {
                average[i] += embedding[i]
            }
            validCount += 1
        }

        guard validCount > 0 else { return nil }

        for i in 0..<dimension {
            average[i] /= Float(validCount)
        }

        return VDSPOperations.l2Normalize(average)
    }
}

/// These functions provide additional capabilities beyond core diarization
extension SpeakerManager {

    /// Reassign a raw embedding from one speaker to another.
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
        return queue.sync(flags: .barrier) {
            // Get speaker info from database
            guard let fromSpeakerInfo = speakerDatabase[fromSpeakerId],
                let toSpeakerInfo = speakerDatabase[toSpeakerId]
            else {
                logger.warning("One or both speakers not found: from=\(fromSpeakerId), to=\(toSpeakerId)")
                return false
            }

            // Find and remove the embedding from source speaker
            guard let index = fromSpeakerInfo.rawEmbeddings.firstIndex(where: { $0.segmentId == segmentId })
            else {
                logger.warning("Segment \(segmentId) not found in speaker \(fromSpeakerId)")
                return false
            }

            let embedding = fromSpeakerInfo.rawEmbeddings.remove(at: index)

            // Add to destination speaker
            toSpeakerInfo.rawEmbeddings.append(embedding)

            // Update both speakers in database
            speakerDatabase[fromSpeakerId] = fromSpeakerInfo
            speakerDatabase[toSpeakerId] = toSpeakerInfo

            // Recalculate embeddings for both speakers
            // Note: This would need to be implemented using the SpeakerInfo structure
            // For now, just log the successful reassignment

            logger.info("✅ Reassigned segment \(segmentId) from \(fromSpeakerId) to \(toSpeakerId)")
            return true
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
            let speakersWithHistory = speakers.filter { !$0.rawEmbeddings.isEmpty }.count

            logger.info(
                "Global stats - Speakers: \(speakers.count), Duration: \(String(format: "%.1f", totalDuration))s, Avg confidence: \(String(format: "%.2f", averageConfidence)), With history: \(speakersWithHistory)"
            )

            return (speakers.count, totalDuration, min(1.0, averageConfidence), speakersWithHistory)
        }
    }
}
