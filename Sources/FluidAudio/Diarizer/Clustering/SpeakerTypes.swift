import Accelerate
import Foundation

/// Speaker profile representation for tracking speakers across audio
/// This represents a speaker's identity, not a specific segment
public final class Speaker: Identifiable, Codable, Equatable, Hashable {
    /// Speaker ID
    public let id: String
    /// Speaker name
    public var name: String
    /// Main embedding vector for this speaker's voice
    public var currentEmbedding: [Float]
    /// Total speech duration for this speaker
    public var duration: Float = 0
    /// Date that this speaker object was created
    public var createdAt: Date
    /// Date that this speaker object was last updated
    public var updatedAt: Date
    /// Number of times the embedding vector was updated
    public var updateCount: Int = 1
    /// Array of raw embedding vectors
    public var rawEmbeddings: [RawEmbedding] = []
    /// Whether this speaker can be deleted due to inactivity or merging
    public var isPermanent: Bool = false

    /// - Parameters:
    ///   - id: Speaker ID
    ///   - name: Speaker name
    ///   - currentEmbedding: Main embedding vector for this speaker's voice
    ///   - duration: Total speech duration for this speaker
    ///   - createdAt: Date that this speaker object was last updated
    ///   - updatedAt: Number of times the embedding vector was updated
    ///   - updateCount: Array of raw embedding vectors
    ///   - rawEmbeddings: Array of raw embedding vectors
    ///   - isPermanent: Whether this speaker can be deleted due to inactivity or merging
    public init(
        id: String? = nil,
        name: String? = nil,
        currentEmbedding: [Float],
        duration: Float = 0,
        createdAt: Date? = nil,
        updatedAt: Date? = nil,
        isPermanent: Bool = false
    ) {
        let now = Date()
        self.id = id ?? UUID().uuidString
        self.name = name ?? self.id
        self.currentEmbedding = VDSPOperations.l2Normalize(currentEmbedding)
        self.duration = duration
        self.createdAt = createdAt ?? now
        self.updatedAt = updatedAt ?? now
        self.updateCount = 1
        self.rawEmbeddings = []
        self.isPermanent = isPermanent
    }

    /// Convert to SendableSpeaker format for cross-boundary usage.
    public func toSendable() -> SendableSpeaker {
        return SendableSpeaker(from: self)
    }

    /// Update main embedding with new segment data using exponential moving average (EMA)
    /// - Parameters:
    ///   - duration: Segment duration
    ///   - embedding: 256D speaker embedding vector
    ///   - segmentId: The ID of the segment
    ///   - alpha: EMA blending parameter
    public func updateMainEmbedding(
        duration: Float,
        embedding: [Float],
        segmentId: UUID,
        alpha: Float = 0.9
    ) {
        // Validate embedding quality
        var sumSquares: Float = 0
        vDSP_svesq(embedding, 1, &sumSquares, vDSP_Length(embedding.count))
        guard sumSquares > 0.01 else { return }

        let normalizedEmbedding = VDSPOperations.l2Normalize(embedding)

        // Add to raw embeddings
        let rawEmbedding = RawEmbedding(
            segmentId: segmentId,
            embedding: normalizedEmbedding,
            timestamp: Date()
        )
        addRawEmbedding(rawEmbedding)

        // Update main embedding using exponential moving average
        if currentEmbedding.count == normalizedEmbedding.count {
            for i in 0..<currentEmbedding.count {
                currentEmbedding[i] = alpha * currentEmbedding[i] + (1 - alpha) * normalizedEmbedding[i]
            }
            currentEmbedding = VDSPOperations.l2Normalize(currentEmbedding)
        }

        // Update metadata
        self.duration += duration
        self.updatedAt = Date()
        self.updateCount += 1
    }

    /// Add a raw embedding with FIFO queue management
    public func addRawEmbedding(_ embedding: RawEmbedding) {
        // Validate embedding quality
        var sumSquares: Float = 0
        vDSP_svesq(embedding.embedding, 1, &sumSquares, vDSP_Length(embedding.embedding.count))
        guard sumSquares > 0.01 else { return }

        // Maintain max of 50 raw embeddings (FIFO)
        if rawEmbeddings.count >= 50 {
            rawEmbeddings.removeFirst()
        }

        rawEmbeddings.append(embedding)
        recalculateMainEmbedding()
    }

    /// Remove a raw embedding by segment ID
    @discardableResult
    public func removeRawEmbedding(segmentId: UUID) -> RawEmbedding? {
        guard let index = rawEmbeddings.firstIndex(where: { $0.segmentId == segmentId }) else {
            return nil
        }

        let removed = rawEmbeddings.remove(at: index)
        recalculateMainEmbedding()
        return removed
    }

    /// Recalculate main embedding as average of all raw embeddings
    public func recalculateMainEmbedding() {
        guard !rawEmbeddings.isEmpty,
            let firstEmbedding = rawEmbeddings.first,
            !firstEmbedding.embedding.isEmpty
        else { return }

        let embeddingSize = firstEmbedding.embedding.count
        var averageEmbedding = [Float](repeating: 0.0, count: embeddingSize)

        // Calculate average of all raw embeddings
        var validCount = 0
        for raw in rawEmbeddings {
            if raw.embedding.count == embeddingSize {
                for i in 0..<embeddingSize {
                    averageEmbedding[i] += raw.embedding[i]
                }
                validCount += 1
            }
        }

        // Divide by count to get average
        if validCount > 0 {
            let count = Float(validCount)
            for i in 0..<embeddingSize {
                averageEmbedding[i] /= count
            }

            self.currentEmbedding = VDSPOperations.l2Normalize(averageEmbedding)
            self.updatedAt = Date()
        }
    }

    /// Merge another speaker into this one
    /// - Parameters:
    ///   - other: Other Speaker to merge
    ///   - keepName: The resulting name after the merge
    public func mergeWith(_ other: Speaker, keepName: String? = nil) {
        // Merge raw embeddings
        var allEmbeddings = rawEmbeddings + other.rawEmbeddings

        // Keep only the most recent 50 embeddings
        if allEmbeddings.count > 50 {
            allEmbeddings = Array(
                allEmbeddings
                    .sorted { $0.timestamp > $1.timestamp }
                    .prefix(50)
            )
        }

        rawEmbeddings = allEmbeddings

        // Update duration
        duration += other.duration

        // Update name if specified
        if let keepName = keepName {
            name = keepName
        }

        // Recalculate main embedding
        recalculateMainEmbedding()

        updatedAt = Date()
        updateCount += other.updateCount
    }

    public static func == (lhs: Speaker, rhs: Speaker) -> Bool {
        return lhs.id == rhs.id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

/// Raw embedding tracking for speaker evolution over time
public struct RawEmbedding: Codable, Sendable {
    public let segmentId: UUID
    public let embedding: [Float]
    public let timestamp: Date

    public init(segmentId: UUID = UUID(), embedding: [Float], timestamp: Date = Date()) {
        self.segmentId = segmentId
        self.embedding = VDSPOperations.l2Normalize(embedding)
        self.timestamp = timestamp
    }
}

/// Sendable speaker data for cross-async boundary usage
public struct SendableSpeaker: Sendable, Identifiable, Hashable {
    public let id: Int
    public let name: String
    public let duration: Float
    public let mainEmbedding: [Float]
    public let createdAt: Date
    public let updatedAt: Date

    /// Label for display
    public var label: String {
        if name.isEmpty {
            return "Speaker #\(id)"
        } else {
            return name
        }
    }

    public init(id: Int, name: String, duration: Float, mainEmbedding: [Float], createdAt: Date, updatedAt: Date) {
        self.id = id
        self.name = name
        self.duration = duration
        self.mainEmbedding = mainEmbedding
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    // Convenience init from FluidAudio's Speaker type
    public init(from speaker: Speaker) {
        // Try to parse as integer first, otherwise use hash of UUID
        if let numericId = Int(speaker.id) {
            self.id = numericId
        } else {
            // For UUID strings, use a stable hash
            self.id = abs(speaker.id.hashValue)
        }
        self.name = speaker.name
        self.duration = speaker.duration
        self.mainEmbedding = speaker.currentEmbedding
        self.createdAt = speaker.createdAt
        self.updatedAt = speaker.updatedAt
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    public static func == (lhs: SendableSpeaker, rhs: SendableSpeaker) -> Bool {
        return lhs.id == rhs.id && lhs.name == rhs.name
    }
}

/// Configuration for handling initializing known speakers
public enum SpeakerInitializationMode {
    /// Reset the speaker database and add the new speakers
    case reset
    /// Merge new speakers whose IDs match with existing ones
    case merge
    /// Overwrite existing speakers with the same IDs as the new ones
    case overwrite
    /// Skip speakers whose IDs match existing ones
    case skip
}
