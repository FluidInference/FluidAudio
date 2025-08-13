import Foundation

/// Speaker profile representation for tracking speakers across audio
/// This represents a speaker's identity, not a specific segment
/// Aligned with Slipbox Speaker schema for compatibility
@available(macOS 13.0, iOS 16.0, *)
public final class Speaker: Identifiable, Codable, Sendable {
    public let id: String  // Note: Slipbox uses Int, conversion needed at integration
    public var name: String
    public var mainEmbedding: [Float]
    public var duration: Float = 0  // Renamed from totalDuration to match Slipbox
    public var createdAt: Date  // Added to match Slipbox
    public var updatedAt: Date  // Added to match Slipbox
    public var updateCount: Int = 1
    public var historicalEmbeddings: [HistoricalEmbedding] = []

    public init(
        id: String? = nil,
        name: String? = nil,
        mainEmbedding: [Float],
        duration: Float = 0,
        createdAt: Date? = nil,
        updatedAt: Date? = nil
    ) {
        let now = Date()
        self.id = id ?? "Speaker_\(UUID().uuidString.prefix(8))"
        self.name = name ?? self.id
        self.mainEmbedding = mainEmbedding
        self.duration = duration
        self.createdAt = createdAt ?? now
        self.updatedAt = updatedAt ?? now
        self.updateCount = 1
        self.historicalEmbeddings = []
    }

    /// Convert to SendableSpeaker format for cross-boundary usage.
    public func toSendable() -> SendableSpeaker {
        return SendableSpeaker(from: self)
    }

    // MARK: - Embedding Management Methods

    /// Update main embedding with new segment data using exponential moving average
    public func updateMainEmbedding(
        duration: Float,
        embedding: [Float],
        segmentId: UUID,
        alpha: Float = 0.9
    ) {
        // Only process embeddings from segments that are at least 2 seconds long
        guard duration >= 2.0 else { return }

        // Validate embedding quality
        let embeddingMagnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
        guard embeddingMagnitude > 0.1 else { return }

        // Add to historical embeddings
        let historicalEmbedding = HistoricalEmbedding(
            segmentId: segmentId,
            embedding: embedding,
            timestamp: Date()
        )
        addHistoricalEmbedding(historicalEmbedding)

        // Update main embedding using exponential moving average
        if mainEmbedding.count == embedding.count {
            for i in 0..<mainEmbedding.count {
                mainEmbedding[i] = alpha * mainEmbedding[i] + (1 - alpha) * embedding[i]
            }
        }

        // Update metadata
        self.duration += duration
        self.updatedAt = Date()
        self.updateCount += 1
    }

    /// Add a historical embedding with FIFO queue management
    public func addHistoricalEmbedding(_ embedding: HistoricalEmbedding) {
        // Validate embedding quality
        let embeddingMagnitude = sqrt(embedding.embedding.map { $0 * $0 }.reduce(0, +))
        guard embeddingMagnitude > 0.1 else { return }

        // Maintain max of 50 historical embeddings (FIFO)
        if historicalEmbeddings.count >= 50 {
            historicalEmbeddings.removeFirst()
        }

        historicalEmbeddings.append(embedding)
        recalculateMainEmbedding()
    }

    /// Remove a historical embedding by segment ID
    @discardableResult
    public func removeHistoricalEmbedding(segmentId: UUID) -> HistoricalEmbedding? {
        guard let index = historicalEmbeddings.firstIndex(where: { $0.segmentId == segmentId }) else {
            return nil
        }

        let removed = historicalEmbeddings.remove(at: index)
        recalculateMainEmbedding()
        return removed
    }

    /// Recalculate main embedding as average of all historical embeddings
    public func recalculateMainEmbedding() {
        guard !historicalEmbeddings.isEmpty,
            let firstEmbedding = historicalEmbeddings.first,
            !firstEmbedding.embedding.isEmpty
        else { return }

        let embeddingSize = firstEmbedding.embedding.count
        var averageEmbedding = [Float](repeating: 0.0, count: embeddingSize)

        // Calculate average of all historical embeddings
        var validCount = 0
        for historical in historicalEmbeddings {
            if historical.embedding.count == embeddingSize {
                for i in 0..<embeddingSize {
                    averageEmbedding[i] += historical.embedding[i]
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

            self.mainEmbedding = averageEmbedding
            self.updatedAt = Date()
        }
    }

    /// Merge another speaker into this one
    public func mergeWith(_ other: Speaker, keepName: String? = nil) {
        // Merge historical embeddings
        var allEmbeddings = historicalEmbeddings + other.historicalEmbeddings

        // Keep only the most recent 50 embeddings
        if allEmbeddings.count > 50 {
            allEmbeddings = Array(
                allEmbeddings
                    .sorted { $0.timestamp > $1.timestamp }
                    .prefix(50)
            )
        }

        historicalEmbeddings = allEmbeddings

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

}

/// Historical embedding tracking for speaker evolution over time
@available(macOS 13.0, iOS 16.0, *)
public struct HistoricalEmbedding: Codable, Sendable {
    public let segmentId: UUID
    public let embedding: [Float]
    public let timestamp: Date

    public init(segmentId: UUID = UUID(), embedding: [Float], timestamp: Date = Date()) {
        self.segmentId = segmentId
        self.embedding = embedding
        self.timestamp = timestamp
    }
}

/// Sendable speaker data for cross-async boundary usage
@available(macOS 13.0, iOS 16.0, *)
public struct SendableSpeaker: Sendable, Identifiable, Hashable {
    public let id: String
    public let name: String
    public let duration: Float
    public let mainEmbedding: [Float]
    public let createdAt: Date
    public let updatedAt: Date

    /// Label for display - matches Slipbox format
    public var label: String {
        if name.isEmpty {
            // Extract numeric part for Slipbox-like formatting
            if let lastPart = id.split(separator: "_").last {
                return "Speaker #\(lastPart)"
            }
            return "Speaker #\(id)"
        } else {
            return name
        }
    }

    public init(from speaker: Speaker) {
        self.id = speaker.id
        self.name = speaker.name
        self.duration = speaker.duration  // Now matches field name
        self.mainEmbedding = speaker.mainEmbedding
        self.createdAt = speaker.createdAt
        self.updatedAt = speaker.updatedAt
    }

    public init(
        id: String,
        name: String,
        duration: Float,
        mainEmbedding: [Float],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.duration = duration
        self.mainEmbedding = mainEmbedding
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    public static func == (lhs: SendableSpeaker, rhs: SendableSpeaker) -> Bool {
        return lhs.id == rhs.id && lhs.name == rhs.name
    }
}
