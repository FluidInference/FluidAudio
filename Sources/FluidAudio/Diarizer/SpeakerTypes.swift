import Foundation

/// Speaker profile representation for tracking speakers across audio
/// This represents a speaker's identity, not a specific segment
/// Aligned with Slipbox Speaker schema for compatibility
@available(macOS 13.0, iOS 16.0, *)
public final class Speaker: Identifiable, Codable, Sendable {
    public let id: String  // Note: Slipbox uses Int, conversion needed at integration
    public var name: String
    public var currentEmbedding: [Float]
    public var duration: Float = 0  // Renamed from totalDuration to match Slipbox
    public var createdAt: Date  // Added to match Slipbox
    public var updatedAt: Date  // Added to match Slipbox
    public var updateCount: Int = 1
    public var rawEmbeddings: [RawEmbedding] = []

    public init(
        id: String? = nil,
        name: String? = nil,
        currentEmbedding: [Float],
        duration: Float = 0,
        createdAt: Date? = nil,
        updatedAt: Date? = nil
    ) {
        let now = Date()
        self.id = id ?? "Speaker_\(UUID().uuidString.prefix(8))"
        self.name = name ?? self.id
        self.currentEmbedding = currentEmbedding
        self.duration = duration
        self.createdAt = createdAt ?? now
        self.updatedAt = updatedAt ?? now
        self.updateCount = 1
        self.rawEmbeddings = []
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
        let historicalEmbedding = RawEmbedding(
            segmentId: segmentId,
            embedding: embedding,
            timestamp: Date()
        )
        addHistoricalEmbedding(historicalEmbedding)

        // Update main embedding using exponential moving average
        if currentEmbedding.count == embedding.count {
            for i in 0..<currentEmbedding.count {
                currentEmbedding[i] = alpha * currentEmbedding[i] + (1 - alpha) * embedding[i]
            }
        }

        // Update metadata
        self.duration += duration
        self.updatedAt = Date()
        self.updateCount += 1
    }

    /// Add a historical embedding with FIFO queue management
    public func addHistoricalEmbedding(_ embedding: RawEmbedding) {
        // Validate embedding quality
        let embeddingMagnitude = sqrt(embedding.embedding.map { $0 * $0 }.reduce(0, +))
        guard embeddingMagnitude > 0.1 else { return }

        // Maintain max of 50 historical embeddings (FIFO)
        if rawEmbeddings.count >= 50 {
            rawEmbeddings.removeFirst()
        }

        rawEmbeddings.append(embedding)
        recalculateMainEmbedding()
    }

    /// Remove a historical embedding by segment ID
    @discardableResult
    public func removeHistoricalEmbedding(segmentId: UUID) -> RawEmbedding? {
        guard let index = rawEmbeddings.firstIndex(where: { $0.segmentId == segmentId }) else {
            return nil
        }

        let removed = rawEmbeddings.remove(at: index)
        recalculateMainEmbedding()
        return removed
    }

    /// Recalculate main embedding as average of all historical embeddings
    public func recalculateMainEmbedding() {
        guard !rawEmbeddings.isEmpty,
            let firstEmbedding = rawEmbeddings.first,
            !firstEmbedding.embedding.isEmpty
        else { return }

        let embeddingSize = firstEmbedding.embedding.count
        var averageEmbedding = [Float](repeating: 0.0, count: embeddingSize)

        // Calculate average of all historical embeddings
        var validCount = 0
        for historical in rawEmbeddings {
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

            self.currentEmbedding = averageEmbedding
            self.updatedAt = Date()
        }
    }

    /// Merge another speaker into this one
    public func mergeWith(_ other: Speaker, keepName: String? = nil) {
        // Merge historical embeddings
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

}

/// Raw embedding tracking for speaker evolution over time
@available(macOS 13.0, iOS 16.0, *)
public struct RawEmbedding: Codable, Sendable {
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

    // Primary init for Slipbox compatibility
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
        self.id = Int(speaker.id.split(separator: "_").last.flatMap { Int($0) } ?? 0)
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
