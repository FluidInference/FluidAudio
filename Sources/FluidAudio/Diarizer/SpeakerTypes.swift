import Foundation

/// Speaker profile representation for tracking speakers across audio
/// This represents a speaker's identity, not a specific segment
/// Aligned with Slipbox Speaker schema for compatibility
@available(macOS 13.0, iOS 16.0, *)
public struct Speaker: Identifiable, Codable, Sendable {
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
