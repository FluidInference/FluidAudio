import Foundation
import OSLog

/// Tracks token stability across hypothesis updates to reduce text fluctuation.
///
/// The stabilizer implements a "progressive locking" algorithm:
/// 1. Track how many consecutive updates each token has survived unchanged
/// 2. Lock tokens that exceed a stability threshold
/// 3. Return split output: locked (immutable) tokens + volatile (changing) tokens
///
/// This creates a "typewriter" effect where text locks in from left to right,
/// rather than the entire hypothesis fluctuating with each update.
public struct HypothesisStabilizer: Sendable {

    /// A token with stability tracking metadata
    public struct TrackedToken: Sendable, Equatable {
        public let tokenId: Int
        public let timestamp: Int
        public let confidence: Float
        public var stabilityCount: Int
        public var isLocked: Bool

        public init(tokenId: Int, timestamp: Int, confidence: Float) {
            self.tokenId = tokenId
            self.timestamp = timestamp
            self.confidence = confidence
            self.stabilityCount = 1
            self.isLocked = false
        }

        public static func == (lhs: TrackedToken, rhs: TrackedToken) -> Bool {
            lhs.tokenId == rhs.tokenId && lhs.timestamp == rhs.timestamp
        }
    }

    /// Configuration for stability behavior
    public struct Config: Sendable {
        /// Number of consecutive updates a token must survive to become locked
        public let lockThreshold: Int

        /// Maximum number of volatile (unlocked) tokens to show
        public let maxVolatileTokens: Int

        /// Time tolerance in encoder frames for matching tokens across updates
        public let timestampTolerance: Int

        /// Minimum confidence for a token to contribute to stability
        public let minConfidenceForStability: Float

        /// If true, high-confidence tokens lock faster (threshold reduced by 1)
        public let fastLockHighConfidence: Bool

        public static let `default` = Config(
            lockThreshold: 3,
            maxVolatileTokens: 15,
            timestampTolerance: 5,
            minConfidenceForStability: 0.6,
            fastLockHighConfidence: true
        )

        /// More aggressive locking for lower latency but potentially less accuracy
        public static let aggressive = Config(
            lockThreshold: 2,
            maxVolatileTokens: 10,
            timestampTolerance: 3,
            minConfidenceForStability: 0.7,
            fastLockHighConfidence: true
        )

        /// Conservative locking for higher accuracy but more latency
        public static let conservative = Config(
            lockThreshold: 4,
            maxVolatileTokens: 20,
            timestampTolerance: 8,
            minConfidenceForStability: 0.5,
            fastLockHighConfidence: false
        )

        public init(
            lockThreshold: Int,
            maxVolatileTokens: Int,
            timestampTolerance: Int,
            minConfidenceForStability: Float,
            fastLockHighConfidence: Bool
        ) {
            self.lockThreshold = lockThreshold
            self.maxVolatileTokens = maxVolatileTokens
            self.timestampTolerance = timestampTolerance
            self.minConfidenceForStability = minConfidenceForStability
            self.fastLockHighConfidence = fastLockHighConfidence
        }
    }

    /// Result of stabilization process
    public struct StabilizedOutput: Sendable {
        /// Tokens that are locked (will never change)
        public let lockedTokens: [Int]
        public let lockedTimestamps: [Int]
        public let lockedConfidences: [Float]

        /// Tokens that are still volatile (may change in future updates)
        public let volatileTokens: [Int]
        public let volatileTimestamps: [Int]
        public let volatileConfidences: [Float]

        /// Total number of tokens (locked + volatile)
        public var totalCount: Int { lockedTokens.count + volatileTokens.count }

        /// All tokens combined (locked first, then volatile)
        public var allTokens: [Int] { lockedTokens + volatileTokens }
        public var allTimestamps: [Int] { lockedTimestamps + volatileTimestamps }
        public var allConfidences: [Float] { lockedConfidences + volatileConfidences }

        /// Number of locked tokens
        public var lockedCount: Int { lockedTokens.count }

        /// Whether any tokens changed from the previous update
        public var hasChanges: Bool

        /// Empty output for initialization
        public static let empty = StabilizedOutput(
            lockedTokens: [],
            lockedTimestamps: [],
            lockedConfidences: [],
            volatileTokens: [],
            volatileTimestamps: [],
            volatileConfidences: [],
            hasChanges: false
        )
    }

    private let config: Config

    private var trackedTokens: [TrackedToken] = []
    private var previousOutput: StabilizedOutput = .empty
    private var updateCount: Int = 0

    public init(config: Config = .default) {
        self.config = config
    }

    /// Process new hypothesis tokens and return stabilized output
    ///
    /// - Parameters:
    ///   - newTokens: Token IDs from the latest hypothesis decode
    ///   - timestamps: Encoder frame timestamps for each token
    ///   - confidences: Confidence scores for each token
    /// - Returns: Stabilized output with locked and volatile token splits
    public mutating func stabilize(
        newTokens: [Int],
        timestamps: [Int],
        confidences: [Float]
    ) -> StabilizedOutput {
        updateCount += 1

        guard !newTokens.isEmpty else {
            // Empty input - preserve locked tokens but clear volatile
            return createOutput(preserveVolatile: false)
        }

        guard newTokens.count == timestamps.count && newTokens.count == confidences.count else {
            // Token/timestamp/confidence count mismatch - return previous output
            return previousOutput
        }

        // Phase 1: Match new tokens against existing tracked tokens
        let matches = findMatches(newTokens: newTokens, timestamps: timestamps, confidences: confidences)

        // Phase 2: Update stability counts for matched tokens
        var newTrackedTokens: [TrackedToken] = []
        var matchedIndices = Set<Int>()

        // Process locked tokens first - they're immutable
        for tracked in trackedTokens where tracked.isLocked {
            newTrackedTokens.append(tracked)
            if let matchIdx = matches[newTrackedTokens.count - 1] {
                matchedIndices.insert(matchIdx)
            }
        }

        // Process unlocked tokens - update stability or reset
        let unlockedStart = newTrackedTokens.count
        for (idx, tracked) in trackedTokens.enumerated() where !tracked.isLocked {
            let originalIdx = unlockedStart + (idx - trackedTokens.filter { $0.isLocked }.count)
            if let matchIdx = matches[originalIdx] {
                // Token matched - increment stability
                var updated = tracked
                updated.stabilityCount += 1

                // Check if token should become locked
                let threshold = effectiveLockThreshold(for: updated)
                if updated.stabilityCount >= threshold {
                    updated.isLocked = true
                }

                newTrackedTokens.append(updated)
                matchedIndices.insert(matchIdx)
            }
            // Non-matched unlocked tokens are dropped (they fluctuated)
        }

        // Phase 3: Add new tokens that didn't match existing ones
        for (idx, tokenId) in newTokens.enumerated() {
            if !matchedIndices.contains(idx) {
                let newToken = TrackedToken(
                    tokenId: tokenId,
                    timestamp: timestamps[idx],
                    confidence: confidences[idx]
                )
                newTrackedTokens.append(newToken)
            }
        }

        // Phase 4: Sort by timestamp to maintain temporal order
        newTrackedTokens.sort { $0.timestamp < $1.timestamp }

        // Phase 5: Limit volatile tokens to maxVolatileTokens
        let lockedCount = newTrackedTokens.filter { $0.isLocked }.count
        let volatileCount = newTrackedTokens.count - lockedCount

        if volatileCount > config.maxVolatileTokens {
            // Remove oldest volatile tokens (keep newest)
            var keptTokens: [TrackedToken] = []
            var volatileKept = 0

            for token in newTrackedTokens.reversed() {
                if token.isLocked {
                    keptTokens.append(token)
                } else if volatileKept < config.maxVolatileTokens {
                    keptTokens.append(token)
                    volatileKept += 1
                }
            }

            newTrackedTokens = keptTokens.reversed()
        }

        trackedTokens = newTrackedTokens

        let output = createOutput(preserveVolatile: true)
        let hasChanges = output.allTokens != previousOutput.allTokens
        previousOutput = output

        return StabilizedOutput(
            lockedTokens: output.lockedTokens,
            lockedTimestamps: output.lockedTimestamps,
            lockedConfidences: output.lockedConfidences,
            volatileTokens: output.volatileTokens,
            volatileTimestamps: output.volatileTimestamps,
            volatileConfidences: output.volatileConfidences,
            hasChanges: hasChanges
        )
    }

    /// Find matches between new tokens and existing tracked tokens
    /// Returns a dictionary mapping tracked token index to new token index
    private func findMatches(
        newTokens: [Int],
        timestamps: [Int],
        confidences: [Float]
    ) -> [Int: Int] {
        var matches: [Int: Int] = [:]
        var usedNewIndices = Set<Int>()

        // Match locked tokens first (they have priority)
        for (trackedIdx, tracked) in trackedTokens.enumerated() where tracked.isLocked {
            if let newIdx = findBestMatch(
                for: tracked,
                in: newTokens,
                timestamps: timestamps,
                confidences: confidences,
                excluding: usedNewIndices
            ) {
                matches[trackedIdx] = newIdx
                usedNewIndices.insert(newIdx)
            }
        }

        // Match unlocked tokens
        for (trackedIdx, tracked) in trackedTokens.enumerated() where !tracked.isLocked {
            if let newIdx = findBestMatch(
                for: tracked,
                in: newTokens,
                timestamps: timestamps,
                confidences: confidences,
                excluding: usedNewIndices
            ) {
                matches[trackedIdx] = newIdx
                usedNewIndices.insert(newIdx)
            }
        }

        return matches
    }

    /// Find the best matching new token for a tracked token
    private func findBestMatch(
        for tracked: TrackedToken,
        in newTokens: [Int],
        timestamps: [Int],
        confidences: [Float],
        excluding usedIndices: Set<Int>
    ) -> Int? {
        var bestMatch: Int?
        var bestScore = Float.infinity

        for (idx, tokenId) in newTokens.enumerated() {
            guard !usedIndices.contains(idx) else { continue }
            guard tokenId == tracked.tokenId else { continue }

            let timestampDiff = abs(timestamps[idx] - tracked.timestamp)
            guard timestampDiff <= config.timestampTolerance else { continue }

            // Score based on timestamp proximity (lower is better)
            let score = Float(timestampDiff)

            if score < bestScore {
                bestScore = score
                bestMatch = idx
            }
        }

        return bestMatch
    }

    /// Calculate effective lock threshold for a token
    private func effectiveLockThreshold(for token: TrackedToken) -> Int {
        if config.fastLockHighConfidence && token.confidence >= 0.9 {
            return max(1, config.lockThreshold - 1)
        }
        return config.lockThreshold
    }

    /// Create output from current tracked tokens
    private func createOutput(preserveVolatile: Bool) -> StabilizedOutput {
        var lockedTokens: [Int] = []
        var lockedTimestamps: [Int] = []
        var lockedConfidences: [Float] = []
        var volatileTokens: [Int] = []
        var volatileTimestamps: [Int] = []
        var volatileConfidences: [Float] = []

        for token in trackedTokens {
            if token.isLocked {
                lockedTokens.append(token.tokenId)
                lockedTimestamps.append(token.timestamp)
                lockedConfidences.append(token.confidence)
            } else if preserveVolatile {
                volatileTokens.append(token.tokenId)
                volatileTimestamps.append(token.timestamp)
                volatileConfidences.append(token.confidence)
            }
        }

        return StabilizedOutput(
            lockedTokens: lockedTokens,
            lockedTimestamps: lockedTimestamps,
            lockedConfidences: lockedConfidences,
            volatileTokens: volatileTokens,
            volatileTimestamps: volatileTimestamps,
            volatileConfidences: volatileConfidences,
            hasChanges: true
        )
    }

    /// Reset the stabilizer state
    public mutating func reset() {
        trackedTokens.removeAll()
        previousOutput = .empty
        updateCount = 0
    }

    /// Called when confirmed tokens are promoted from the main track
    /// Removes locked tokens that have been confirmed
    public mutating func acknowledgeConfirmed(upToTimestamp: Int) {
        trackedTokens.removeAll { $0.isLocked && $0.timestamp <= upToTimestamp }
    }

    /// Get current statistics for debugging
    public var statistics: (locked: Int, volatile: Int, updates: Int) {
        let locked = trackedTokens.filter { $0.isLocked }.count
        let volatile = trackedTokens.count - locked
        return (locked, volatile, updateCount)
    }
}
