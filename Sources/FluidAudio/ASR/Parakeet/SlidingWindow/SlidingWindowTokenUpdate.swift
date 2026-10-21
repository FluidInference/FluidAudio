import Foundation

/// Producer-owned edit positions in one sliding-window job. Confidence is not
/// an immutability promise: only tokens before `stableTokenCount` are irrevocable.
public struct SlidingWindowTokenRevision: Sendable {
    /// Identifies one native job; revisions from different jobs never share lineage.
    public let jobID: UUID
    /// Contiguous publication number within the job, starting at 1.
    public let ordinal: Int
    /// Token count the consumer held before applying this revision.
    public let previousTokenCount: Int
    /// Leading tokens kept from the previous count; the rest are retired.
    public let retainedTokenCount: Int
    /// Leading tokens that are irrevocable; this count never decreases.
    public let stableTokenCount: Int
    /// Audio samples the decoder read for this revision.
    public let window: Range<Int>
    /// Samples this revision is authoritative for, within `window`.
    public let center: Range<Int>
    /// The final revision of the job; every token is stable.
    public let isTerminal: Bool

    public init(
        jobID: UUID, ordinal: Int, previousTokenCount: Int, retainedTokenCount: Int,
        stableTokenCount: Int, window: Range<Int>, center: Range<Int>, isTerminal: Bool
    ) {
        self.jobID = jobID
        self.ordinal = ordinal
        self.previousTokenCount = previousTokenCount
        self.retainedTokenCount = retainedTokenCount
        self.stableTokenCount = stableTokenCount
        self.window = window
        self.center = center
        self.isTerminal = isTerminal
    }
}

/// Raw decoder tokens, not vocabulary-rescored display text. An empty event can
/// retire a suffix or finalize a job and must not be discarded.
public struct SlidingWindowTokenUpdate: Sendable {
    /// Tokens appended after the retained prefix.
    public let tokens: [TokenTiming]
    /// Where `tokens` attach and which prefix is now stable.
    public let revision: SlidingWindowTokenRevision
    /// Legacy display-confidence flag, not token immutability authority.
    public let isConfirmed: Bool

    public init(tokens: [TokenTiming], revision: SlidingWindowTokenRevision, isConfirmed: Bool = false) {
        self.tokens = tokens
        self.revision = revision
        self.isConfirmed = isConfirmed
    }
}

public enum SlidingWindowTokenStreamError: Error, Sendable {
    /// Another subscriber already owns this job's token stream.
    case multipleSubscribers
    /// The job already accepted audio, finished or was cancelled.
    case unavailableJob
    /// The decoder produced an update that cannot be published.
    case invalidNativeUpdate
    /// A revision would break ordering or shrink the stable prefix.
    case invalidLineage
}

/// Shared by real publication and contract tests; no text/timestamp deduplication.
struct SlidingWindowTokenCursor {
    let jobID = UUID()
    private(set) var ordinal = 0
    private(set) var count = 0
    private(set) var stable = 0
    private(set) var isTerminal = false
    private(set) var window = 0..<0
    private(set) var center = 0..<0

    mutating func next(
        tokens: [TokenTiming], previous: Int, retained: Int, stable: Int,
        window: Range<Int>, center: Range<Int>, terminal: Bool = false, isConfirmed: Bool = false
    ) throws
        -> SlidingWindowTokenUpdate
    {
        let total = retained.addingReportingOverflow(tokens.count)
        let ordinal = self.ordinal.addingReportingOverflow(1)
        guard !isTerminal, !total.overflow, !ordinal.overflow, previous == count,
            retained >= self.stable, retained <= previous, stable >= self.stable,
            stable <= total.partialValue, !terminal || stable == total.partialValue,
            window.lowerBound >= 0, window.lowerBound <= center.lowerBound,
            center.upperBound <= window.upperBound,
            // A drain reads less right context, so only the center must advance.
            center.lowerBound >= self.center.lowerBound, center.upperBound >= self.center.upperBound,
            tokens.allSatisfy({
                $0.startTime.isFinite && $0.endTime.isFinite && $0.startTime >= 0
                    && $0.endTime >= $0.startTime && $0.confidence.isFinite && (0...1).contains($0.confidence)
            }),
            zip(tokens, tokens.dropFirst()).allSatisfy({ $0.startTime <= $1.startTime })
        else {
            throw SlidingWindowTokenStreamError.invalidLineage
        }
        let revision = SlidingWindowTokenRevision(
            jobID: jobID, ordinal: ordinal.partialValue, previousTokenCount: previous,
            retainedTokenCount: retained, stableTokenCount: stable, window: window,
            center: center, isTerminal: terminal)
        self.ordinal = ordinal.partialValue
        count = total.partialValue
        self.stable = stable
        self.window = window
        self.center = center
        isTerminal = terminal
        return SlidingWindowTokenUpdate(tokens: tokens, revision: revision, isConfirmed: isConfirmed)
    }

    mutating func finish() throws -> SlidingWindowTokenUpdate {
        try next(
            tokens: [], previous: count, retained: count, stable: count,
            window: window, center: center, terminal: true)
    }
}
