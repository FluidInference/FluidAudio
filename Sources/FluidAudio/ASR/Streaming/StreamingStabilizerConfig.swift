import Foundation

@available(macOS 13.0, iOS 16.0, *)
public struct StreamingStabilizerConfig: Sendable, Equatable {
    public enum TokenizerKind: String, Sendable {
        case sentencePiece
        case bytePairEncoding
        case wordPiece
    }

    public let windowSize: Int
    public let emitWordBoundaries: Bool
    public let maxWaitMilliseconds: Int
    public let tokenizerKind: TokenizerKind
    public let debugDumpEnabled: Bool

    public init(
        windowSize: Int = 4,  // 4-window consensus (highest stability)
        emitWordBoundaries: Bool = true,
        maxWaitMilliseconds: Int = 1_200,  // wait up to 1.2 s before confirming
        tokenizerKind: TokenizerKind = .sentencePiece,
        debugDumpEnabled: Bool = false
    ) {
        self.windowSize = windowSize
        self.emitWordBoundaries = emitWordBoundaries
        self.maxWaitMilliseconds = maxWaitMilliseconds
        self.tokenizerKind = tokenizerKind
        self.debugDumpEnabled = debugDumpEnabled
    }

    internal var sanitizedWindowSize: Int { max(1, windowSize) }
    internal var sanitizedMaxWait: Int { max(0, maxWaitMilliseconds) }
}

@available(macOS 13.0, iOS 16.0, *)
extension StreamingStabilizerConfig {
    /// Prioritizes stability by requiring a four-window consensus and full word-boundary trimming.
    public static let highAccuracy = StreamingStabilizerConfig()

    /// Prefers shorter latency by shrinking the consensus window and wait budget.
    public static let lowLatency = StreamingStabilizerConfig(windowSize: 3, maxWaitMilliseconds: 600)

    /// Fluent copy helpers keep the struct immutable while making configuration concise.
    public func withDebugDumpEnabled(_ enabled: Bool) -> StreamingStabilizerConfig {
        StreamingStabilizerConfig(
            windowSize: windowSize,
            emitWordBoundaries: emitWordBoundaries,
            maxWaitMilliseconds: maxWaitMilliseconds,
            tokenizerKind: tokenizerKind,
            debugDumpEnabled: enabled
        )
    }

    public func withMaxWaitMilliseconds(_ value: Int) -> StreamingStabilizerConfig {
        StreamingStabilizerConfig(
            windowSize: windowSize,
            emitWordBoundaries: emitWordBoundaries,
            maxWaitMilliseconds: value,
            tokenizerKind: tokenizerKind,
            debugDumpEnabled: debugDumpEnabled
        )
    }

    public func withTokenizerKind(_ kind: TokenizerKind) -> StreamingStabilizerConfig {
        StreamingStabilizerConfig(
            windowSize: windowSize,
            emitWordBoundaries: emitWordBoundaries,
            maxWaitMilliseconds: maxWaitMilliseconds,
            tokenizerKind: kind,
            debugDumpEnabled: debugDumpEnabled
        )
    }
}
