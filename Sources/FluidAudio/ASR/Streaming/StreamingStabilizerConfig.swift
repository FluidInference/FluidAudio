import Foundation

@available(macOS 13.0, iOS 16.0, *)
public enum StreamingStabilizerProfile: String, Sendable, CaseIterable {
    case balanced
    case lowLatency = "low-latency"
    case highStability = "high-stability"
}

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
        windowSize: Int = 4,
        emitWordBoundaries: Bool = true,
        maxWaitMilliseconds: Int = 1_200,
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
    public static func preset(_ profile: StreamingStabilizerProfile) -> StreamingStabilizerConfig {
        switch profile {
        case .balanced:
            return StreamingStabilizerConfig(
                windowSize: 3,
                emitWordBoundaries: true,
                maxWaitMilliseconds: 800,
                tokenizerKind: .sentencePiece,
                debugDumpEnabled: false
            )
        case .lowLatency:
            return StreamingStabilizerConfig(
                windowSize: 2,
                emitWordBoundaries: false,
                maxWaitMilliseconds: 450,
                tokenizerKind: .sentencePiece,
                debugDumpEnabled: false
            )
        case .highStability:
            return StreamingStabilizerConfig(
                windowSize: 4,
                emitWordBoundaries: true,
                maxWaitMilliseconds: 1200,
                tokenizerKind: .sentencePiece,
                debugDumpEnabled: false
            )
        }
    }

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
