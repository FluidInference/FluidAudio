import Foundation

/// Reason why text was committed to finalized state.
public enum CommitReason: Sendable, Equatable {
    /// Text was committed because punctuation was detected (., !, or ?).
    case punctuation(Character)
    /// Text was committed because the debounce timeout expired.
    case debounceTimeout
    /// Text was committed by an explicit manual commit call.
    case manualCommit
    /// Text was committed because an End-of-Utterance signal was received.
    case endOfUtterance
}

/// Update delivered by PunctuationCommitLayer containing committed and ghost text.
public struct CommitLayerUpdate: Sendable {
    /// Finalized text that has been committed (at punctuation boundaries or EOU).
    public let committedText: String

    /// Speculative text that has not yet been committed.
    public let ghostText: String

    /// Combined text (committedText + ghostText).
    public let totalText: String

    /// Reason for the most recent commit, if any.
    public let lastCommitReason: CommitReason?

    /// Timestamp when this update was created.
    public let timestamp: Date

    public init(
        committedText: String,
        ghostText: String,
        totalText: String,
        lastCommitReason: CommitReason?,
        timestamp: Date
    ) {
        self.committedText = committedText
        self.ghostText = ghostText
        self.totalText = totalText
        self.lastCommitReason = lastCommitReason
        self.timestamp = timestamp
    }
}

/// Actor-based layer that wraps streaming ASR results to provide punctuation-aware text segmentation.
///
/// This layer accumulates text until punctuation marks (`.`, `!`, `?`) are detected, separating
/// text into "committed" (finalized) and "ghost" (speculative) portions. It handles debounce
/// timeouts for mid-sentence pauses with configurable commit behavior.
///
/// # Usage Example
///
/// ```swift
/// let engine = StreamingModelVariant.parakeetEou160ms.createManager()
/// try await engine.loadModels()
///
/// let commitLayer = PunctuationCommitLayer(
///     debounceTimeout: 3.0,
///     commitOnTimeout: true
/// )
///
/// engine.setPartialTranscriptCallback { partial in
///     Task {
///         let update = await commitLayer.processPartialText(partial)
///         print("Committed: \(update.committedText)")
///         print("Ghost: \(update.ghostText)")
///     }
/// }
///
/// engine.setEouCallback {
///     Task {
///         let update = await commitLayer.processEOU()
///         print("Final: \(update.committedText)")
///     }
/// }
/// ```
///
/// # Thread Safety
///
/// `PunctuationCommitLayer` is an actor, providing Swift 6 concurrency safety:
/// - Committed text is Sendable and safe to share across actors
/// - Ghost text is isolated within the actor until promoted to committed
/// - All callbacks are `@Sendable` closures
///
/// # Design Rationale
///
/// This design addresses the challenge of providing responsive real-time transcription UX
/// while maintaining sentence-aware text segmentation. See GitHub issue discussion at
/// https://github.com/FluidInference/FluidAudio/issues/415 for architectural context.
public actor PunctuationCommitLayer {
    // MARK: - Configuration

    /// Duration to wait before committing ghost text on timeout (in seconds).
    private let debounceTimeout: TimeInterval

    /// Set of punctuation marks that trigger commits.
    private let punctuationMarks: Set<Character>

    /// Whether to commit ghost text when debounce timeout expires.
    private let commitOnTimeout: Bool

    // MARK: - State

    /// Finalized text that has been committed.
    private var committedText: String = ""

    /// Speculative text that has not yet been committed.
    private var ghostText: String = ""

    /// Timestamp of the last update.
    private var lastUpdateTime: Date = Date()

    /// Active debounce timer task.
    private var debounceTask: Task<Void, Never>?

    /// Callback invoked when updates occur.
    private var updateCallback: (@Sendable (CommitLayerUpdate) -> Void)?

    // MARK: - Initialization

    /// Creates a new punctuation commit layer.
    ///
    /// - Parameters:
    ///   - debounceTimeout: Duration to wait before committing ghost text on timeout (default: 3.0 seconds).
    ///   - punctuationMarks: Set of punctuation marks that trigger commits (default: `.`, `!`, `?`).
    ///   - commitOnTimeout: Whether to commit ghost text when debounce timeout expires (default: `true`).
    public init(
        debounceTimeout: TimeInterval = 3.0,
        punctuationMarks: Set<Character> = [".", "!", "?"],
        commitOnTimeout: Bool = true
    ) {
        self.debounceTimeout = debounceTimeout
        self.punctuationMarks = punctuationMarks
        self.commitOnTimeout = commitOnTimeout
    }

    // MARK: - Public API

    /// Processes partial text from streaming ASR and returns an update with committed/ghost text.
    ///
    /// This method detects punctuation marks and splits text accordingly:
    /// - Text up to and including the last punctuation mark is committed
    /// - Remaining text becomes ghost text
    /// - If no punctuation is found, all text is treated as ghost text
    ///
    /// - Parameter text: The partial transcription text from the ASR engine.
    /// - Returns: Update containing committed text, ghost text, and commit reason.
    public func processPartialText(_ text: String) async -> CommitLayerUpdate {
        // Cancel existing debounce timer
        debounceTask?.cancel()
        lastUpdateTime = Date()

        // Find last punctuation mark in text
        if let lastPunc = text.lastIndex(where: { punctuationMarks.contains($0) }) {
            let commitIndex = text.index(after: lastPunc)

            // Skip whitespace after punctuation when determining ghost text start
            var ghostStart = commitIndex
            while ghostStart < text.endIndex && text[ghostStart].isWhitespace {
                ghostStart = text.index(after: ghostStart)
            }

            let newGhostText = String(text[ghostStart...])

            // Preserve whitespace after punctuation, or add one space if none exists
            let whitespaceAfterPunc = String(text[commitIndex..<ghostStart])
            let whitespace = whitespaceAfterPunc.isEmpty ? " " : whitespaceAfterPunc

            let textToCommit = String(text[..<commitIndex])
            committedText += textToCommit + whitespace
            ghostText = newGhostText

            let totalText: String
            if ghostText.isEmpty {
                totalText = committedText
            } else if committedText.isEmpty {
                totalText = ghostText
            } else {
                // committedText already ends with whitespace from line 172
                totalText = committedText + ghostText
            }

            let update = CommitLayerUpdate(
                committedText: committedText,
                ghostText: ghostText,
                totalText: totalText,
                lastCommitReason: .punctuation(text[lastPunc]),
                timestamp: Date()
            )

            updateCallback?(update)
            return update
        } else {
            // No punctuation: all text is ghost
            ghostText = text
            startDebounceTimer()

            let totalText: String
            if committedText.isEmpty {
                totalText = ghostText
            } else if ghostText.isEmpty {
                totalText = committedText
            } else {
                totalText = committedText + ghostText
            }

            let update = CommitLayerUpdate(
                committedText: committedText,
                ghostText: ghostText,
                totalText: totalText,
                lastCommitReason: nil,
                timestamp: Date()
            )

            updateCallback?(update)
            return update
        }
    }

    /// Processes an End-of-Utterance (EOU) signal, committing all ghost text.
    ///
    /// - Returns: Update with all text committed and EOU commit reason.
    public func processEOU() async -> CommitLayerUpdate {
        debounceTask?.cancel()
        lastUpdateTime = Date()

        // EOU signals end of utterance: commit everything
        guard !ghostText.isEmpty else {
            let update = CommitLayerUpdate(
                committedText: committedText,
                ghostText: ghostText,
                totalText: committedText,
                lastCommitReason: .endOfUtterance,
                timestamp: Date()
            )
            updateCallback?(update)
            return update
        }

        // Add separator space if committed text is not empty and doesn't already end with whitespace
        if !committedText.isEmpty && committedText.last?.isWhitespace == false {
            committedText += " "
        }
        committedText += ghostText
        ghostText = ""

        let update = CommitLayerUpdate(
            committedText: committedText,
            ghostText: ghostText,
            totalText: committedText,
            lastCommitReason: .endOfUtterance,
            timestamp: Date()
        )

        updateCallback?(update)
        return update
    }

    /// Manually commits all ghost text immediately.
    ///
    /// - Returns: Update with ghost text promoted to committed text.
    public func manualCommit() async -> CommitLayerUpdate {
        debounceTask?.cancel()
        lastUpdateTime = Date()

        guard !ghostText.isEmpty else {
            let update = CommitLayerUpdate(
                committedText: committedText,
                ghostText: ghostText,
                totalText: committedText,
                lastCommitReason: .manualCommit,
                timestamp: Date()
            )
            updateCallback?(update)
            return update
        }

        // Add separator space if committed text is not empty and doesn't already end with whitespace
        if !committedText.isEmpty && committedText.last?.isWhitespace == false {
            committedText += " "
        }
        committedText += ghostText
        ghostText = ""

        let update = CommitLayerUpdate(
            committedText: committedText,
            ghostText: ghostText,
            totalText: committedText,
            lastCommitReason: .manualCommit,
            timestamp: Date()
        )

        updateCallback?(update)
        return update
    }

    /// Resets the commit layer, clearing all committed and ghost text.
    public func reset() async {
        debounceTask?.cancel()
        committedText = ""
        ghostText = ""
        lastUpdateTime = Date()

        let update = CommitLayerUpdate(
            committedText: committedText,
            ghostText: ghostText,
            totalText: committedText,
            lastCommitReason: nil,
            timestamp: Date()
        )

        updateCallback?(update)
    }

    /// Sets a callback to be invoked when updates occur.
    ///
    /// - Parameter callback: Closure called with `CommitLayerUpdate` on each update.
    public func setUpdateCallback(_ callback: @escaping @Sendable (CommitLayerUpdate) -> Void) {
        self.updateCallback = callback
    }

    // MARK: - Private Helpers

    /// Starts a debounce timer that commits ghost text after the timeout expires.
    private func startDebounceTimer() {
        debounceTask?.cancel()

        debounceTask = Task { [weak self, debounceTimeout, commitOnTimeout] in
            try? await Task.sleep(nanoseconds: UInt64(debounceTimeout * 1_000_000_000))

            guard !Task.isCancelled else { return }
            guard let self = self else { return }

            // Timeout expired
            if commitOnTimeout {
                await self.commitGhostText(reason: .debounceTimeout)
            }
        }
    }

    /// Commits ghost text to committed text with the specified reason.
    ///
    /// - Parameter reason: The reason for committing.
    private func commitGhostText(reason: CommitReason) async {
        guard !ghostText.isEmpty else { return }

        // Add separator space if committed text is not empty and doesn't already end with whitespace
        if !committedText.isEmpty && committedText.last?.isWhitespace == false {
            committedText += " "
        }
        committedText += ghostText
        ghostText = ""

        let update = CommitLayerUpdate(
            committedText: committedText,
            ghostText: ghostText,
            totalText: committedText,
            lastCommitReason: reason,
            timestamp: Date()
        )

        updateCallback?(update)
    }
}
