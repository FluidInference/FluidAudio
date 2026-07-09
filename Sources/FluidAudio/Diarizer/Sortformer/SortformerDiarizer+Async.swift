import Foundation

/// Async-friendly inference entry points for `SortformerDiarizer`.
///
/// ## Why these exist
/// The synchronous `process()` / `process(samples:)` / `finalizeSession()` /
/// `processComplete(_:)` methods run CoreML inference inline on whatever thread calls them,
/// with no suspension point. When the caller is a Swift-concurrency actor, that inference
/// occupies one of the (few) shared cooperative-pool threads to completion — a ~100 ms–1 s
/// forward pass starves every other actor in the process for its duration.
///
/// The `*Async` variants below run the *same* code on the diarizer's own dedicated serial
/// dispatch queue (`inferenceQueue`). The calling task suspends across the hop, immediately
/// returning its cooperative-pool thread, and resumes when inference completes.
///
/// ## Determinism
/// `inferenceQueue` is serial and FIFO, and each forwarded call acquires the diarizer's
/// internal lock exactly like its synchronous twin. A sequence of awaited `*Async` calls
/// therefore produces byte-identical output to the same sequence of synchronous calls.
///
/// ## Cancellation
/// Each variant checks `Task.checkCancellation()` *before* enqueueing work. Once a forward
/// pass has started on the queue it runs to completion (CoreML predictions are not
/// interruptible); the cooperative `Task.checkCancellation()` inside the synchronous
/// implementations is a no-op on the queue thread because no task context exists there.
extension SortformerDiarizer {

    // MARK: - Async Streaming Processing

    /// Async variant of ``process()``: processes buffered audio on the diarizer's serial
    /// inference queue while the calling task suspends.
    ///
    /// - Returns: New chunk results if enough audio was processed, `nil` otherwise.
    @discardableResult
    public func processAsync() async throws -> DiarizerTimelineUpdate? {
        try Task.checkCancellation()
        return try await runOnInferenceQueue { diarizer in
            try diarizer.process()
        }
    }

    /// Async variant of ``process(samples:sourceSampleRate:)``: adds and processes a chunk of
    /// audio on the diarizer's serial inference queue while the calling task suspends.
    ///
    /// - Parameters:
    ///   - samples: Mono audio samples to process.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    /// - Returns: New chunk results if enough audio was processed, `nil` otherwise.
    @discardableResult
    public func processAsync(
        samples: some Collection<Float>,
        sourceSampleRate: Double? = nil
    ) async throws -> DiarizerTimelineUpdate? {
        try Task.checkCancellation()
        let buffered = Array(samples)
        return try await runOnInferenceQueue { diarizer in
            try diarizer.process(samples: buffered, sourceSampleRate: sourceSampleRate)
        }
    }

    /// Async variant of ``finalizeSession()``: drains and finalizes on the diarizer's serial
    /// inference queue while the calling task suspends.
    ///
    /// Idempotent like its synchronous twin: subsequent calls return `nil`.
    @discardableResult
    public func finalizeSessionAsync() async throws -> DiarizerTimelineUpdate? {
        try Task.checkCancellation()
        return try await runOnInferenceQueue { diarizer in
            try diarizer.finalizeSession()
        }
    }

    // MARK: - Queue Hop

    /// Runs `work` on the diarizer's serial inference queue, suspending the calling task until
    /// it completes. `SortformerDiarizer` is `@unchecked Sendable` (all mutable state is
    /// lock-guarded), so handing `self` to the queue closure is safe.
    private func runOnInferenceQueue<T: Sendable>(
        _ work: @escaping @Sendable (SortformerDiarizer) throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            inferenceQueue.async { [self] in
                continuation.resume(with: Result { try work(self) })
            }
        }
    }
}
