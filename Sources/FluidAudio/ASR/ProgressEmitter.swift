import Foundation

actor ProgressEmitter {
    private var continuation: AsyncThrowingStream<Double, Error>.Continuation?
    private var streamStorage: AsyncThrowingStream<Double, Error>?
    private var isActive = false

    init() {}

    func startSession() async throws -> AsyncThrowingStream<Double, Error> {
        guard !isActive else {
            throw ASRError.processingFailed("A transcription session is already in progress.")
        }

        let (stream, continuation) = AsyncThrowingStream<Double, Error>.makeStream()
        self.streamStorage = stream
        self.continuation = continuation
        self.isActive = true

        continuation.onTermination =
            { [weak self] (_: AsyncThrowingStream<Double, Error>.Continuation.Termination) in
                Task { [weak self] in
                    guard let self else { return }
                    await self.reset()
                }
            }

        continuation.yield(0.0)
        return stream
    }

    func currentStream() async -> AsyncThrowingStream<Double, Error>? {
        streamStorage
    }

    func report(progress: Double) async {
        guard isActive else { return }
        let clamped = min(max(progress, 0.0), 1.0)
        continuation?.yield(clamped)
    }

    func finishSession() async {
        if isActive {
            continuation?.yield(1.0)
        }
        continuation?.finish()
        await reset()
    }

    func failSession(_ error: Error) async {
        continuation?.finish(throwing: error)
        await reset()
    }

    private func reset() async {
        continuation = nil
        streamStorage = nil
        isActive = false
    }
}
