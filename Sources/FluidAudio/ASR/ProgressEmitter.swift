import Foundation

actor ProgressEmitter {
    private var continuation: AsyncStream<Double>.Continuation?
    private var streamStorage: AsyncStream<Double>?
    private var isActive = false

    init() {}

    func startSession() async throws -> AsyncStream<Double> {
        guard !isActive else {
            throw ASRError.processingFailed("A transcription session is already in progress.")
        }

        let (stream, continuation) = AsyncStream<Double>.makeStream()
        self.streamStorage = stream
        self.continuation = continuation
        self.isActive = true

        continuation.onTermination = { @Sendable _ in
            Task { [weak self] in
                guard let self else { return }
                await self.reset()
            }
        }

        continuation.yield(0.0)
        return stream
    }

    func currentStream() async -> AsyncStream<Double>? {
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

    func failSession() async {
        continuation?.finish()
        await reset()
    }

    private func reset() async {
        continuation = nil
        streamStorage = nil
        isActive = false
    }
}
