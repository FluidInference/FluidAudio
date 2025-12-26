import Foundation

actor ProgressEmitter {
    private var continuation: AsyncThrowingStream<Double, Error>.Continuation?
    private var streamStorage: AsyncThrowingStream<Double, Error>?
    private var isActive = false

    init() {}

    func ensureSession() async -> AsyncThrowingStream<Double, Error> {
        if let stream = streamStorage {
            return stream
        }
        return await startSession()
    }

    func currentStream() async -> AsyncThrowingStream<Double, Error> {
        await ensureSession()
    }

    func report(progress: Double) async {
        guard isActive else { return }
        let clamped = min(max(progress, 0.0), 1.0)
        continuation?.yield(clamped)
    }

    func finishSession() async {
        guard isActive else {
            _ = await ensureSession()
            return
        }

        continuation?.yield(1.0)
        continuation?.finish()
    }

    func failSession(_ error: Error) async {
        continuation?.finish(throwing: error)
    }

    private func startSession() async -> AsyncThrowingStream<Double, Error> {
        if let stream = streamStorage {
            return stream
        }

        let (stream, continuation) = AsyncThrowingStream<Double, Error>.makeStream()
        self.streamStorage = stream
        self.continuation = continuation
        self.isActive = true

        continuation.onTermination =
            { [weak self] (_: AsyncThrowingStream<Double, Error>.Continuation.Termination) in
                Task { [weak self] in
                    guard let self else { return }
                    await self.resetAndPrepareNextSession()
                }
            }

        continuation.yield(0.0)
        return stream
    }

    private func resetAndPrepareNextSession() async {
        continuation = nil
        streamStorage = nil
        isActive = false
        _ = await startSession()
    }
}
