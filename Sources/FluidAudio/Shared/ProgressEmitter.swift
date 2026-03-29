import Foundation

actor ProgressEmitter {
    private var continuation: AsyncThrowingStream<Double, Error>.Continuation?
    private var streamStorage: AsyncThrowingStream<Double, Error>?
    private var isActive = false

    init() {}

    func ensureSession() -> AsyncThrowingStream<Double, Error> {
        if let stream = streamStorage {
            return stream
        }
        return startSession()
    }

    func report(progress: Double) {
        guard isActive else { return }
        let clamped = min(max(progress, 0.0), 1.0)
        continuation?.yield(clamped)
    }

    func finishSession() {
        guard isActive else { return }

        continuation?.yield(1.0)
        continuation?.finish()
        reset()
    }

    func failSession(_ error: Error) {
        continuation?.finish(throwing: error)
        reset()
    }

    private func startSession() -> AsyncThrowingStream<Double, Error> {
        if let stream = streamStorage {
            return stream
        }

        let (stream, continuation) = AsyncThrowingStream<Double, Error>.makeStream()
        self.streamStorage = stream
        self.continuation = continuation
        self.isActive = true

        continuation.yield(0.0)
        return stream
    }

    private func reset() {
        continuation = nil
        streamStorage = nil
        isActive = false
    }
}
