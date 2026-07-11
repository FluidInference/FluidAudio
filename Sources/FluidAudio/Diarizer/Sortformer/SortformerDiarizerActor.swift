import CoreML
import Dispatch
import Foundation

/// A serial executor backed by a dedicated dispatch queue.
///
/// Actor jobs run as Swift concurrency jobs on this queue, so task cancellation and actor
/// isolation remain intact while blocking CoreML inference stays off the cooperative pool.
private final class SortformerInferenceExecutor: SerialExecutor {
    private let queue = DispatchQueue(
        label: "com.fluidaudio.sortformer.inference",
        qos: .userInitiated
    )

    func enqueue(_ job: consuming ExecutorJob) {
        let unownedJob = UnownedJob(job)
        let executor = asUnownedSerialExecutor()
        queue.async {
            unownedJob.runSynchronously(on: executor)
        }
    }
}

/// Async, actor-isolated streaming Sortformer diarization.
///
/// `SortformerDiarizerActor` creates and exclusively owns a synchronous
/// ``SortformerDiarizer``. Its custom serial executor runs every actor job on a dedicated FIFO
/// dispatch queue, so CoreML inference never pins a cooperative-pool thread. No non-Sendable
/// diarizer reference crosses an isolation boundary.
///
/// Use ``SortformerDiarizer`` directly from synchronous code. Use this actor when driving a
/// streaming diarizer from Swift concurrency.
public actor SortformerDiarizerActor {
    private nonisolated let inferenceExecutor = SortformerInferenceExecutor()
    private let diarizer: SortformerDiarizer

    public nonisolated let config: SortformerConfig

    public nonisolated var unownedExecutor: UnownedSerialExecutor {
        inferenceExecutor.asUnownedSerialExecutor()
    }

    public init(
        config: SortformerConfig = .default,
        timelineConfig: DiarizerTimelineConfig = .sortformerDefault
    ) {
        self.config = config
        self.diarizer = SortformerDiarizer(config: config, timelineConfig: timelineConfig)
    }

    /// Loads and initializes a combined Sortformer model from a local model package.
    public func initialize(
        mainModelPath: URL,
        computeUnits: MLComputeUnits? = nil
    ) async throws {
        let models = try await SortformerModels.load(
            config: config,
            mainModelPath: mainModelPath,
            computeUnits: computeUnits
        )
        diarizer.initialize(models: models)
    }

    /// Downloads (when needed), loads, and initializes a combined Sortformer model.
    public func initializeFromHuggingFace(
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits? = nil
    ) async throws {
        let models = try await SortformerModels.loadFromHuggingFace(
            config: config,
            cacheDirectory: cacheDirectory,
            computeUnits: computeUnits
        )
        diarizer.initialize(models: models)
    }

    /// Whether models are loaded and processing can begin.
    public var isAvailable: Bool {
        diarizer.isAvailable
    }

    /// Current streaming state copied out of the owned diarizer.
    public var state: SortformerStreamingState {
        diarizer.state
    }

    /// A Sendable value snapshot of the current timeline.
    public func timelineSnapshot() -> DiarizerTimeline.Snapshot {
        diarizer.timeline.takeSnapshot()
    }

    /// Resets the owned diarizer for a new stream.
    public func reset() {
        diarizer.reset()
    }

    /// Processes already-buffered audio on the dedicated serial executor.
    @discardableResult
    public func process() throws -> DiarizerTimelineUpdate? {
        try Task.checkCancellation()
        return try diarizer.process()
    }

    /// Adds and processes one audio chunk on the dedicated serial executor.
    @discardableResult
    public func process<C: Collection & Sendable>(
        samples: C,
        sourceSampleRate: Double? = nil
    ) throws -> DiarizerTimelineUpdate? where C.Element == Float {
        try Task.checkCancellation()
        return try diarizer.process(samples: samples, sourceSampleRate: sourceSampleRate)
    }

    /// Drains and finalizes the current stream on the dedicated serial executor.
    @discardableResult
    public func finalizeSession() throws -> DiarizerTimelineUpdate? {
        try Task.checkCancellation()
        return try diarizer.finalizeSession()
    }
}
