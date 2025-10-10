import Foundation

struct StreamingVadSegment: Sendable {
    let samples: [Float]
    let cumulativeDroppedSamples: Int
    let isSpeechOnset: Bool
}

enum StreamingVadEvent: Sendable {
    case speech(StreamingVadSegment)
    case silence(samples: Int, cumulativeDroppedSamples: Int)
}

final class StreamingVadPipeline {
    /// Hard limit for overflow buffer to prevent unbounded growth (seconds).
    /// Triggers forced emission when pre-speech buffer exceeds this duration.
    private static let overflowHardLimitSeconds: Double = 2.0

    /// Minimum reserved capacity for residual audio buffering (samples).
    /// Sized to cover four VAD chunks to avoid immediate reallocations.
    private static let residualPreallocationSamples: Int = VadManager.chunkSize * 4

    /// Minimum reserved capacity for pre-speech buffering (samples).
    /// Allows capture of left-context audio plus two chunks for padding.
    private static let preSpeechPreallocationSamples: Int = VadManager.chunkSize * 3

    private let config: StreamingAsrConfig
    private let injectedManager: VadManager?

    private var manager: VadManager?
    private var streamState: VadStreamState?
    private var residualBuffer: CircularBuffer<Float>
    private var preSpeechBuffer: CircularBuffer<Float>
    private var overflowBuffer: [Float] = []
    private var speechActive: Bool = false
    private var wasAutoInitialized: Bool = false
    private var totalDroppedSamples: Int = 0
    private let overflowSampleHardLimit: Int

    init(config: StreamingAsrConfig, injectedManager: VadManager?) {
        self.config = config
        self.injectedManager = injectedManager
        self.manager = injectedManager
        self.residualBuffer = CircularBuffer(initialCapacity: Self.residualPreallocationSamples)
        self.preSpeechBuffer = CircularBuffer(initialCapacity: Self.preSpeechPreallocationSamples)
        self.overflowSampleHardLimit = Int(Self.overflowHardLimitSeconds * Double(VadManager.sampleRate))
    }

    var isEnabled: Bool {
        config.vad.isEnabled
    }

    func prepare(for source: AudioSource, logger: AppLogger) async throws {
        guard config.vad.isEnabled else {
            if injectedManager == nil {
                manager = nil
                wasAutoInitialized = false
            }
            streamState = nil
            residualBuffer.removeAll(keepingCapacity: false)
            preSpeechBuffer.removeAll(keepingCapacity: false)
            speechActive = false
            totalDroppedSamples = 0
            return
        }

        if let provided = injectedManager {
            manager = provided
            wasAutoInitialized = false
            logger.debug("Using injected VadManager instance for streaming source \(String(describing: source))")
        } else if manager == nil || !wasAutoInitialized {
            logger.debug("Auto-initializing VadManager for streaming source \(String(describing: source))")
            let newManager = try await VadManager(config: config.vad.vadConfig)
            manager = newManager
            wasAutoInitialized = true
        } else {
            logger.debug("Reusing auto-initialized VadManager for streaming source \(String(describing: source))")
        }

        guard let activeManager = manager else {
            logger.error("VAD is enabled but no manager is available; continuing without VAD gating")
            streamState = nil
            return
        }

        streamState = await activeManager.makeStreamState()
        residualBuffer.removeAll(keepingCapacity: true)
        preSpeechBuffer.removeAll(keepingCapacity: true)
        overflowBuffer.removeAll(keepingCapacity: true)
        speechActive = false
        totalDroppedSamples = 0
    }

    func process(
        samples: [Float],
        logger: AppLogger,
        segmentHandler: @Sendable (StreamingVadEvent) async -> Void
    ) async {
        guard config.vad.isEnabled,
            let activeManager = manager,
            var state = streamState
        else {
            guard !samples.isEmpty else { return }
            let isOnset = !speechActive
            await segmentHandler(
                .speech(
                    StreamingVadSegment(
                        samples: samples,
                        cumulativeDroppedSamples: totalDroppedSamples,
                        isSpeechOnset: isOnset
                    )
                )
            )
            speechActive = true
            return
        }

        residualBuffer.append(contentsOf: samples)
        let chunkSize = VadManager.chunkSize
        let segmentation = config.vad.segmentationConfig

        while residualBuffer.count >= chunkSize {
            let chunk = residualBuffer.popFirst(chunkSize)

            preSpeechBuffer.append(contentsOf: chunk)
            if preSpeechBuffer.count > preSpeechSampleLimit {
                let overflow = preSpeechBuffer.count - preSpeechSampleLimit
                if overflow > 0 {
                    overflowBuffer.append(contentsOf: preSpeechBuffer.copyPrefix(overflow))
                    preSpeechBuffer.dropFirst(overflow)
                    await enforceOverflowLimit(handler: segmentHandler)
                }
            }

            do {
                let result = try await activeManager.processStreamingChunk(
                    chunk,
                    state: state,
                    config: segmentation,
                    returnSeconds: false
                )
                state = result.state

                if let event = result.event, event.kind == .speechStart {
                    let bufferCountBefore = preSpeechBuffer.count
                    let chunkEndSample = result.state.processedSamples
                    let samplesSinceStart = chunkEndSample - event.sampleIndex
                    let emitCount = min(max(samplesSinceStart, 0), bufferCountBefore)
                    var recoveredSamples: [Float] = []
                    if !overflowBuffer.isEmpty {
                        recoveredSamples.append(contentsOf: overflowBuffer)
                        overflowBuffer.removeAll(keepingCapacity: true)
                    }
                    if emitCount > 0 {
                        let startIndex = bufferCountBefore - emitCount
                        let droppedPrefix = startIndex
                        if droppedPrefix > 0 {
                            await emitSilence(
                                count: droppedPrefix,
                                handler: segmentHandler
                            )
                        }
                        let recovered = preSpeechBuffer.copyRange(startIndex..<bufferCountBefore)
                        recoveredSamples.append(contentsOf: recovered)
                        let segment = recoveredSamples
                        await emitSpeechSegment(segment, isSpeechOnset: true, handler: segmentHandler)
                    } else if !recoveredSamples.isEmpty {
                        await emitSpeechSegment(recoveredSamples, isSpeechOnset: true, handler: segmentHandler)
                        if bufferCountBefore > 0 {
                            await emitSilence(
                                count: bufferCountBefore,
                                handler: segmentHandler
                            )
                        }
                    } else if bufferCountBefore > 0 {
                        await emitSilence(
                            count: bufferCountBefore,
                            handler: segmentHandler
                        )
                    }
                    speechActive = true
                    preSpeechBuffer.removeAll(keepingCapacity: true)
                } else if speechActive {
                    if preSpeechBuffer.count > 0 {
                        let segment = preSpeechBuffer.asArray()
                        preSpeechBuffer.removeAll(keepingCapacity: true)
                        await emitSpeechSegment(segment, isSpeechOnset: false, handler: segmentHandler)
                    }
                }

                if let event = result.event, event.kind == .speechEnd {
                    speechActive = false
                    let padSamples = Int(segmentation.speechPadding * Double(VadManager.sampleRate))
                    if padSamples > 0 {
                        let tailCount = min(padSamples, chunk.count)
                        if tailCount > 0 {
                            let tail = Array(chunk.suffix(tailCount))
                            preSpeechBuffer.removeAll(keepingCapacity: true)
                            preSpeechBuffer.append(contentsOf: tail)
                        }
                    }
                }
            } catch {
                logger.error(
                    "VAD streaming chunk failed: \(error.localizedDescription). Falling back to ungated streaming.")
                var fallbackSamples = preSpeechBuffer.asArray()
                fallbackSamples.append(contentsOf: residualBuffer.asArray())

                if !overflowBuffer.isEmpty {
                    fallbackSamples.insert(contentsOf: overflowBuffer, at: 0)
                }

                speechActive = false
                preSpeechBuffer.removeAll(keepingCapacity: false)
                overflowBuffer.removeAll(keepingCapacity: false)
                residualBuffer.removeAll(keepingCapacity: false)
                streamState = nil
                await emitSpeechSegment(fallbackSamples, isSpeechOnset: true, handler: segmentHandler)
                return
            }
        }

        streamState = state
    }

    func flushPending(
        logger: AppLogger,
        segmentHandler: @Sendable (StreamingVadEvent) async -> Void
    ) async {
        guard config.vad.isEnabled else { return }

        var pending: [Float] = []
        if !overflowBuffer.isEmpty {
            pending.append(contentsOf: overflowBuffer)
        }
        if preSpeechBuffer.count > 0 {
            pending.append(contentsOf: preSpeechBuffer.asArray())
        }
        if residualBuffer.count > 0 {
            pending.append(contentsOf: residualBuffer.asArray())
        }

        if !pending.isEmpty {
            await emitSpeechSegment(pending, isSpeechOnset: false, handler: segmentHandler)
        }

        preSpeechBuffer.removeAll(keepingCapacity: false)
        overflowBuffer.removeAll(keepingCapacity: false)
        residualBuffer.removeAll(keepingCapacity: false)
        speechActive = false
    }

    func resetState() async {
        residualBuffer.removeAll(keepingCapacity: false)
        preSpeechBuffer.removeAll(keepingCapacity: false)
        overflowBuffer.removeAll(keepingCapacity: false)
        speechActive = false
        totalDroppedSamples = 0

        if config.vad.isEnabled, let activeManager = manager {
            streamState = await activeManager.makeStreamState()
        } else {
            streamState = nil
        }
    }

    func activeManager() -> VadManager? {
        manager
    }

    private var preSpeechSampleLimit: Int {
        let paddingSamples = Int(config.vad.segmentationConfig.speechPadding * Double(VadManager.sampleRate))
        let leftContext = config.leftContextSamples
        let baseline = max(paddingSamples, leftContext)
        return baseline + VadManager.chunkSize
    }

    private func emitSpeechSegment(
        _ segment: [Float],
        isSpeechOnset: Bool,
        handler: @Sendable (StreamingVadEvent) async -> Void
    ) async {
        guard !segment.isEmpty else { return }
        await handler(
            .speech(
                StreamingVadSegment(
                    samples: segment,
                    cumulativeDroppedSamples: totalDroppedSamples,
                    isSpeechOnset: isSpeechOnset
                )
            )
        )
    }

    private func emitSilence(
        count: Int,
        handler: @Sendable (StreamingVadEvent) async -> Void
    ) async {
        guard count > 0 else { return }
        totalDroppedSamples += count
        await handler(
            .silence(samples: count, cumulativeDroppedSamples: totalDroppedSamples)
        )
    }

    private func enforceOverflowLimit(
        handler: @Sendable (StreamingVadEvent) async -> Void
    ) async {
        let excess = overflowBuffer.count - overflowSampleHardLimit
        guard excess > 0 else { return }
        let overflowSegment = overflowBuffer
        overflowBuffer.removeAll(keepingCapacity: true)
        if !overflowSegment.isEmpty {
            speechActive = true
            await emitSpeechSegment(overflowSegment, isSpeechOnset: true, handler: handler)
        }
    }
}
