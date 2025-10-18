import AVFoundation
import FluidAudio
import Foundation

actor StreamingTranscriptionCoordinator {

    private let config: StreamingAsrConfig
    private var streamingManager: StreamingAsrManager?
    private var updateTask: Task<Void, Never>?
    private let audioConverter = AudioConverter()
    private let chunkDuration: TimeInterval = 0.5
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,
        channels: 1,
        interleaved: false
    )!
    private var firstTokenLatency: TimeInterval?
    private var firstConfirmedTokenLatency: TimeInterval?
    private var sessionStartTime: Date?
    private var ingestedAudioSeconds: TimeInterval = 0

    init(config: StreamingAsrConfig = .streaming) {
        self.config = config
    }

    func startStreamingSession(
        models: AsrModels,
        source: AudioSource,
        onUpdate: @MainActor @escaping (StreamingTranscriptionUpdate) -> Void
    ) async throws {
        await cancelActiveSession()

        let manager = StreamingAsrManager(config: config)
        streamingManager = manager

        do {
            try await manager.start(models: models, source: source)
        } catch {
            streamingManager = nil
            throw error
        }

        let startTime = Date()
        sessionStartTime = startTime
        ingestedAudioSeconds = 0
        firstTokenLatency = nil
        firstConfirmedTokenLatency = nil

        updateTask = Task { [weak self, startTime, manager] in
            let updates = await manager.transcriptionUpdates
            for await update in updates {
                guard let coordinator = self else { break }
                await coordinator.recordLatencies(for: update, startedAt: startTime)
                await onUpdate(update)
            }
        }
    }

    func streamAudioBuffer(_ buffer: AVAudioPCMBuffer) async throws {
        guard let manager = streamingManager else {
            throw StreamingExampleError.noActiveSession
        }

        accumulateDuration(from: buffer)
        await manager.streamAudio(buffer)
    }

    func finishStreamingSession(
        audioDurationOverride: TimeInterval? = nil
    ) async throws -> StreamingTranscriptionResult {
        guard let manager = streamingManager else {
            throw StreamingExampleError.noActiveSession
        }
        guard let startTime = sessionStartTime else {
            throw StreamingExampleError.noActiveSession
        }

        let transcript = try await manager.finish()
        let wallTime = Date().timeIntervalSince(startTime)
        let audioSeconds = audioDurationOverride ?? ingestedAudioSeconds
        let tokenLatency = firstTokenLatency
        let confirmedLatency = firstConfirmedTokenLatency

        await clearActiveSession()

        return StreamingTranscriptionResult(
            transcript: transcript,
            wallClockSeconds: wallTime,
            audioSeconds: audioSeconds,
            firstTokenLatency: tokenLatency,
            firstConfirmedTokenLatency: confirmedLatency
        )
    }

    func transcribeFile(
        at url: URL,
        models: AsrModels,
        onUpdate: @MainActor @escaping (StreamingTranscriptionUpdate) -> Void
    ) async throws -> StreamingTranscriptionResult {
        let samples = try audioConverter.resampleAudioFile(url)
        guard !samples.isEmpty else {
            throw StreamingExampleError.emptyFile
        }

        let audioDuration = Double(samples.count) / targetFormat.sampleRate

        try await startStreamingSession(
            models: models,
            source: .system,
            onUpdate: onUpdate
        )

        do {
            try await streamSamples(samples)
            try await Task.sleep(nanoseconds: 250_000_000)
            return try await finishStreamingSession(audioDurationOverride: audioDuration)
        } catch {
            await clearActiveSession()
            throw error
        }
    }

    func cancelActiveSession() async {
        await clearActiveSession()
    }

    private func recordLatencies(for update: StreamingTranscriptionUpdate, startedAt startTime: Date) {
        let normalized = update.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }

        let latency = update.timestamp.timeIntervalSince(startTime)
        guard latency.isFinite else { return }

        if firstTokenLatency == nil {
            firstTokenLatency = max(0, latency)
        }
        if update.isConfirmed, firstConfirmedTokenLatency == nil {
            firstConfirmedTokenLatency = max(0, latency)
        }
    }

    private func clearActiveSession() async {
        updateTask?.cancel()
        updateTask = nil
        firstTokenLatency = nil
        firstConfirmedTokenLatency = nil
        sessionStartTime = nil
        ingestedAudioSeconds = 0

        if let manager = streamingManager {
            await manager.cancel()
        }
        streamingManager = nil
    }

    private func streamSamples(_ samples: [Float]) async throws {
        guard let manager = streamingManager else {
            throw StreamingExampleError.noActiveSession
        }

        let samplesPerChunk = max(1, Int(chunkDuration * targetFormat.sampleRate))

        var position = 0
        while position < samples.count {
            try Task.checkCancellation()

            let remaining = samples.count - position
            let chunkSize = min(samplesPerChunk, remaining)

            guard
                let chunkBuffer = AVAudioPCMBuffer(
                    pcmFormat: targetFormat,
                    frameCapacity: AVAudioFrameCount(chunkSize)
                )
            else {
                throw StreamingExampleError.bufferAllocationFailed
            }

            chunkBuffer.frameLength = AVAudioFrameCount(chunkSize)

            guard let destination = chunkBuffer.floatChannelData?[0] else {
                throw StreamingExampleError.bufferAllocationFailed
            }
            if let pointer = samples[position..<(position + chunkSize)]
                .withContiguousStorageIfAvailable({ $0 })
            {
                destination.update(from: pointer.baseAddress!, count: chunkSize)
            } else {
                for index in 0..<chunkSize {
                    destination[index] = samples[position + index]
                }
            }

            await manager.streamAudio(chunkBuffer)
            accumulateDuration(from: chunkBuffer)

            position += chunkSize
            await Task.yield()
        }
    }

    private func accumulateDuration(from buffer: AVAudioPCMBuffer) {
        let sampleRate = buffer.format.sampleRate
        guard sampleRate > 0 else { return }

        let frames = Double(buffer.frameLength)
        guard frames > 0 else { return }

        let duration = frames / sampleRate
        guard duration.isFinite, duration > 0 else { return }

        ingestedAudioSeconds += duration
    }
}

enum StreamingExampleError: LocalizedError {
    case emptyFile
    case bufferAllocationFailed
    case noActiveSession

    var errorDescription: String? {
        switch self {
        case .emptyFile:
            return "The selected audio file is empty."
        case .bufferAllocationFailed:
            return "Unable to allocate audio buffers for streaming."
        case .noActiveSession:
            return "No active streaming session is currently running."
        }
    }
}

struct StreamingTranscriptionResult {
    let transcript: String
    let wallClockSeconds: TimeInterval
    let audioSeconds: TimeInterval
    let firstTokenLatency: TimeInterval?
    let firstConfirmedTokenLatency: TimeInterval?
}
