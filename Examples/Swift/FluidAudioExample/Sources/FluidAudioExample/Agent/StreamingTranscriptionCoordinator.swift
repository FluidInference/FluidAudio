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

    init(config: StreamingAsrConfig = .streaming) {
        self.config = config
    }

    func transcribeFile(
        at url: URL,
        models: AsrModels,
        onUpdate: @MainActor @escaping (StreamingTranscriptionUpdate) -> Void
    ) async throws -> StreamingTranscriptionResult {
        await cancelActiveSession()

        let manager = StreamingAsrManager(config: config)
        streamingManager = manager

        try await manager.start(models: models, source: .system)

        do {
            let samples = try audioConverter.resampleAudioFile(url)
            guard !samples.isEmpty else {
                throw StreamingExampleError.emptyFile
            }

            let audioDuration = Double(samples.count) / targetFormat.sampleRate
            let streamingStart = Date()
            firstTokenLatency = nil
            firstConfirmedTokenLatency = nil

            updateTask = Task { [weak self, streamingStart, manager] in
                let updates = await manager.transcriptionUpdates
                for await update in updates {
                    guard let coordinator = self else { break }
                    await coordinator.recordLatencies(for: update, startedAt: streamingStart)
                    await onUpdate(update)
                }
            }

            try await streamSamples(samples, into: manager)
            try await Task.sleep(nanoseconds: 250_000_000)
            let transcript = try await manager.finish()
            let wallTime = Date().timeIntervalSince(streamingStart)
            let latencySnapshot = (
                firstTokenLatency,
                firstConfirmedTokenLatency
            )

            await clearActiveSession()

            return StreamingTranscriptionResult(
                transcript: transcript,
                wallClockSeconds: wallTime,
                audioSeconds: audioDuration,
                firstTokenLatency: latencySnapshot.0,
                firstConfirmedTokenLatency: latencySnapshot.1
            )
        } catch {
            updateTask?.cancel()
            updateTask = nil
            if let manager = streamingManager {
                await manager.cancel()
            }
            streamingManager = nil
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

        if let manager = streamingManager {
            await manager.cancel()
        }
        streamingManager = nil
    }

    private func streamSamples(_ samples: [Float], into manager: StreamingAsrManager) async throws {
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

            position += chunkSize
            await Task.yield()
        }
    }
}

enum StreamingExampleError: LocalizedError {
    case emptyFile
    case bufferAllocationFailed

    var errorDescription: String? {
        switch self {
        case .emptyFile:
            return "The selected audio file is empty."
        case .bufferAllocationFailed:
            return "Unable to allocate audio buffers for streaming."
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
