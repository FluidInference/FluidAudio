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

        updateTask = Task {
            let updates = await manager.transcriptionUpdates
            for await update in updates {
                await MainActor.run {
                    onUpdate(update)
                }
            }
        }

        do {
            let samples = try audioConverter.resampleAudioFile(url)
            guard !samples.isEmpty else {
                throw StreamingExampleError.emptyFile
            }

            let audioDuration = Double(samples.count) / targetFormat.sampleRate
            let startTime = Date()

            try await streamSamples(samples, into: manager)
            try await Task.sleep(nanoseconds: 250_000_000)
            let transcript = try await manager.finish()
            let wallTime = Date().timeIntervalSince(startTime)
            let metrics = await manager.metricsSnapshot()

            await clearActiveSession()

            return StreamingTranscriptionResult(
                transcript: transcript,
                wallClockSeconds: wallTime,
                audioSeconds: audioDuration,
                metrics: metrics
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


    private func clearActiveSession() async {
        updateTask?.cancel()
        updateTask = nil

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
    let metrics: StreamingAsrEngineMetrics
}
