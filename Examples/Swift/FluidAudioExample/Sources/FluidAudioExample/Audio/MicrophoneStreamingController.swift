import AVFAudio
import AVFoundation
import FluidAudio
import Foundation

@MainActor
final class MicrophoneStreamingController {

    private let coordinator: StreamingTranscriptionCoordinator
    private let engine = AVAudioEngine()
    private var isTapInstalled = false
    private var engineRunning = false
    private let tapBufferSize: AVAudioFrameCount = 4_096
    private let finalizationDelayNanoseconds: UInt64 = 250_000_000
    #if os(iOS)
    private var audioSessionActivated = false
    #endif

    init(coordinator: StreamingTranscriptionCoordinator) {
        self.coordinator = coordinator
        engine.mainMixerNode.outputVolume = 0
    }

    var isStreaming: Bool {
        engineRunning
    }

    func requestPermissionIfNeeded() async -> Bool {
        await ensureRecordPermission()
    }

    func startStreaming(
        models: AsrModels,
        onUpdate: @MainActor @escaping (StreamingTranscriptionUpdate) -> Void,
        skipPermissionCheck: Bool = false
    ) async throws {
        guard !engineRunning else { return }

        if !skipPermissionCheck {
            let permissionGranted = await ensureRecordPermission()
            guard permissionGranted else {
                throw MicrophoneStreamingError.permissionDenied
            }
        }

        try configureAudioSessionIfNeeded()

        let inputNode = engine.inputNode
        let hardwareFormat = inputNode.outputFormat(forBus: 0)

        guard
            let tapFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: hardwareFormat.sampleRate,
                channels: 1,
                interleaved: false
            )
        else {
            deactivateAudioSessionIfNeeded()
            throw MicrophoneStreamingError.inputConfigurationFailed
        }

        installTap(format: tapFormat)

        do {
            try await coordinator.startStreamingSession(
                models: models,
                source: .microphone,
                onUpdate: onUpdate
            )
        } catch {
            removeTapIfNeeded()
            deactivateAudioSessionIfNeeded()
            throw error
        }

        engine.prepare()

        do {
            try engine.start()
            engineRunning = true
        } catch {
            engineRunning = false
            removeTapIfNeeded()
            deactivateAudioSessionIfNeeded()
            await coordinator.cancelActiveSession()
            throw MicrophoneStreamingError.engineStartFailed(error)
        }
    }

    func stopStreaming() async throws -> StreamingTranscriptionResult {
        guard engineRunning else {
            throw MicrophoneStreamingError.notStreaming
        }

        teardownEngine()

        do {
            try await Task.sleep(nanoseconds: finalizationDelayNanoseconds)
        } catch is CancellationError {
            // Allow cancellations to propagate without treating as failure.
        }

        let result = try await coordinator.finishStreamingSession()
        deactivateAudioSessionIfNeeded()
        return result
    }

    func cancel() async {
        teardownEngine()
        deactivateAudioSessionIfNeeded()
        await coordinator.cancelActiveSession()
    }

    private func installTap(format: AVAudioFormat) {
        removeTapIfNeeded()

        let streamingCoordinator = coordinator

        engine.inputNode.installTap(onBus: 0, bufferSize: tapBufferSize, format: format) {
            [weak self] buffer, _ in
            guard buffer.frameLength > 0 else { return }

            guard let copy = self?.makeBufferCopy(buffer) else { return }

            Task(priority: .userInitiated) {
                do {
                    try await streamingCoordinator.streamAudioBuffer(copy)
                } catch is CancellationError {
                    // Expected if the session is shutting down.
                } catch StreamingExampleError.noActiveSession {
                    // Session already closed; ignore stray buffers.
                } catch {
                    await MainActor.run {
                        guard let controller = self else { return }
                        controller.handleStreamingError(error)
                    }
                }
            }
        }

        isTapInstalled = true
    }

    private func teardownEngine() {
        removeTapIfNeeded()
        if engine.isRunning {
            engine.stop()
        }
        engine.reset()
        engineRunning = false
    }

    private func removeTapIfNeeded() {
        if isTapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            isTapInstalled = false
        }
    }

    private func ensureRecordPermission() async -> Bool {
        if #available(macOS 14.0, iOS 17.0, *) {
            let application = AVAudioApplication.shared
            switch application.recordPermission {
            case .granted:
                return true
            case .denied:
                return false
            case .undetermined:
                return await AVAudioApplication.requestRecordPermission()
            @unknown default:
                return false
            }
        } else {
            #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            switch session.recordPermission {
            case .granted:
                return true
            case .denied:
                return false
            case .undetermined:
                return await withCheckedContinuation { continuation in
                    session.requestRecordPermission { granted in
                        continuation.resume(returning: granted)
                    }
                }
            @unknown default:
                return false
            }
            #else
            return false
            #endif
        }
    }

    nonisolated private func makeBufferCopy(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard
            let copy = AVAudioPCMBuffer(
                pcmFormat: buffer.format,
                frameCapacity: buffer.frameCapacity
            ),
            let source = buffer.floatChannelData,
            let destination = copy.floatChannelData
        else {
            return nil
        }

        copy.frameLength = buffer.frameLength

        let frameCount = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)

        for channel in 0..<channels {
            destination[channel].update(from: source[channel], count: frameCount)
        }

        return copy
    }

    private func handleStreamingError(_ error: Error) {
        NSLog("Microphone streaming error: \(error.localizedDescription)")
        teardownEngine()
        deactivateAudioSessionIfNeeded()
        Task {
            await coordinator.cancelActiveSession()
        }
    }

    private func configureAudioSessionIfNeeded() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        if !audioSessionActivated {
            try session.setCategory(
                .playAndRecord,
                mode: .measurement,
                options: [.allowBluetooth, .defaultToSpeaker]
            )
            try session.setActive(true, options: [])
            audioSessionActivated = true
        }
        #endif
    }

    private func deactivateAudioSessionIfNeeded() {
        #if os(iOS)
        guard audioSessionActivated else { return }
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setActive(false, options: [.notifyOthersOnDeactivation])
        } catch {
            NSLog("Failed to deactivate audio session: \(error.localizedDescription)")
        }
        audioSessionActivated = false
        #endif
    }
}

enum MicrophoneStreamingError: LocalizedError {
    case permissionDenied
    case inputConfigurationFailed
    case engineStartFailed(Error)
    case notStreaming

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Microphone access is required to start live transcription."
        case .inputConfigurationFailed:
            return "Unable to configure the microphone input format."
        case .engineStartFailed(let error):
            return "Failed to start microphone capture: \(error.localizedDescription)"
        case .notStreaming:
            return "Microphone streaming is not currently active."
        }
    }
}
