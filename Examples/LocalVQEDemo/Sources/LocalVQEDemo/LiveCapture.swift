import AVFoundation
import FluidAudio
import Foundation
import os

/// One 256 ms step of paired audio at the taps' native rates.
struct LivePair: Sendable {
    let mic: [Float]
    let micRate: Double
    let reference: [Float]
    let referenceRate: Double
}

/// Lock-protected pairing of the two audio taps. The input-node tap appends
/// mic samples and the main-mixer tap appends the rendered far-end signal;
/// whenever both hold at least one step, a `LivePair` is emitted. Both taps
/// start with the engine, so pairing by elapsed time aligns them to within
/// the device round-trip latency, which the model's own delay search absorbs.
final class PairingBox: Sendable {
    private struct State {
        var mic: [Float] = []
        var reference: [Float] = []
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let micRate: Double
    private let referenceRate: Double
    private let stepSeconds: Double
    private let continuation: AsyncStream<LivePair>.Continuation

    init(micRate: Double, referenceRate: Double, stepSeconds: Double, continuation: AsyncStream<LivePair>.Continuation)
    {
        self.micRate = micRate
        self.referenceRate = referenceRate
        self.stepSeconds = stepSeconds
        self.continuation = continuation
    }

    func appendMic(_ samples: [Float]) {
        state.withLock { $0.mic.append(contentsOf: samples) }
        emitReadyPairs()
    }

    func appendReference(_ samples: [Float]) {
        state.withLock { $0.reference.append(contentsOf: samples) }
        emitReadyPairs()
    }

    func finish() {
        continuation.finish()
    }

    private func emitReadyPairs() {
        let micStep = Int(micRate * stepSeconds)
        let referenceStep = Int(referenceRate * stepSeconds)
        while true {
            let pair: LivePair? = state.withLock { s in
                guard s.mic.count >= micStep, s.reference.count >= referenceStep else { return nil }
                let pair = LivePair(
                    mic: Array(s.mic[..<micStep]), micRate: micRate,
                    reference: Array(s.reference[..<referenceStep]), referenceRate: referenceRate)
                s.mic.removeFirst(micStep)
                s.reference.removeFirst(referenceStep)
                return pair
            }
            guard let pair else { return }
            continuation.yield(pair)
        }
    }
}

/// Levels published to the UI once per processed step.
struct LiveLevels: Sendable {
    var micDb: Float = -80
    var referenceDb: Float = -80
    var enhancedDb: Float = -80
    var seconds: Double = 0
    var stepMilliseconds: Double = 0
}

/// Captured 16 kHz clips after a live session ends.
struct LiveRecording: Sendable {
    var mic: [Float] = []
    var reference: [Float] = []
    var enhanced: [Float] = []
}

/// Live loop: plays a far-end file through the speakers, captures the mic,
/// and streams both through `LocalVqeStream`.
@MainActor
final class LiveCapture {
    private let engine = AVAudioEngine()
    private let farEndPlayer = AVAudioPlayerNode()
    private var box: PairingBox?
    private var feeder: Task<LiveRecording, Error>?

    static func requestMicrophoneAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    /// Starts playback + capture. `onLevels` is called on the main actor per step.
    func start(
        farEnd: AVAudioFile, stream: LocalVqeStream, stepSeconds: Double,
        onLevels: @escaping @MainActor (LiveLevels) -> Void
    ) throws {
        engine.attach(farEndPlayer)
        engine.connect(farEndPlayer, to: engine.mainMixerNode, format: farEnd.processingFormat)

        let input = engine.inputNode
        let micFormat = input.outputFormat(forBus: 0)
        let mixFormat = engine.mainMixerNode.outputFormat(forBus: 0)
        guard micFormat.sampleRate > 0, mixFormat.sampleRate > 0 else {
            throw DemoError.message("No audio input/output device available")
        }

        let (pairs, continuation) = AsyncStream<LivePair>.makeStream()
        let box = PairingBox(
            micRate: micFormat.sampleRate, referenceRate: mixFormat.sampleRate,
            stepSeconds: stepSeconds, continuation: continuation)
        self.box = box

        input.installTap(onBus: 0, bufferSize: 2048, format: micFormat) { buffer, _ in
            box.appendMic(Self.channelZero(buffer))
        }
        engine.mainMixerNode.installTap(onBus: 0, bufferSize: 2048, format: mixFormat) { buffer, _ in
            box.appendReference(Self.channelZero(buffer))
        }

        feeder = Task.detached(priority: .userInitiated) {
            var recording = LiveRecording()
            let micConverter = AudioConverter()
            let referenceConverter = AudioConverter()
            var seconds = 0.0
            for await pair in pairs {
                var mic = try micConverter.resample(pair.mic, from: pair.micRate)
                var reference = try referenceConverter.resample(pair.reference, from: pair.referenceRate)
                let n = min(mic.count, reference.count)
                mic.removeLast(mic.count - n)
                reference.removeLast(reference.count - n)

                let started = ContinuousClock.now
                let enhanced = try await stream.enhance(mic: mic, reference: reference)
                let elapsed = started.duration(to: .now)
                seconds += Double(n) / AudioFiles.sampleRate

                recording.mic.append(contentsOf: mic)
                recording.reference.append(contentsOf: reference)
                recording.enhanced.append(contentsOf: enhanced)

                let levels = LiveLevels(
                    micDb: AudioFiles.dbfs(AudioFiles.rms(mic[...])),
                    referenceDb: AudioFiles.dbfs(AudioFiles.rms(reference[...])),
                    enhancedDb: AudioFiles.dbfs(AudioFiles.rms(enhanced[...])),
                    seconds: seconds,
                    stepMilliseconds: Double(elapsed.components.attoseconds) / 1e15
                        + Double(elapsed.components.seconds) * 1000)
                await onLevels(levels)
            }
            // Drain the delay line so enhanced length == mic length.
            recording.enhanced.append(contentsOf: try await stream.flush())
            return recording
        }

        engine.prepare()
        try engine.start()
        farEndPlayer.scheduleFile(farEnd, at: nil)
        farEndPlayer.play()
    }

    /// Stops the engine and returns the captured 16 kHz clips.
    func stop() async throws -> LiveRecording {
        farEndPlayer.stop()
        engine.inputNode.removeTap(onBus: 0)
        engine.mainMixerNode.removeTap(onBus: 0)
        engine.stop()
        engine.detach(farEndPlayer)
        box?.finish()
        box = nil
        defer { feeder = nil }
        guard let feeder else { return LiveRecording() }
        return try await feeder.value
    }

    private nonisolated static func channelZero(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let data = buffer.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(buffer.frameLength)))
    }
}
