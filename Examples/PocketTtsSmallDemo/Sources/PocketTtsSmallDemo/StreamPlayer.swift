import AVFoundation
import Foundation

/// Plays 24 kHz mono frames the moment they arrive: each 80 ms frame is
/// scheduled on an `AVAudioPlayerNode` as its own buffer, so audio starts on
/// the first frame and the generator only has to stay ahead of playback.
/// Playback level, fed from the player node's audio tap (audio thread) in
/// 20 ms blocks and read by `OceanWaveView` at display rate. Lock-protected,
/// deliberately not `@Published`.
final class LevelMeter: @unchecked Sendable {
    static let blockSamples = 480  // 20 ms at 24 kHz
    private let lock = NSLock()
    private var level: Float = 0
    private var history = [Float](repeating: 0, count: 160)  // 3.2 s
    private var head = 0
    private var lastPush = Date.distantPast

    func push(rms: Float) {
        lock.lock()
        defer { lock.unlock() }
        level = max(rms, level * 0.7 + rms * 0.3)  // fast attack, slow release
        history[head] = rms
        head = (head + 1) % history.count
        lastPush = Date()
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        level = 0
        history = [Float](repeating: 0, count: history.count)
        head = 0
    }

    /// Current (age-decayed) level and the history oldest -> newest. While no
    /// audio plays the tap is silent, so the history is padded with zeros for
    /// the elapsed time to keep the sea scrolling.
    func snapshot(at now: Date) -> (level: Float, history: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        let age = now.timeIntervalSince(lastPush)
        let gap = min(history.count, Int(max(0, age - 0.05) / 0.02))
        var out = [Float]()
        out.reserveCapacity(history.count)
        for i in gap..<history.count {
            out.append(history[(head + i) % history.count])
        }
        out.append(contentsOf: [Float](repeating: 0, count: gap))
        let decayed = age > 0.1 ? level * Float(max(0, 1 - (age - 0.1) * 4)) : level
        return (decayed, out)
    }
}

@MainActor
final class StreamPlayer: ObservableObject {
    /// Level of the audio being played, for the wave view.
    let meter = LevelMeter()
    static let sampleRate: Double = 24_000

    /// Frames handed to the player so far (this utterance).
    @Published private(set) var scheduledFrames = 0
    /// Frames the hardware has finished playing (this utterance).
    @Published private(set) var playedFrames = 0

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: StreamPlayer.sampleRate, channels: 1)!
    private var generation = 0

    init() {
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        node.installTap(
            onBus: 0, bufferSize: AVAudioFrameCount(LevelMeter.blockSamples), format: format,
            block: Self.makeTap(meter: meter))
    }

    /// The tap runs on an audio-render helper thread: built outside the
    /// `@MainActor` context so the closure carries no actor isolation (an
    /// isolated closure here traps in `dispatch_assert_queue` on first call).
    nonisolated private static func makeTap(meter: LevelMeter) -> AVAudioNodeTapBlock {
        { buffer, _ in
            guard let data = buffer.floatChannelData?[0] else { return }
            let n = Int(buffer.frameLength)
            var start = 0
            while start < n {
                let end = min(n, start + LevelMeter.blockSamples)
                var sum: Float = 0
                for i in start..<end { sum += data[i] * data[i] }
                meter.push(rms: (sum / Float(end - start)).squareRoot())
                start = end
            }
        }
    }

    func begin() throws {
        generation += 1
        scheduledFrames = 0
        playedFrames = 0
        meter.reset()
        if !engine.isRunning {
            try engine.start()
        }
        if !node.isPlaying {
            node.play()
        }
    }

    /// Queue one generated frame. Returns immediately; playback is continuous
    /// as long as frames arrive faster than 80 ms each.
    func enqueue(_ samples: [Float]) {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        let current = generation
        node.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.generation == current else { return }
                self.playedFrames += 1
            }
        }
        scheduledFrames += 1
    }

    func stop() {
        generation += 1
        node.stop()
        node.play()
        scheduledFrames = 0
        playedFrames = 0
        meter.reset()
    }
}
