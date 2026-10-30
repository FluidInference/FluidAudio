import AVFoundation
import Foundation

/// What the speaker is playing right now, for the ring: an overall level and a
/// coarse spectrum (`bandCount` log-spaced bands, 90 Hz to 9 kHz), both from the
/// player's audio tap in 20 ms blocks. Lock-protected and read at display rate,
/// deliberately not `@Published`, so the rest of the UI never re-renders per block.
final class SpectrumMeter: @unchecked Sendable {
    static let blockSamples = 480  // 20 ms at 24 kHz
    static let bandCount = 24

    /// Goertzel coefficients for the band centres, computed once.
    private static let coefficients: [Float] = (0..<bandCount).map { i in
        let lo = log(90.0)
        let hi = log(9_000.0)
        let f = exp(lo + (hi - lo) * Double(i) / Double(bandCount - 1))
        return Float(2 * cos(2 * Double.pi * f / 24_000))
    }

    private let lock = NSLock()
    private var level: Float = 0
    private var bands = [Float](repeating: 0, count: bandCount)
    private var lastPush = Date.distantPast

    /// Called on the audio thread with one 20 ms block.
    func push(_ block: UnsafePointer<Float>, count n: Int) {
        var sum: Float = 0
        for i in 0..<n { sum += block[i] * block[i] }
        let rms = (sum / Float(max(n, 1))).squareRoot()
        var mags = [Float](repeating: 0, count: Self.bandCount)
        for (b, c) in Self.coefficients.enumerated() {
            var s1: Float = 0
            var s2: Float = 0
            for i in 0..<n {
                let s0 = block[i] + c * s1 - s2
                s2 = s1
                s1 = s0
            }
            let power = max(0, s1 * s1 + s2 * s2 - c * s1 * s2)
            // dB above the noise floor, squashed to 0...1; higher bands get a tilt
            // boost so sibilants show as well as vowels.
            let db = 10 * log10(power / Float(n * n) + 1e-12) + 6 * Float(b) / Float(Self.bandCount)
            mags[b] = min(1, max(0, (db + 70) / 55))
        }
        lock.lock()
        defer { lock.unlock() }
        level = max(rms, level * 0.75 + rms * 0.25)  // fast attack, slow release
        for b in 0..<Self.bandCount {
            bands[b] = mags[b] > bands[b] ? mags[b] * 0.7 + bands[b] * 0.3 : mags[b] * 0.25 + bands[b] * 0.75
        }
        lastPush = Date()
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        level = 0
        bands = [Float](repeating: 0, count: Self.bandCount)
    }

    /// Level and bands, decayed to zero once the tap goes quiet (no audio queued).
    func snapshot(at now: Date) -> (level: Float, bands: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        let age = now.timeIntervalSince(lastPush)
        let k = age > 0.06 ? Float(max(0, 1 - (age - 0.06) * 3)) : 1
        return (level * k, bands.map { $0 * k })
    }
}

/// Plays 24 kHz mono frames the moment they arrive: each 80 ms frame is scheduled
/// on an `AVAudioPlayerNode` as its own buffer, so audio starts on the first frame
/// and the generator only has to stay ahead of playback.
@MainActor
final class StreamPlayer: ObservableObject {
    let meter = SpectrumMeter()
    static let sampleRate: Double = 24_000

    @Published private(set) var scheduledFrames = 0
    @Published private(set) var playedFrames = 0

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: StreamPlayer.sampleRate, channels: 1)!
    private var generation = 0

    init() {
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        node.installTap(
            onBus: 0, bufferSize: AVAudioFrameCount(SpectrumMeter.blockSamples), format: format,
            block: Self.makeTap(meter: meter))
    }

    /// The tap runs on an audio-render thread: built outside the `@MainActor`
    /// context so the closure carries no actor isolation (an isolated closure here
    /// traps in `dispatch_assert_queue` on first call).
    nonisolated private static func makeTap(meter: SpectrumMeter) -> AVAudioNodeTapBlock {
        { buffer, _ in
            guard let data = buffer.floatChannelData?[0] else { return }
            let n = Int(buffer.frameLength)
            var start = 0
            while start < n {
                let count = min(SpectrumMeter.blockSamples, n - start)
                meter.push(data.advanced(by: start), count: count)
                start += count
            }
        }
    }

    var isPlaying: Bool { playedFrames < scheduledFrames }

    /// Silence the speaker but keep playing (and metering) the audio, so the ring
    /// still moves: for recordings and screenshots.
    func setMuted(_ muted: Bool) { engine.mainMixerNode.outputVolume = muted ? 0 : 1 }

    func begin() throws {
        generation += 1
        scheduledFrames = 0
        playedFrames = 0
        meter.reset()
        if !engine.isRunning { try engine.start() }
        if !node.isPlaying { node.play() }
    }

    func enqueue(_ samples: [Float]) {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
            let channel = buffer.floatChannelData?[0]
        else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            if let base = src.baseAddress { channel.update(from: base, count: samples.count) }
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
