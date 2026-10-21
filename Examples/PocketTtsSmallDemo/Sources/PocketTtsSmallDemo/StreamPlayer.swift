import AVFoundation
import Foundation

/// Plays 24 kHz mono frames the moment they arrive: each 80 ms frame is
/// scheduled on an `AVAudioPlayerNode` as its own buffer, so audio starts on
/// the first frame and the generator only has to stay ahead of playback.
@MainActor
final class StreamPlayer: ObservableObject {
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
    }

    func begin() throws {
        generation += 1
        scheduledFrames = 0
        playedFrames = 0
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
    }
}
