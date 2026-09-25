import AVFoundation
import Foundation

/// Plays a 16 kHz mono clip through the default output; used for A/B listening.
@MainActor
final class Player: ObservableObject {
    @Published private(set) var playingLabel: String?

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private var generation = 0

    init() {
        engine.attach(node)
        let format = AVAudioFormat(standardFormatWithSampleRate: AudioFiles.sampleRate, channels: 1)
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }

    func play(_ samples: [Float], label: String) throws {
        try playSequence([(samples, label)])
    }

    /// Plays clips back to back with a short gap, updating `playingLabel` as each starts.
    func playSequence(_ clips: [(samples: [Float], label: String)]) throws {
        stop()
        let clips = clips.filter { !$0.samples.isEmpty }
        guard !clips.isEmpty else { return }
        if !engine.isRunning {
            try engine.start()
        }
        generation += 1
        let current = generation
        let gap = [Float](repeating: 0, count: Int(AudioFiles.sampleRate * 0.6))
        for (index, clip) in clips.enumerated() {
            let buffer = try AudioFiles.pcmBuffer(index == 0 ? clip.samples : gap + clip.samples)
            let next: String? = index + 1 < clips.count ? clips[index + 1].label : nil
            node.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) {
                [weak self] _ in
                Task { @MainActor in
                    guard let self, self.generation == current else { return }
                    self.playingLabel = next
                    print("Player: \(next.map { "now playing \($0)" } ?? "finished")")
                }
            }
        }
        node.play()
        playingLabel = clips[0].label
        print("Player: playing \(clips.map(\.label).joined(separator: " → "))")
    }

    func stop() {
        generation += 1
        node.stop()
        playingLabel = nil
    }
}
