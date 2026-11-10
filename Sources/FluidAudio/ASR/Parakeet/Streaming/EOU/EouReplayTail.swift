import Foundation

/// Audio shifted out of the streaming buffer since the EOU debounce anchor.
///
/// The RNN-T loop stops at a raw `<EOU>` token and does not consume the rest of that chunk, so speech that follows
/// can decode nothing until the debounce confirms. `StreamingEouAsrManager.rearm(keepTranscript:)` replays this
/// audio through a fresh decoder, which is why the manager keeps it. The tail is bounded: once it holds more than
/// `maxSamples`, the oldest shifts are dropped and the replay starts later.
struct EouReplayTail: Equatable {
    /// Samples shifted out since the anchor chunk started, oldest first.
    private(set) var samples: [Float] = []
    /// Number of chunks those samples were shifted out by.
    var chunkCount: Int { shiftLengths.count }

    private var shiftLengths: [Int] = []
    private let maxSamples: Int

    init(maxSamples: Int) {
        self.maxSamples = maxSamples
    }

    /// Appends the samples one chunk shifted out of the buffer.
    mutating func record(_ shifted: ArraySlice<Float>) {
        samples.append(contentsOf: shifted)
        shiftLengths.append(shifted.count)
        while samples.count > maxSamples, let oldest = shiftLengths.first {
            samples.removeFirst(oldest)
            shiftLengths.removeFirst()
        }
    }

    /// Returns the recorded audio and its chunk count, leaving the tail empty.
    mutating func take() -> (samples: [Float], chunkCount: Int) {
        let taken = (samples: samples, chunkCount: chunkCount)
        clear()
        return taken
    }

    mutating func clear() {
        samples.removeAll()
        shiftLengths.removeAll()
    }
}
