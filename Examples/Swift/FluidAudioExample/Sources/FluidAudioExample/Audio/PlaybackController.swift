import AVFoundation
import Foundation

final class PlaybackController: NSObject {
    private var player: AVAudioPlayer?
    private var completionHandler: (() -> Void)?

    @MainActor
    func play(data: Data, completion: @escaping () -> Void) throws {
        stop()

        let audioPlayer = try AVAudioPlayer(data: data)
        audioPlayer.delegate = self
        audioPlayer.prepareToPlay()

        guard audioPlayer.play() else {
            throw PlaybackError.failedToStart
        }

        player = audioPlayer
        completionHandler = completion
    }

    @MainActor
    func stop() {
        player?.stop()
        player = nil
        completionHandler = nil
    }

    enum PlaybackError: LocalizedError {
        case failedToStart

        var errorDescription: String? {
            "Unable to play the synthesized audio."
        }
    }
}

extension PlaybackController: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.player = nil
            let handler = self.completionHandler
            self.completionHandler = nil
            handler?()
        }
    }
}
