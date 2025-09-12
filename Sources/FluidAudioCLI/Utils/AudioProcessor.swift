#if os(macOS)
import AVFoundation
import Foundation
import FluidAudio

/// Audio loading and processing utilities
struct AudioProcessor {

    /// Load an audio file and convert to 16kHz mono Float32 samples using the shared library converter
    static func loadAudioFile(path: String) async throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        return try await loadAudioFileDirectly(url: url)
    }

    private static func loadAudioFileDirectly(url: URL) async throws -> [Float] {
        let converter = AudioConverter()
        return try converter.resampleAudioFile(url)
    }
}

#endif
