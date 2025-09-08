import FluidAudio
import Foundation

@available(macOS 13.0, *)
public struct TTS {

    public static func run(arguments: [String]) async {
        // Parse arguments
        guard arguments.count >= 1 else {
            print("Usage: fluidaudio tts \"text\" [--output file.wav] [--voice af_heart] [--auto-download]")
            return
        }

        let text = arguments[0]
        var output = "output.wav"
        var voice = "af_heart"
        var autoDownload = true  // Default to auto-download

        // Parse optional arguments
        for i in 0..<arguments.count {
            if arguments[i] == "--output" || arguments[i] == "-o", i + 1 < arguments.count {
                output = arguments[i + 1]
            }
            if arguments[i] == "--voice" || arguments[i] == "-v", i + 1 < arguments.count {
                voice = arguments[i + 1]
            }
            if arguments[i] == "--no-auto-download" {
                autoDownload = false
            }
        }

        print("🎤 Text-to-Speech Synthesis")
        print("===========================\n")
        print("📝 Text: \"\(text)\"")
        print("🗣️  Voice: \(voice)")
        print("📁 Output: \(output)")
        print()

        let startTime = Date()

        do {
            // Auto-download files if needed
            if autoDownload && !KokoroModelDownloader.checkRequiredFiles() {
                print("📥 Downloading required model and data files...")
                try await KokoroModelDownloader.downloadRequiredFilesIfNeeded()
            }

            // Download voice file if needed
            if autoDownload {
                try await KokoroModelDownloader.downloadVoiceIfNeeded(voice: voice)
            }

            // Synthesize using direct approach
            let audioData = try KokoroDirectTTS.synthesize(text: text, voice: voice)

            // Save to file
            let outputURL = URL(fileURLWithPath: output)
            try audioData.write(to: outputURL)

            let elapsed = Date().timeIntervalSince(startTime)
            let fileSize = Double(audioData.count) / 1024.0

            print()
            print("✅ Success!")
            print("📊 Processing time: \(String(format: "%.2f", elapsed)) seconds")
            print("💾 File size: \(String(format: "%.1f", fileSize)) KB")

        } catch {
            print("❌ Error: \(error.localizedDescription)")
        }
    }
}
