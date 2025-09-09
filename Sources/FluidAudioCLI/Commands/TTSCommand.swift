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
        var autoDownload = false

        // Parse optional arguments
        for i in 0..<arguments.count {
            if arguments[i] == "--output" || arguments[i] == "-o", i + 1 < arguments.count {
                output = arguments[i + 1]
            }
            if arguments[i] == "--voice" || arguments[i] == "-v", i + 1 < arguments.count {
                voice = arguments[i + 1]
            }
            if arguments[i] == "--auto-download" {
                autoDownload = true
            }
        }

        print("Text-to-Speech Synthesis")
        print("===========================\n")
        print("Text: \"\(text)\"")
        print("Voice: \(voice)")
        print("Output: \(output)")
        print()

        do {
            var modelLoadTime: TimeInterval = 0

            // Download models if requested (not counted in processing time)
            if autoDownload {
                print("Downloading required models...")
                try await KokoroModel.ensureRequiredFiles()
                print("Models ready")
                print()
            }

            // Measure model loading time separately
            let modelLoadStart = Date()
            try await KokoroModel.loadModels()
            modelLoadTime = Date().timeIntervalSince(modelLoadStart)

            // Start timing only for actual synthesis (excluding model loading)
            let inferenceStart = Date()

            // Synthesize using Kokoro TTS (multi-model architecture)
            let audioData = try await KokoroModel.synthesize(text: text, voice: voice)

            // End inference timing
            let inferenceTime = Date().timeIntervalSince(inferenceStart)

            // Save to file (not counted in processing time)
            let outputURL = URL(fileURLWithPath: output)
            try audioData.write(to: outputURL)

            let elapsed = inferenceTime  // Use only inference time
            let fileSize = Double(audioData.count) / 1024.0

            // Calculate audio duration from WAV header
            // WAV format: 24000Hz, 16-bit mono (2 bytes per sample)
            // 24000 samples/sec * 2 bytes/sample = 48000 bytes per second
            // Skip 44-byte WAV header
            let audioDuration = Double(audioData.count - 44) / 48000.0
            let rtfx = audioDuration / elapsed

            print()
            print("Success!")
            print("Model loading time: \(String(format: "%.3f", modelLoadTime)) seconds")
            print("Inference time: \(String(format: "%.2f", elapsed)) seconds")
            print("Audio duration: \(String(format: "%.2f", audioDuration)) seconds")
            print("RTFx: \(String(format: "%.1f", rtfx))x (realtime factor)")
            print("File size: \(String(format: "%.1f", fileSize)) KB")

        } catch {
            print("Error: \(error.localizedDescription)")
        }
    }
}
