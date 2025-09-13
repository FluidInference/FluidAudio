import FluidAudio
import Foundation

@available(macOS 13.0, *)
public struct TTS {

    public static func run(arguments: [String]) async {
        // Usage: fluidaudio tts "text" [--output file.wav] [--voice af_heart] [--auto-download]
        guard arguments.count >= 1 else {
            print("Usage: fluidaudio tts \"text\" [--output file.wav] [--voice af_heart] [--auto-download]")
            return
        }

        let text = arguments[0]
        var output = "output.wav"
        var voice = "af_heart"
        var autoDownload = false

        for i in 0..<arguments.count {
            if (arguments[i] == "--output" || arguments[i] == "-o"), i + 1 < arguments.count { output = arguments[i + 1] }
            if (arguments[i] == "--voice" || arguments[i] == "-v"), i + 1 < arguments.count { voice = arguments[i + 1] }
            if arguments[i] == "--auto-download" { autoDownload = true }
        }

        do {
            if autoDownload {
                try await KokoroModel.ensureRequiredFiles()
            }
            // Delegate to the unified harness so tts and tts-harness are identical
            var harnessArgs: [String] = []
            harnessArgs.append(text)
            harnessArgs.append(contentsOf: ["--voice", voice, "--output", output])
            await TTSHarness.run(arguments: harnessArgs)
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    }
}
