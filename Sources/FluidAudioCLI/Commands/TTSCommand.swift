#if os(macOS)
import FluidAudio
import Foundation

@available(macOS 13.0, *)
enum TTSCommand {

    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            print("No text specified")
            printUsage()
            exit(1)
        }

        let text = arguments[0]
        var outputPath = "output.wav"
        var voiceSpeed: Float = 1.0
        var speakerId = 0
        var autoDownload = false

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--output", "-o":
                i += 1
                guard i < arguments.count else {
                    print("Missing value for --output")
                    exit(1)
                }
                outputPath = arguments[i]
            case "--speed", "-s":
                i += 1
                guard i < arguments.count else {
                    print("Missing value for --speed")
                    exit(1)
                }
                guard let speed = Float(arguments[i]) else {
                    print("Invalid speed value: \(arguments[i])")
                    exit(1)
                }
                voiceSpeed = speed
            case "--speaker-id":
                i += 1
                guard i < arguments.count else {
                    print("Missing value for --speaker-id")
                    exit(1)
                }
                guard let id = Int(arguments[i]) else {
                    print("Invalid speaker ID: \(arguments[i])")
                    exit(1)
                }
                speakerId = id
            case "--auto-download":
                autoDownload = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                print("Unknown option: \(arguments[i])")
                printUsage()
                exit(1)
            }
            i += 1
        }

        print("🎤 Text-to-Speech Synthesis")
        print("===========================\n")
        print("📝 Text: \"\(text)\"")
        print("🎚️  Speed: \(voiceSpeed)")
        print("🗣️  Speaker ID: \(speakerId)")
        print("📁 Output: \(outputPath)\n")

        do {
            print("Loading TTS model...")

            let manager = TTSManager()

            if autoDownload {
                print("Downloading Kokoro TTS model...")
                try await manager.initialize()
            } else {
                let models = try await TTSModels.download { progress in
                    let percentage = Int(progress * 100)
                    print("\rDownloading: \(percentage)%", terminator: "")
                    fflush(stdout)
                }
                print("")
                try await manager.initialize(models: models)
            }

            let outputURL = URL(fileURLWithPath: outputPath)

            print("\nSynthesizing speech...")
            let startTime = Date()

            try await manager.synthesizeToFile(
                text: text,
                outputURL: outputURL,
                voiceSpeed: voiceSpeed,
                speakerId: speakerId
            )

            let processingTime = Date().timeIntervalSince(startTime)

            print("\n✅ Success!")
            print("📊 Processing time: \(String(format: "%.2f", processingTime)) seconds")

            if FileManager.default.fileExists(atPath: outputURL.path) {
                let attributes = try FileManager.default.attributesOfItem(atPath: outputURL.path)
                if let fileSize = attributes[.size] as? Int {
                    let sizeInKB = Double(fileSize) / 1024.0
                    print("💾 File size: \(String(format: "%.1f", sizeInKB)) KB")
                }
            }

        } catch {
            print("\n❌ Error: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func printUsage() {
        print(
            """
            FluidAudio TTS Command

            Usage: fluidaudio tts <text> [options]

            Options:
                --output, -o <path>     Output audio file path (default: output.wav)
                --speed, -s <float>     Voice speed (0.5-2.0, default: 1.0)
                --speaker-id <int>      Speaker ID for voice selection (default: 0)
                --auto-download         Auto-download model if not present
                --help, -h              Show this help message

            Examples:
                fluidaudio tts "Hello world" --output hello.wav
                fluidaudio tts "Speak slower please" --speed 0.8
                fluidaudio tts "Different voice" --speaker-id 1 --output voice1.wav
            """
        )
    }
}
#endif
