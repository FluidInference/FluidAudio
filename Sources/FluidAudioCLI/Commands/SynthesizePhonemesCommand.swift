import FluidAudio
import FluidAudioTTS
import Foundation

enum SynthesizePhonemesCommand {
    private static let logger = AppLogger(category: "SynthesizePhonemes")

    static func run(arguments: [String]) async {
        guard arguments.count >= 1 else {
            print("Usage: fluidaudio synthesize-phonemes \"phoneme_string\" [--output out.wav] [--voice zf_xiaobei]")
            return
        }

        let phonemes = arguments[0]
        var output = "out_phonemes.wav"
        var voice = "zf_xiaobei"
        
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--output", "-o":
                if i + 1 < arguments.count {
                    output = arguments[i + 1]
                    i += 1
                }
            case "--voice", "-v":
                if i + 1 < arguments.count {
                    voice = arguments[i + 1]
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        do {
            let manager = TtSManager()
            let preloadVoices: Set<String> = [voice]
            try await manager.initialize(preloadVoices: preloadVoices)

            print("Synthesizing phonemes: \(phonemes)")
            
            // Use the detailed method which accepts raw phoneme strings
            // We pass it as a single chunk
            let result = try await manager.synthesizePhonemeStringsDetailed(
                phonemes: [phonemes],
                voice: voice,
                voiceSpeed: 1.0,
                variantPreference: .fifteenSecond
            )

            let outURL = URL(fileURLWithPath: output)
            try result.audio.write(to: outURL)
            print("Saved to \(output)")
            
        } catch {
            print("Error: \(error)")
        }
    }
}
