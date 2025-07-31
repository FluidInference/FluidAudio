#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Command to debug transcription issues
@available(macOS 13.0, *)
enum DebugTranscribeCommand {
    
    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            print("❌ No audio file specified")
            print("Usage: fluidaudio debug-transcribe <audio_file>")
            return
        }
        
        let audioFile = arguments[0]
        
        do {
            // Load audio
            print("Loading audio file: \(audioFile)")
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                print("Failed to create buffer")
                return
            }
            
            try audioFileHandle.read(into: buffer)
            
            // Convert to mono 16kHz
            var audioSamples: [Float] = []
            let channelData = buffer.floatChannelData!
            let channelCount = Int(format.channelCount)
            let frameLength = Int(buffer.frameLength)
            
            for frame in 0..<frameLength {
                var sum: Float = 0
                for channel in 0..<channelCount {
                    sum += channelData[channel][frame]
                }
                audioSamples.append(sum / Float(channelCount))
            }
            
            // Resample to 16kHz if needed
            if format.sampleRate != 16000 {
                let ratio = Float(16000) / Float(format.sampleRate)
                let targetLength = Int(Float(audioSamples.count) * ratio)
                var resampled = [Float](repeating: 0, count: targetLength)
                
                for i in 0..<targetLength {
                    let sourceIndex = Float(i) / ratio
                    let index = Int(sourceIndex)
                    let fraction = sourceIndex - Float(index)
                    
                    if index < audioSamples.count - 1 {
                        resampled[i] = audioSamples[index] * (1 - fraction) + audioSamples[index + 1] * fraction
                    } else if index < audioSamples.count {
                        resampled[i] = audioSamples[index]
                    }
                }
                audioSamples = resampled
            }
            
            print("Audio duration: \(Float(audioSamples.count) / 16000.0) seconds")
            
            // Load models
            print("\nLoading ASR models...")
            let models = try await AsrModels.downloadAndLoad()
            
            // Test different configurations
            let chunkSizes: [(name: String, seconds: Float)] = [
                ("2.5s (streaming default)", 2.5),
                ("5.0s", 5.0),
                ("10.0s", 10.0),
                ("30.0s", 30.0)
            ]
            
            for (name, seconds) in chunkSizes {
                print("\n=== Testing with \(name) chunks ===")
                
                let asrManager = AsrManager()
                try await asrManager.initialize(models: models)
                
                let chunkSize = Int(seconds * 16000)
                var position = 0
                var chunkIndex = 0
                var allTexts: [String] = []
                var emptyChunks = 0
                
                while position < audioSamples.count {
                    let endPos = min(position + chunkSize, audioSamples.count)
                    let chunk = Array(audioSamples[position..<endPos])
                    let chunkDuration = Float(chunk.count) / 16000.0
                    
                    let result = try await asrManager.transcribe(chunk)
                    
                    if result.text.isEmpty {
                        emptyChunks += 1
                        print("  Chunk \(chunkIndex) [\(String(format: "%.1f", Float(position)/16000))-\(String(format: "%.1f", Float(endPos)/16000))s]: [EMPTY]")
                    } else {
                        allTexts.append(result.text)
                        let wordCount = result.text.split(separator: " ").count
                        print("  Chunk \(chunkIndex) [\(String(format: "%.1f", Float(position)/16000))-\(String(format: "%.1f", Float(endPos)/16000))s]: \(wordCount) words - '\(result.text)'")
                    }
                    
                    position = endPos
                    chunkIndex += 1
                }
                
                let fullText = allTexts.joined(separator: " ")
                let totalWords = fullText.split(separator: " ").count
                
                print("\nSummary:")
                print("  Total chunks: \(chunkIndex)")
                print("  Empty chunks: \(emptyChunks)")
                print("  Total words: \(totalWords)")
                print("  Coverage: \((chunkIndex - emptyChunks) * 100 / chunkIndex)%")
            }
            
            // Test with full audio
            print("\n=== Testing with full audio (no chunks) ===")
            let asrManager = AsrManager()
            try await asrManager.initialize(models: models)
            
            let fullResult = try await asrManager.transcribe(audioSamples)
            let fullWords = fullResult.text.split(separator: " ").count
            
            print("Full transcription: \(fullWords) words")
            print("First 200 chars: \(String(fullResult.text.prefix(200)))...")
            
        } catch {
            print("Error: \(error)")
        }
    }
}
#endif