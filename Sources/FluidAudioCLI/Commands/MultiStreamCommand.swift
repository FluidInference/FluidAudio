#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Command to demonstrate multi-stream ASR with shared model loading
@available(macOS 13.0, *)
enum MultiStreamCommand {
    
    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            print("❌ No audio file specified")
            printUsage()
            exit(1)
        }
        
        let audioFile = arguments[0]
        var showDebug = false
        
        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--debug":
                showDebug = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                print("⚠️  Unknown option: \(arguments[i])")
            }
            i += 1
        }
        
        print("🎤 Multi-Stream ASR Test")
        print("========================\n")
        
        do {
            // Load audio file
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                print("❌ Failed to create audio buffer")
                return
            }
            
            try audioFileHandle.read(into: buffer)
            
            print("📊 Audio file info:")
            print("  Sample rate: \(format.sampleRate) Hz")
            print("  Channels: \(format.channelCount)")
            print("  Duration: \(String(format: "%.2f", Double(audioFileHandle.length) / format.sampleRate)) seconds\n")
            
            // Create a streaming session
            print("🔄 Creating streaming session...")
            let session = StreamingAsrSession()
            
            // Initialize models once
            print("📦 Loading ASR models (shared across streams)...")
            let startTime = Date()
            try await session.initialize()
            let loadTime = Date().timeIntervalSince(startTime)
            print("✅ Models loaded in \(String(format: "%.2f", loadTime))s\n")
            
            // Create streams for different sources
            print("🎙️ Creating streams for different audio sources...")
            let micStream = try await session.createStream(
                source: .microphone,
                config: .lowLatency
            )
            print("✅ Created microphone stream")
            
            let systemStream = try await session.createStream(
                source: .system,
                config: .default
            )
            print("✅ Created system audio stream\n")
            
            // Set up update tracking
            var micUpdates: [String] = []
            var systemUpdates: [String] = []
            
            // Listen for updates from both streams
            let micTask = Task {
                for await update in await micStream.transcriptionUpdates {
                    if showDebug {
                        print("[MIC] \(update.isConfirmed ? "✓" : "~") \(update.text)")
                    }
                    micUpdates.append(update.text)
                }
            }
            
            let systemTask = Task {
                for await update in await systemStream.transcriptionUpdates {
                    if showDebug {
                        print("[SYS] \(update.isConfirmed ? "✓" : "~") \(update.text)")
                    }
                    systemUpdates.append(update.text)
                }
            }
            
            print("🎵 Streaming same audio to both sources...")
            print("  Microphone stream: Low-latency config (2.0s chunks)")
            print("  System stream: Default config (2.5s chunks)\n")
            
            // Stream the audio to both
            let chunkDuration = 0.5 // 500ms chunks
            let samplesPerChunk = Int(chunkDuration * format.sampleRate)
            var position = 0
            
            while position < Int(buffer.frameLength) {
                let remainingSamples = Int(buffer.frameLength) - position
                let chunkSize = min(samplesPerChunk, remainingSamples)
                
                // Create chunk buffer
                guard let chunkBuffer = AVAudioPCMBuffer(
                    pcmFormat: format,
                    frameCapacity: AVAudioFrameCount(chunkSize)
                ) else {
                    break
                }
                
                // Copy samples
                for channel in 0..<Int(format.channelCount) {
                    if let sourceData = buffer.floatChannelData?[channel],
                       let destData = chunkBuffer.floatChannelData?[channel] {
                        for i in 0..<chunkSize {
                            destData[i] = sourceData[position + i]
                        }
                    }
                }
                chunkBuffer.frameLength = AVAudioFrameCount(chunkSize)
                
                // Stream to both
                await micStream.streamAudio(chunkBuffer)
                await systemStream.streamAudio(chunkBuffer)
                
                position += chunkSize
            }
            
            print("⏳ Finalizing transcriptions...")
            
            // Get final results
            let micFinal = try await micStream.finish()
            let systemFinal = try await systemStream.finish()
            
            // Cancel update tasks
            micTask.cancel()
            systemTask.cancel()
            
            // Print results
            print("\n" + String(repeating: "=", count: 60))
            print("📝 TRANSCRIPTION RESULTS")
            print(String(repeating: "=", count: 60))
            
            print("\n🎙️ MICROPHONE STREAM (Low-latency):")
            print("Final: \(micFinal)")
            print("Updates received: \(micUpdates.count)")
            print("Confirmed: \(await micStream.confirmedTranscript)")
            print("Volatile: \(await micStream.volatileTranscript)")
            
            print("\n💻 SYSTEM AUDIO STREAM (Default):")
            print("Final: \(systemFinal)")
            print("Updates received: \(systemUpdates.count)")
            print("Confirmed: \(await systemStream.confirmedTranscript)")
            print("Volatile: \(await systemStream.volatileTranscript)")
            
            print("\n📊 COMPARISON:")
            print("Match: \(micFinal == systemFinal ? "✅ YES" : "❌ NO")")
            
            // Show active streams
            print("\n🔍 Session info:")
            let activeStreams = await session.activeStreams
            print("Active streams: \(activeStreams.count)")
            for (source, stream) in activeStreams {
                print("  - \(source): \(await stream.source)")
            }
            
            // Cleanup
            await session.cleanup()
            print("\n✅ Session cleaned up")
            
        } catch {
            print("❌ Error: \(error)")
        }
    }
    
    private static func printUsage() {
        print(
            """
            
            Multi-Stream Command Usage:
                fluidaudio multi-stream <audio_file> [options]
            
            Options:
                --debug            Show debug information
                --help, -h         Show this help message
            
            Examples:
                fluidaudio multi-stream audio.wav
                fluidaudio multi-stream audio.wav --debug
            
            This command demonstrates:
            - Loading ASR models once and sharing across streams
            - Creating separate streams for microphone and system audio
            - Different configurations for each stream
            - Parallel transcription with shared resources
            """
        )
    }
}
#endif