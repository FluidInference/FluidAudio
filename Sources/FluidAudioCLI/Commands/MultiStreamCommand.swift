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
            print("❌ No audio files specified")
            printUsage()
            exit(1)
        }
        
        let audioFile1 = arguments[0]
        var audioFile2: String? = nil
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
                // Check if it's a second audio file
                if audioFile2 == nil && arguments[i].hasSuffix(".wav") {
                    audioFile2 = arguments[i]
                } else {
                    print("⚠️  Unknown option: \(arguments[i])")
                }
            }
            i += 1
        }
        
        // Use same file for both streams if only one provided
        let micAudioFile = audioFile1
        let systemAudioFile = audioFile2 ?? audioFile1
        
        print("🎤 Multi-Stream ASR Test")
        print("========================\n")
        
        if audioFile2 != nil {
            print("📁 Processing two different files:")
            print("  Microphone: \(micAudioFile)")
            print("  System: \(systemAudioFile)\n")
        } else {
            print("📁 Processing single file on both streams: \(audioFile1)\n")
        }
        
        do {
            // Load first audio file (microphone)
            let micFileURL = URL(fileURLWithPath: micAudioFile)
            let micFileHandle = try AVAudioFile(forReading: micFileURL)
            let micFormat = micFileHandle.processingFormat
            let micFrameCount = AVAudioFrameCount(micFileHandle.length)
            
            guard let micBuffer = AVAudioPCMBuffer(pcmFormat: micFormat, frameCapacity: micFrameCount) else {
                print("❌ Failed to create microphone audio buffer")
                return
            }
            try micFileHandle.read(into: micBuffer)
            
            // Load second audio file (system)
            let systemFileURL = URL(fileURLWithPath: systemAudioFile)
            let systemFileHandle = try AVAudioFile(forReading: systemFileURL)
            let systemFormat = systemFileHandle.processingFormat
            let systemFrameCount = AVAudioFrameCount(systemFileHandle.length)
            
            guard let systemBuffer = AVAudioPCMBuffer(pcmFormat: systemFormat, frameCapacity: systemFrameCount) else {
                print("❌ Failed to create system audio buffer")
                return
            }
            try systemFileHandle.read(into: systemBuffer)
            
            print("📊 Audio file info:")
            print("🎙️ Microphone file:")
            print("  Sample rate: \(micFormat.sampleRate) Hz")
            print("  Channels: \(micFormat.channelCount)")
            print("  Duration: \(String(format: "%.2f", Double(micFileHandle.length) / micFormat.sampleRate)) seconds")
            
            print("\n💻 System audio file:")
            print("  Sample rate: \(systemFormat.sampleRate) Hz")
            print("  Channels: \(systemFormat.channelCount)")
            print("  Duration: \(String(format: "%.2f", Double(systemFileHandle.length) / systemFormat.sampleRate)) seconds\n")
            
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
            
            print("🎵 Streaming audio files in parallel...")
            print("  Microphone stream: Low-latency config (2.0s chunks)")
            print("  System stream: Default config (2.5s chunks)\n")
            
            // Process both files in parallel
            let micProcessingTask = Task {
                await streamAudioFile(
                    buffer: micBuffer,
                    format: micFormat,
                    to: micStream,
                    label: "MIC",
                    showDebug: showDebug
                )
            }
            
            let systemProcessingTask = Task {
                await streamAudioFile(
                    buffer: systemBuffer,
                    format: systemFormat,
                    to: systemStream,
                    label: "SYS",
                    showDebug: showDebug
                )
            }
            
            // Wait for both to complete
            await micProcessingTask.value
            await systemProcessingTask.value
            
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
    
    /// Helper function to stream an audio file to a stream
    private static func streamAudioFile(
        buffer: AVAudioPCMBuffer,
        format: AVAudioFormat,
        to stream: StreamingAsrManager,
        label: String,
        showDebug: Bool
    ) async {
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
            
            // Stream the chunk
            await stream.streamAudio(chunkBuffer)
            
            if showDebug {
                let progress = Float(position) / Float(buffer.frameLength) * 100
                print("[\(label)] Progress: \(String(format: "%.1f", progress))%")
            }
            
            // Process as fast as possible
            
            position += chunkSize
        }
        
        if showDebug {
            print("[\(label)] Streaming complete")
        }
    }
    
    private static func printUsage() {
        print(
            """
            
            Multi-Stream Command Usage:
                fluidaudio multi-stream <audio_file1> [audio_file2] [options]
            
            Options:
                --debug            Show debug information
                --help, -h         Show this help message
            
            Examples:
                # Process same file on both streams
                fluidaudio multi-stream audio.wav
                
                # Process two different files in parallel
                fluidaudio multi-stream mic_audio.wav system_audio.wav
                
                # With debug output
                fluidaudio multi-stream audio1.wav audio2.wav --debug
                
            
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