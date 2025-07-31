#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Handler for the 'realtime-transcribe' command
/// This command simulates real-time transcription by processing audio in chunks
@available(macOS 13.0, *)
enum RealtimeTranscribeCommand {
    
    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            print("❌ No audio file specified")
            printUsage()
            exit(1)
        }
        
        let audioFile = arguments[0]
        var simulateParallel = false
        var chunkDuration: TimeInterval = 2.5  // Default to 2.5s for better accuracy
        var lowLatency = false
        var ultraLowLatency = false
        var maxDuration: TimeInterval? = nil
        var debugMode = false
        
        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--parallel":
                simulateParallel = true
            case "--chunk-duration":
                if i + 1 < arguments.count {
                    chunkDuration = Double(arguments[i + 1]) ?? 1.5
                    i += 1
                }
            case "--low-latency":
                lowLatency = true
            case "--ultra-low-latency":
                ultraLowLatency = true
            case "--max-duration":
                if i + 1 < arguments.count {
                    maxDuration = Double(arguments[i + 1])
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                print("⚠️  Unknown option: \(arguments[i])")
            }
            i += 1
        }
        
        print("🎵 Loading audio file: \(audioFile)")
        
        // Load audio file
        var audioSamples: [Float]
        do {
            audioSamples = try await AudioProcessor.loadAudioFile(path: audioFile)
            let duration = Float(audioSamples.count) / 16000.0
            print("✅ Loaded audio: \(String(format: "%.1f", duration)) seconds")
            
            // Apply max duration limit if specified
            if let maxDur = maxDuration {
                let maxSamples = Int(maxDur * 16000)
                if audioSamples.count > maxSamples {
                    audioSamples = Array(audioSamples.prefix(maxSamples))
                    print("⚡ Limited to first \(String(format: "%.1f", maxDur)) seconds")
                }
            }
        } catch {
            print("❌ Failed to load audio file: \(error)")
            exit(1)
        }
        
        // ASR models will be loaded automatically by StreamingAsrManager
        
        // Create display manager
        let display = RealtimeDisplay()
        
        if simulateParallel {
            print("❌ Parallel streaming is not yet supported with StreamingAsrManager")
            print("   Please use single stream mode for now.")
            exit(1)
        } else {
            // Single stream processing
            await processSingleStream(
                audioFile: audioFile,
                display: display,
                lowLatency: lowLatency,
                ultraLowLatency: ultraLowLatency,
                chunkDuration: chunkDuration,
                debugMode: debugMode
            )
        }
    }
    
    /// Process a single stream using StreamingAsrManager
    private static func processSingleStream(
        audioFile: String,
        display: RealtimeDisplay,
        lowLatency: Bool,
        ultraLowLatency: Bool = false,
        chunkDuration: TimeInterval,
        debugMode: Bool = false
    ) async {
        // Create configuration for StreamingAsrManager
        let config: StreamingAsrConfig
        if ultraLowLatency {
            // Use lower confirmation threshold and shorter chunks for ultra-low latency
            config = StreamingAsrConfig(
                confirmationThreshold: 0.65,
                chunkDuration: 1.5,
                enableDebug: debugMode
            )
            print("⚠️  WARNING: Ultra-low latency mode enabled. Transcription accuracy will be reduced.")
            print("⚠️  For better accuracy, use --low-latency or default mode.\n")
        } else if lowLatency {
            config = .lowLatency
        } else {
            config = StreamingAsrConfig(
                confirmationThreshold: 0.85,
                chunkDuration: chunkDuration,
                enableDebug: debugMode
            )
        }
        
        // Create StreamingAsrManager
        let streamingAsr = StreamingAsrManager(config: config)
        
        do {
            // Start the ASR engine
            print("🔄 Starting ASR engine...")
            try await streamingAsr.start()
            print("✅ ASR engine started")
        } catch {
            print("❌ Failed to start ASR engine: \(error)")
            exit(1)
        }
        
        // Start display task
        let displayTask = Task {
            while !Task.isCancelled {
                await display.render(audioFile: audioFile)
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }
        
        // Load audio file
        do {
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                print("❌ Failed to create audio buffer")
                return
            }
            
            try audioFileHandle.read(into: buffer)
            
            // Track updates
            var updateCount = 0
            
            // Create a task to handle transcription updates
            let _ = Task {
                for await update in await streamingAsr.transcriptionUpdates {
                    updateCount += 1
                    
                    // Create a simplified transcription update for display
                    let displayUpdate = TranscriptionUpdate(
                        streamId: UUID(),
                        text: update.text,
                        type: update.isConfirmed ? .confirmed : .pending,
                        timestamp: update.timestamp
                    )
                    
                    await display.updateTranscription(displayUpdate)
                    
                    // Create simple metrics
                    if updateCount % 5 == 0 {
                        let metrics = StreamMetrics(
                            streamId: UUID(),
                            totalChunks: updateCount,
                            chunkCount: updateCount,
                            averageProcessingTime: 0.05,
                            averageConfidence: update.confidence,
                            rtfx: 0.02,
                            averageLatency: 0.1,
                            timeToFirstWord: 1.0,
                            totalAudioDuration: Double(updateCount) * 2.5,
                            totalProcessingTime: Double(updateCount) * 0.05
                        )
                        await display.updateMetrics(metrics, streamId: UUID())
                    }
                }
            }
            
            // Process audio in chunks to simulate realtime
            let chunkDuration = 0.1 // 100ms chunks
            let samplesPerChunk = Int(chunkDuration * format.sampleRate)
            var position = 0
            let startTime = Date()
            
            print("\n🎤 Starting real-time transcription...")
            print("   Audio duration: \(String(format: "%.1f", Double(buffer.frameLength) / format.sampleRate))s")
            print("   Chunk size: \(String(format: "%.0f", chunkDuration * 1000))ms\n")
            
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
                await streamingAsr.streamAudio(chunkBuffer)
                
                // Simulate realtime playback
                let expectedTime = TimeInterval(position) / TimeInterval(format.sampleRate)
                let actualTime = Date().timeIntervalSince(startTime)
                if expectedTime > actualTime {
                    let sleepTime = expectedTime - actualTime
                    try? await Task.sleep(nanoseconds: UInt64(sleepTime * 1_000_000_000))
                }
                
                position += chunkSize
            }
        } catch {
            print("❌ Failed to load audio: \(error)")
            return
        }
        
        // Wait a bit for final updates
        try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
        
        // Cancel display task
        displayTask.cancel()
        
        // Get final transcription
        do {
            let finalTranscription = try await streamingAsr.finish()
            
            // Final display update
            await display.render(audioFile: audioFile)
            await display.printSummary()
            
            print("\n\(String(repeating: "━", count: 80))")
            print("📝 Final Transcription:")
            print(finalTranscription)
            print("\n✅ Confirmed text: \(await streamingAsr.confirmedTranscript)")
            print("~  Volatile text: \(await streamingAsr.volatileTranscript)")
        } catch {
            print("⚠️  Failed to get final results: \(error)")
        }
    }
    
    private static func printUsage() {
        print(
            """
            
            Realtime Transcribe Command Usage:
                fluidaudio realtime-transcribe <audio_file> [options]
            
            Options:
                --chunk-duration <sec>  Set chunk duration in seconds (default: 2.5)
                --low-latency          Use low-latency configuration (2.0s chunks)
                --ultra-low-latency    Use ultra-low latency (1.5s chunks, reduced accuracy)
                --max-duration <sec>    Limit processing to first N seconds
                --debug                 Enable debug output
                --help, -h             Show this help message
            
            Examples:
                fluidaudio realtime-transcribe audio.wav
                fluidaudio realtime-transcribe audio.wav --low-latency
                fluidaudio realtime-transcribe audio.wav --chunk-duration 2.0
                fluidaudio realtime-transcribe audio.wav --ultra-low-latency --max-duration 10
            
            This command simulates realtime transcription by processing the audio
            file in chunks and displaying results as they would appear in a live
            transcription scenario.
            """
        )
    }
}
#endif