#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Handler for the 'realtime-transcribe' command
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
        
        // Load ASR models
        print("🔄 Loading ASR models...")
        let models: AsrModels
        do {
            models = try await AsrModels.downloadAndLoad()
            print("✅ Models loaded successfully")
        } catch {
            print("❌ Failed to load ASR models: \(error)")
            exit(1)
        }
        
        // Create realtime manager
        let realtimeManager = RealtimeAsrManager(models: models)
        
        // Create display manager
        let display = RealtimeDisplay()
        
        if simulateParallel {
            // Demo: Process same file with 2 different configs
            await processParallelStreams(
                audioFile: audioFile,
                audioSamples: audioSamples,
                manager: realtimeManager,
                display: display
            )
        } else {
            // Single stream processing
            await processSingleStream(
                audioFile: audioFile,
                audioSamples: audioSamples,
                manager: realtimeManager,
                display: display,
                lowLatency: lowLatency,
                ultraLowLatency: ultraLowLatency,
                chunkDuration: chunkDuration,
                debugMode: debugMode
            )
        }
    }
    
    /// Process a single stream
    private static func processSingleStream(
        audioFile: String,
        audioSamples: [Float],
        manager: RealtimeAsrManager,
        display: RealtimeDisplay,
        lowLatency: Bool,
        ultraLowLatency: Bool = false,
        chunkDuration: TimeInterval,
        debugMode: Bool = false
    ) async {
        // Create configuration
        let config: RealtimeAsrConfig
        if ultraLowLatency {
            config = RealtimeAsrConfig.ultraLowLatency
            print("⚠️  WARNING: Ultra-low latency mode enabled. Transcription accuracy will be reduced.")
            print("⚠️  For better accuracy, use --low-latency or default mode.\n")
        } else if lowLatency {
            config = RealtimeAsrConfig.lowLatency
        } else {
            config = RealtimeAsrConfig(
                asrConfig: ASRConfig(
                    sampleRate: 16000,
                    maxSymbolsPerFrame: 3,
                    enableDebug: debugMode,
                    realtimeMode: true,
                    chunkSizeMs: Int(chunkDuration * 1000)
                ),
                chunkDuration: chunkDuration,
                bufferCapacity: 160_000,
                stabilizationDelay: 3
            )
        }
        
        // Create stream
        let stream: RealtimeAsrStream
        do {
            stream = try await manager.createStream(source: .microphone, config: config)
            print("✅ Created realtime stream")
        } catch {
            print("❌ Failed to create stream: \(error)")
            exit(1)
        }
        
        // Start display task
        let displayTask = Task {
            while !Task.isCancelled {
                await display.render(audioFile: audioFile)
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }
        
        // Process audio in chunks to simulate realtime
        let sampleRate = 16000
        let samplesPerChunk = Int(0.1 * Double(sampleRate)) // 100ms chunks for smooth simulation
        var position = 0
        let startTime = Date()
        
        // Create a task to handle transcription updates
        let updateTask = Task {
            do {
                let updateStream = try await manager.getTranscriptionStream(streamId: stream.id)
                for try await update in updateStream {
                    await display.updateTranscription(update)
                    
                    // Update metrics periodically
                    if update.type == .pending {
                        let metrics = try await manager.getStreamMetrics(streamId: stream.id)
                        await display.updateMetrics(metrics, streamId: stream.id)
                    }
                }
            } catch {
                print("⚠️  Update stream error: \(error)")
            }
        }
        
        // Feed audio chunks
        print("Starting to feed audio chunks. Total samples: \(audioSamples.count)")
        while position < audioSamples.count {
            let endPosition = min(position + samplesPerChunk, audioSamples.count)
            let chunk = Array(audioSamples[position..<endPosition])
            
            print("Feeding chunk at position \(position)-\(endPosition) (\(chunk.count) samples)")
            
            // Process chunk
            do {
                _ = try await manager.processAudio(streamId: stream.id, samples: chunk)
            } catch {
                print("⚠️  Failed to process audio chunk: \(error)")
            }
            
            // Simulate realtime playback
            let expectedTime = TimeInterval(position) / TimeInterval(sampleRate)
            let actualTime = Date().timeIntervalSince(startTime)
            if expectedTime > actualTime {
                let sleepTime = expectedTime - actualTime
                try? await Task.sleep(nanoseconds: UInt64(sleepTime * 1_000_000_000))
            }
            
            position = endPosition
        }
        
        // Wait a bit for final updates
        try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
        
        // Cancel tasks
        updateTask.cancel()
        displayTask.cancel()
        
        // Get final transcription and metrics
        do {
            let finalTranscription = try await manager.getFinalTranscription(streamId: stream.id)
            let metrics = try await manager.getStreamMetrics(streamId: stream.id)
            
            await display.updateMetrics(metrics, streamId: stream.id)
            await display.render(audioFile: audioFile)
            await display.printSummary()
            
            print("\n\(String(repeating: "━", count: 80))")
            print("📝 Final Transcription:")
            print(finalTranscription)
        } catch {
            print("⚠️  Failed to get final results: \(error)")
        }
        
        // Clean up
        await manager.removeStream(stream.id)
    }
    
    /// Process parallel streams for demo
    private static func processParallelStreams(
        audioFile: String,
        audioSamples: [Float],
        manager: RealtimeAsrManager,
        display: RealtimeDisplay
    ) async {
        // Create two streams with different configs
        let defaultStream: RealtimeAsrStream
        let lowLatencyStream: RealtimeAsrStream
        
        do {
            defaultStream = try await manager.createStream(
                source: .microphone,
                config: .default
            )
            lowLatencyStream = try await manager.createStream(
                source: .microphone,
                config: .lowLatency
            )
            print("✅ Created 2 parallel streams")
        } catch {
            print("❌ Failed to create streams: \(error)")
            exit(1)
        }
        
        // Start display task
        let displayTask = Task {
            while !Task.isCancelled {
                await display.render(audioFile: audioFile)
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }
        
        // Create update tasks for both streams
        let updateTask1 = Task {
            do {
                let updateStream = try await manager.getTranscriptionStream(streamId: defaultStream.id)
                for try await update in updateStream {
                    await display.updateTranscription(update, streamName: "Default Config")
                    if update.type == .pending {
                        let metrics = try await manager.getStreamMetrics(streamId: defaultStream.id)
                        await display.updateMetrics(metrics, streamId: defaultStream.id)
                    }
                }
            } catch {}
        }
        
        let updateTask2 = Task {
            do {
                let updateStream = try await manager.getTranscriptionStream(streamId: lowLatencyStream.id)
                for try await update in updateStream {
                    await display.updateTranscription(update, streamName: "Low Latency Config")
                    if update.type == .pending {
                        let metrics = try await manager.getStreamMetrics(streamId: lowLatencyStream.id)
                        await display.updateMetrics(metrics, streamId: lowLatencyStream.id)
                    }
                }
            } catch {}
        }
        
        // Process audio in chunks
        let sampleRate = 16000
        let samplesPerChunk = Int(0.1 * Double(sampleRate)) // 100ms chunks
        var position = 0
        let startTime = Date()
        
        while position < audioSamples.count {
            let endPosition = min(position + samplesPerChunk, audioSamples.count)
            let chunk = Array(audioSamples[position..<endPosition])
            
            // Process chunk for both streams
            async let _ = try? manager.processAudio(streamId: defaultStream.id, samples: chunk)
            async let _ = try? manager.processAudio(streamId: lowLatencyStream.id, samples: chunk)
            
            // Simulate realtime playback
            let expectedTime = TimeInterval(position) / TimeInterval(sampleRate)
            let actualTime = Date().timeIntervalSince(startTime)
            if expectedTime > actualTime {
                let sleepTime = expectedTime - actualTime
                try? await Task.sleep(nanoseconds: UInt64(sleepTime * 1_000_000_000))
            }
            
            position = endPosition
        }
        
        // Wait for final updates
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        
        // Cancel tasks
        updateTask1.cancel()
        updateTask2.cancel()
        displayTask.cancel()
        
        // Final display
        await display.render(audioFile: audioFile)
        await display.printSummary()
        
        // Clean up
        await manager.removeAllStreams()
    }
    
    private static func printUsage() {
        print(
            """
            
            Realtime Transcribe Command Usage:
                fluidaudio realtime-transcribe <audio_file> [options]
            
            Options:
                --parallel              Process with parallel streams (demo)
                --chunk-duration <sec>  Set chunk duration in seconds (default: 2.5)
                --low-latency          Use low-latency configuration (2.0s chunks)
                --ultra-low-latency    Use ultra-low latency (1.5s chunks, reduced accuracy)
                --max-duration <sec>    Limit processing to first N seconds
                --debug                 Enable debug output
                --help, -h             Show this help message
            
            Examples:
                fluidaudio realtime-transcribe audio.wav
                fluidaudio realtime-transcribe audio.wav --low-latency
                fluidaudio realtime-transcribe audio.wav --parallel
                fluidaudio realtime-transcribe audio.wav --chunk-duration 2.0
            
            This command simulates realtime transcription by processing the audio
            file in chunks and displaying results as they would appear in a live
            transcription scenario.
            """
        )
    }
}
#endif