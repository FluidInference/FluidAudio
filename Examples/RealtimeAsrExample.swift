import FluidAudio
import Foundation

/// Example demonstrating the improved realtime ASR API with proper decoder state management
@available(macOS 13.0, *)
func runRealtimeAsrExample() async throws {
    print("🎤 Realtime ASR Example")
    print("=======================\n")
    
    // Load ASR models
    print("Loading ASR models...")
    let models = try await AsrModels.downloadAndLoad()
    print("✅ Models loaded successfully\n")
    
    // Create realtime ASR manager
    let realtimeManager = RealtimeAsrManager(models: models)
    
    // Example 1: Basic realtime transcription with decoder state control
    print("Example 1: Basic Realtime Transcription")
    print("---------------------------------------")
    await runBasicExample(manager: realtimeManager)
    
    // Example 2: Multiple concurrent streams
    print("\nExample 2: Multiple Concurrent Streams")
    print("--------------------------------------")
    await runConcurrentStreamsExample(manager: realtimeManager)
    
    // Example 3: Custom configuration with different chunk sizes
    print("\nExample 3: Custom Configuration")
    print("-------------------------------")
    await runCustomConfigExample(manager: realtimeManager)
    
    // Example 4: Stream with manual decoder state resets
    print("\nExample 4: Manual Decoder State Control")
    print("---------------------------------------")
    await runManualStateExample(manager: realtimeManager)
    
    // Clean up
    await realtimeManager.cleanup()
    print("\n✅ All examples completed!")
}

/// Example 1: Basic realtime transcription
@available(macOS 13.0, *)
func runBasicExample(manager: RealtimeAsrManager) async {
    do {
        // Create a stream with default settings
        let stream = try await manager.createStream(
            source: .microphone,
            config: .default,
            resetDecoderState: true  // Reset decoder state for fresh start
        )
        
        print("Created stream: \(stream.id)")
        
        // Simulate audio input (1.5 second chunks)
        let testAudio: [[Float]] = [
            generateTestAudio("Hello, this is a test", duration: 1.5),
            generateTestAudio("of realtime transcription", duration: 1.5),
            generateTestAudio("with proper state management", duration: 1.5)
        ]
        
        // Process audio chunks
        for (index, chunk) in testAudio.enumerated() {
            print("Processing chunk \(index + 1)...")
            
            if let update = try await manager.processAudio(
                streamId: stream.id,
                samples: chunk
            ) {
                print("  Transcription: '\(update.text)'")
                print("  Type: \(update.type)")
                print("  Confidence: \(String(format: "%.2f", update.confidence))")
            }
            
            // Simulate realtime delay
            try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        }
        
        // Get final transcription
        let finalText = try await manager.getFinalTranscription(streamId: stream.id)
        print("\nFinal transcription: '\(finalText)'")
        
        // Get metrics
        let metrics = try await manager.getStreamMetrics(streamId: stream.id)
        printMetrics(metrics)
        
        // Clean up stream
        await manager.removeStream(stream.id)
        
    } catch {
        print("❌ Error in basic example: \(error)")
    }
}

/// Example 2: Multiple concurrent streams
@available(macOS 13.0, *)
func runConcurrentStreamsExample(manager: RealtimeAsrManager) async {
    do {
        // Create two streams with different configurations
        let defaultStream = try await manager.createStream(
            source: .microphone,
            config: .default
        )
        
        let lowLatencyStream = try await manager.createStream(
            source: .system,
            config: .lowLatency
        )
        
        print("Created streams:")
        print("  Default: \(defaultStream.id)")
        print("  Low latency: \(lowLatencyStream.id)")
        
        // Test audio
        let testAudio = generateTestAudio("Processing multiple streams concurrently", duration: 2.0)
        
        // Process both streams concurrently
        async let update1 = manager.processAudio(streamId: defaultStream.id, samples: testAudio)
        async let update2 = manager.processAudio(streamId: lowLatencyStream.id, samples: testAudio)
        
        let (defaultUpdate, lowLatencyUpdate) = try await (update1, update2)
        
        if let update = defaultUpdate {
            print("\nDefault stream result: '\(update.text)'")
        }
        
        if let update = lowLatencyUpdate {
            print("Low latency stream result: '\(update.text)'")
        }
        
        // Clean up
        await manager.removeStream(defaultStream.id)
        await manager.removeStream(lowLatencyStream.id)
        
    } catch {
        print("❌ Error in concurrent streams example: \(error)")
    }
}

/// Example 3: Custom configuration
@available(macOS 13.0, *)
func runCustomConfigExample(manager: RealtimeAsrManager) async {
    do {
        // Create custom configuration
        let customConfig = RealtimeAsrConfig(
            asrConfig: ASRConfig(
                sampleRate: 16000,
                maxSymbolsPerFrame: 5,
                enableDebug: true,
                realtimeMode: true
            ),
            chunkDuration: 2.0,      // 2 second chunks
            overlapDuration: 0.5,    // 0.5 second overlap
            bufferCapacity: 320_000, // 20 seconds buffer
            stabilizationDelay: 2    // 2 chunk stabilization
        )
        
        let stream = try await manager.createStream(
            source: .microphone,
            config: customConfig
        )
        
        print("Created stream with custom config:")
        print("  Chunk duration: \(customConfig.chunkDuration)s")
        print("  Overlap: \(customConfig.overlapDuration)s")
        print("  Stabilization delay: \(customConfig.stabilizationDelay) chunks")
        
        // Process a longer audio segment
        let testAudio = generateTestAudio(
            "This is a longer test with custom configuration settings",
            duration: 2.0
        )
        
        if let update = try await manager.processAudio(
            streamId: stream.id,
            samples: testAudio
        ) {
            print("\nTranscription: '\(update.text)'")
            print("Processing time: \(String(format: "%.3f", update.processingTime))s")
        }
        
        await manager.removeStream(stream.id)
        
    } catch {
        print("❌ Error in custom config example: \(error)")
    }
}

/// Example 4: Manual decoder state control
@available(macOS 13.0, *)
func runManualStateExample(manager: RealtimeAsrManager) async {
    do {
        // Create stream without resetting decoder state
        let stream = try await manager.createStream(
            source: .microphone,
            config: .default,
            resetDecoderState: false  // Keep existing state
        )
        
        print("Processing with persistent decoder state...")
        
        // Process first utterance
        let audio1 = generateTestAudio("The patient", duration: 1.5)
        if let update = try await manager.processAudio(streamId: stream.id, samples: audio1) {
            print("Chunk 1: '\(update.text)'")
        }
        
        // Continue with context
        let audio2 = generateTestAudio("has arrived", duration: 1.5)
        if let update = try await manager.processAudio(streamId: stream.id, samples: audio2) {
            print("Chunk 2: '\(update.text)'")
        }
        
        // Now reset decoder state manually
        print("\nResetting decoder state...")
        // Note: This would require exposing resetDecoderState in RealtimeAsrManager
        // For now, we demonstrate by creating a new stream
        
        let newStream = try await manager.createStream(
            source: .microphone,
            config: .default,
            resetDecoderState: true  // Fresh decoder state
        )
        
        // Process new utterance without context
        let audio3 = generateTestAudio("New sentence begins", duration: 1.5)
        if let update = try await manager.processAudio(streamId: newStream.id, samples: audio3) {
            print("New stream: '\(update.text)'")
        }
        
        await manager.removeStream(stream.id)
        await manager.removeStream(newStream.id)
        
    } catch {
        print("❌ Error in manual state example: \(error)")
    }
}

/// Helper function to generate test audio (placeholder)
func generateTestAudio(_ text: String, duration: Double) -> [Float] {
    // In a real implementation, this would generate actual audio samples
    // For this example, we create dummy data
    let sampleRate = 16000
    let sampleCount = Int(duration * Double(sampleRate))
    return Array(repeating: 0.0, count: sampleCount)
}

/// Helper function to print metrics
func printMetrics(_ metrics: StreamMetrics) {
    print("\nMetrics:")
    print("  Chunks processed: \(metrics.chunkCount)")
    print("  Total audio: \(String(format: "%.2f", metrics.totalAudioDuration))s")
    print("  Processing time: \(String(format: "%.2f", metrics.totalProcessingTime))s")
    if metrics.rtfx > 0 {
        print("  RTFx: \(String(format: "%.2f", metrics.rtfx))x")
    }
    if let ttfw = metrics.timeToFirstWord {
        print("  Time to first word: \(String(format: "%.2f", ttfw))s")
    }
}

// Run the example
@available(macOS 13.0, *)
@main
struct RealtimeAsrExampleApp {
    static func main() async {
        do {
            try await runRealtimeAsrExample()
        } catch {
            print("❌ Fatal error: \(error)")
        }
    }
}