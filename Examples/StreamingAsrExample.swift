import AVFoundation
import FluidAudio
import Foundation

/// Example demonstrating the new StreamingAsrManager API
/// This shows how simple it is to implement real-time transcription with automatic audio conversion
@available(macOS 13.0, *)
func runStreamingAsrExample() async throws {
    print("🎙️ Streaming ASR Example")
    print("========================\n")
    
    // Create streaming ASR manager
    let streamingAsr = StreamingAsrManager()
    
    // Start the engine (downloads models if needed)
    print("Starting ASR engine...")
    try await streamingAsr.start()
    print("✅ ASR engine ready\n")
    
    // Set up audio engine for microphone input
    let audioEngine = AVAudioEngine()
    let inputNode = audioEngine.inputNode
    
    // Install tap to capture audio
    // Note: We don't need to worry about the format - StreamingAsrManager handles conversion
    let inputFormat = inputNode.outputFormat(forBus: 0)
    print("Microphone format: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) channels")
    print("StreamingAsrManager will automatically convert to 16kHz mono\n")
    
    inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
        // Simply pass the buffer - no manual conversion needed!
        streamingAsr.streamAudio(buffer)
    }
    
    // Listen for transcription updates
    let transcriptionTask = Task {
        print("Listening for transcriptions...\n")
        
        for await update in streamingAsr.transcriptionUpdates {
            // Clear line and move cursor to beginning
            print("\r", terminator: "")
            
            if update.isConfirmed {
                // High confidence - show as confirmed
                print("✓ Confirmed: \(update.text)", terminator: "")
            } else {
                // Low confidence - show as volatile (would be purple in UI)
                print("~ Volatile: \(update.text)", terminator: "")
            }
            
            // Flush output
            fflush(stdout)
        }
    }
    
    // Start audio engine
    try audioEngine.start()
    print("🎤 Microphone active. Start speaking...")
    print("Press Enter to stop.\n")
    
    // Wait for user to press Enter
    _ = readLine()
    
    // Stop audio engine
    audioEngine.stop()
    inputNode.removeTap(onBus: 0)
    
    // Get final transcription
    print("\n\nFinalizing transcription...")
    let finalText = try await streamingAsr.finish()
    
    // Cancel the transcription task
    transcriptionTask.cancel()
    
    // Display results
    print("\n✅ Transcription Complete!")
    print("========================")
    print("Final text: \(finalText)")
    print("\nStatistics:")
    print("- Confirmed text: \(streamingAsr.confirmedTranscript)")
    print("- Last volatile text: \(streamingAsr.volatileTranscript)")
}

/// Example showing different configuration options
@available(macOS 13.0, *)
func runConfigurationExamples() async throws {
    print("\n📋 Configuration Examples")
    print("========================\n")
    
    // Example 1: Default configuration
    print("1. Default Configuration:")
    let defaultManager = StreamingAsrManager(config: .default)
    print("   - Chunk duration: 2.5s")
    print("   - Confirmation threshold: 0.85")
    print("   - Balanced accuracy/latency\n")
    
    // Example 2: Low latency configuration
    print("2. Low Latency Configuration:")
    let lowLatencyManager = StreamingAsrManager(config: .lowLatency)
    print("   - Chunk duration: 2.0s")
    print("   - Confirmation threshold: 0.75")
    print("   - Faster updates, slightly less accurate\n")
    
    // Example 3: High accuracy configuration
    print("3. High Accuracy Configuration:")
    let highAccuracyManager = StreamingAsrManager(config: .highAccuracy)
    print("   - Chunk duration: 3.0s")
    print("   - Confirmation threshold: 0.90")
    print("   - More accurate, slightly slower\n")
    
    // Example 4: Custom configuration
    print("4. Custom Configuration:")
    let customConfig = StreamingAsrConfig(
        confirmationThreshold: 0.8,
        chunkDuration: 2.2,
        enableDebug: true
    )
    let customManager = StreamingAsrManager(config: customConfig)
    print("   - Chunk duration: 2.2s")
    print("   - Confirmation threshold: 0.80")
    print("   - Debug logging enabled\n")
}

/// Example showing integration with SwiftUI
@available(macOS 13.0, *)
func swiftUIIntegrationExample() {
    print("\n🖼️ SwiftUI Integration Example")
    print("================================\n")
    
    print("""
    ```swift
    import SwiftUI
    import FluidAudio
    
    @MainActor
    class TranscriptionViewModel: ObservableObject {
        @Published var confirmedText = ""
        @Published var volatileText = ""
        @Published var isRecording = false
        
        private var streamingAsr: StreamingAsrManager?
        private var audioEngine: AVAudioEngine?
        
        func startRecording() async throws {
            // Initialize
            streamingAsr = StreamingAsrManager()
            audioEngine = AVAudioEngine()
            
            // Start ASR
            try await streamingAsr?.start()
            
            // Set up audio capture
            let inputNode = audioEngine!.inputNode
            inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputNode.outputFormat(forBus: 0)) { [weak self] buffer, _ in
                self?.streamingAsr?.streamAudio(buffer)
            }
            
            // Listen for updates
            Task { [weak self] in
                guard let updates = self?.streamingAsr?.transcriptionUpdates else { return }
                
                for await update in updates {
                    await MainActor.run {
                        if update.isConfirmed {
                            self?.confirmedText = self?.streamingAsr?.confirmedTranscript ?? ""
                            self?.volatileText = self?.streamingAsr?.volatileTranscript ?? ""
                        } else {
                            self?.volatileText = update.text
                        }
                    }
                }
            }
            
            // Start audio
            try audioEngine!.start()
            isRecording = true
        }
        
        func stopRecording() async throws -> String {
            isRecording = false
            audioEngine?.stop()
            audioEngine?.inputNode.removeTap(onBus: 0)
            
            return try await streamingAsr?.finish() ?? ""
        }
    }
    
    struct TranscriptionView: View {
        @StateObject private var viewModel = TranscriptionViewModel()
        
        var body: some View {
            VStack(alignment: .leading, spacing: 20) {
                // Confirmed text
                Text(viewModel.confirmedText)
                    .font(.body)
                    .foregroundColor(.primary)
                
                // Volatile text (purple like Apple's API)
                Text(viewModel.volatileText)
                    .font(.body)
                    .foregroundColor(.purple.opacity(0.8))
                
                // Record button
                Button(action: {
                    Task {
                        if viewModel.isRecording {
                            let finalText = try await viewModel.stopRecording()
                            print("Final: \\(finalText)")
                        } else {
                            try await viewModel.startRecording()
                        }
                    }
                }) {
                    Label(
                        viewModel.isRecording ? "Stop" : "Record",
                        systemImage: viewModel.isRecording ? "stop.circle.fill" : "mic.circle.fill"
                    )
                }
                .buttonStyle(.borderedProminent)
                .tint(viewModel.isRecording ? .red : .blue)
            }
            .padding()
        }
    }
    ```
    """)
}

// Run the example
@available(macOS 13.0, *)
@main
struct StreamingAsrExampleApp {
    static func main() async {
        do {
            // Run the main streaming example
            try await runStreamingAsrExample()
            
            // Show configuration options
            try await runConfigurationExamples()
            
            // Show SwiftUI integration
            swiftUIIntegrationExample()
            
        } catch {
            print("❌ Error: \(error)")
        }
    }
}