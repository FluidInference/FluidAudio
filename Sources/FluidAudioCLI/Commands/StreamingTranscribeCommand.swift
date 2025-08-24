#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Thread-safe tracker for transcription updates and audio position
@available(macOS 13.0, *)
actor TranscriptionTracker {
    private var volatileUpdates: [String] = []
    private var confirmedUpdates: [String] = []
    private var currentAudioPosition: Double = 0.0
    private let startTime: Date

    init() {
        self.startTime = Date()
    }

    func addVolatileUpdate(_ text: String) {
        volatileUpdates.append(text)
    }

    func addConfirmedUpdate(_ text: String) {
        confirmedUpdates.append(text)
    }

    func updateAudioPosition(_ position: Double) {
        currentAudioPosition = position
    }

    func getCurrentAudioPosition() -> Double {
        return currentAudioPosition
    }

    func getElapsedProcessingTime() -> Double {
        return Date().timeIntervalSince(startTime)
    }

    func getVolatileCount() -> Int {
        return volatileUpdates.count
    }

    func getConfirmedCount() -> Int {
        return confirmedUpdates.count
    }
}

/// Command to transcribe audio files using StreamingAsrManager
@available(macOS 13.0, *)
enum StreamingTranscribeCommand {

    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            print("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                print("⚠️  Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("🎤 Audio Transcription")
        print("=====================\n")

        // Test loading audio at different sample rates
        await testAudioConversion(audioFile: audioFile)

        // Test transcription with StreamingAsrManager
        await testStreamingTranscription(audioFile: audioFile)
    }

    /// Test audio conversion capabilities
    private static func testAudioConversion(audioFile: String) async {
        print("📊 Testing Audio Conversion")
        print("--------------------------")

        do {
            // Load the audio file info
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat

            print("Original format:")
            print("  Sample rate: \(format.sampleRate) Hz")
            print("  Channels: \(format.channelCount)")
            print("  Format: \(format.commonFormat.rawValue)")
            print(
                "  Duration: \(String(format: "%.2f", Double(audioFileHandle.length) / format.sampleRate)) seconds"
            )
            print()

            // The StreamingAsrManager will handle conversion automatically
            print("StreamingAsrManager will automatically convert to 16kHz mono\n")

        } catch {
            print("Failed to load audio file info: \(error)")
        }
    }

    /// Test streaming transcription
    private static func testStreamingTranscription(audioFile: String) async {
        print("🎙️ Testing Streaming Transcription")
        print("----------------------------------")

        // Use default configuration
        let config = StreamingAsrConfig.default
        print("Using default configuration")

        print("Configuration:")
        print("  Chunk duration: \(config.chunkDuration)s")
        print("  Confirmation threshold: \(config.confirmationThreshold)")
        print()

        // Create StreamingAsrManager
        let streamingAsr = StreamingAsrManager(config: config)

        do {
            // Start the engine
            print("Starting ASR engine...")
            try await streamingAsr.start()
            print("Engine started\n")

            // Load audio file
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
            else {
                print("Failed to create audio buffer")
                return
            }

            try audioFileHandle.read(into: buffer)

            // Track transcription updates
            let tracker = TranscriptionTracker()

            // Listen for updates
            let updateTask = Task {
                for await update in await streamingAsr.transcriptionUpdates {
                    let audioPos = await tracker.getCurrentAudioPosition()
                    let audioTimestamp = String(format: "%.2f", audioPos)

                    if update.isConfirmed {
                        print(
                            "[\(audioTimestamp)s] ✓ Confirmed: '\(update.text)' (confidence: \(String(format: "%.3f", update.confidence)))"
                        )
                        await tracker.addConfirmedUpdate(update.text)
                    } else {
                        print(
                            "[\(audioTimestamp)s] ~ Volatile: '\(update.text)' (confidence: \(String(format: "%.3f", update.confidence)))"
                        )
                        await tracker.addVolatileUpdate(update.text)
                    }
                }
            }

            // Simulate streaming by feeding audio in chunks
            let chunkDuration = 1.0  // 1 second chunks
            let samplesPerChunk = Int(chunkDuration * format.sampleRate)
            var position = 0

            print("Streaming audio...")
            while position < Int(buffer.frameLength) {
                let remainingSamples = Int(buffer.frameLength) - position
                let chunkSize = min(samplesPerChunk, remainingSamples)

                // Create a chunk buffer
                guard
                    let chunkBuffer = AVAudioPCMBuffer(
                        pcmFormat: format,
                        frameCapacity: AVAudioFrameCount(chunkSize)
                    )
                else {
                    break
                }

                // Copy samples to chunk
                for channel in 0..<Int(format.channelCount) {
                    if let sourceData = buffer.floatChannelData?[channel],
                        let destData = chunkBuffer.floatChannelData?[channel]
                    {
                        for i in 0..<chunkSize {
                            destData[i] = sourceData[position + i]
                        }
                    }
                }
                chunkBuffer.frameLength = AVAudioFrameCount(chunkSize)

                // Update audio time position in tracker
                let audioTimePosition = Double(position) / format.sampleRate
                await tracker.updateAudioPosition(audioTimePosition)

                // Stream the chunk
                await streamingAsr.streamAudio(chunkBuffer)

                // Process as fast as possible - no artificial delays

                position += chunkSize
            }

            print("\nFinalizing transcription...")
            let finalText = try await streamingAsr.finish()

            // Cancel update task
            updateTask.cancel()

            // Print results
            print("\n" + String(repeating: "=", count: 50))
            print("📝 TRANSCRIPTION RESULTS")
            print(String(repeating: "=", count: 50))
            print("\nFinal transcription:")
            print(finalText)
            print("\nStatistics:")
            print("  Total volatile updates: \(await tracker.getVolatileCount())")
            print("  Total confirmed updates: \(await tracker.getConfirmedCount())")
            print("  Final confirmed text: \(await streamingAsr.confirmedTranscript)")
            print("  Final volatile text: \(await streamingAsr.volatileTranscript)")

            let processingTime = await tracker.getElapsedProcessingTime()
            let audioDuration = Double(audioFileHandle.length) / format.sampleRate
            let rtfx = audioDuration / processingTime

            print("\nPerformance:")
            print("  Audio duration: \(String(format: "%.2f", audioDuration))s")
            print("  Processing time: \(String(format: "%.2f", processingTime))s")
            print("  RTFx: \(String(format: "%.2f", rtfx))x")

        } catch {
            print("Streaming transcription failed: \(error)")
        }
    }

    private static func printUsage() {
        print(
            """

            Transcribe Command Usage:
                fluidaudio transcribe <audio_file> [options]

            Options:
                --help, -h         Show this help message

            Examples:
                fluidaudio transcribe audio.wav

            This command uses StreamingAsrManager which provides:
            - Automatic audio format conversion to 16kHz mono
            - Real-time transcription updates
            - Volatile (unconfirmed) and confirmed text states
            - AsyncStream-based API for easy integration
            """
        )
    }
}
#endif
