#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Thread-safe tracker for transcription updates
@available(macOS 13.0, *)
actor TranscriptionTracker {
    private var volatileUpdates: [String] = []
    private var confirmedUpdates: [String] = []

    func addVolatileUpdate(_ text: String) {
        volatileUpdates.append(text)
    }

    func addConfirmedUpdate(_ text: String) {
        confirmedUpdates.append(text)
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

    /// Convert audio buffer to 16kHz mono
    private static func convertTo16kHzMono(_ buffer: AVAudioPCMBuffer) async throws -> [Float] {
        let format = buffer.format
        let sourceRate = Float(format.sampleRate)
        let targetRate: Float = 16000.0
        let frameCount = buffer.frameLength

        // Get channel data
        guard let channelData = buffer.floatChannelData else {
            throw NSError(
                domain: "Transcribe", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to access audio data"])
        }

        let channelCount = Int(format.channelCount)

        // First, convert to mono if needed
        var monoSamples: [Float]
        if channelCount > 1 {
            // Average all channels
            monoSamples = Array(repeating: 0, count: Int(frameCount))
            for frame in 0..<Int(frameCount) {
                var sum: Float = 0
                for channel in 0..<channelCount {
                    sum += channelData[channel][frame]
                }
                monoSamples[frame] = sum / Float(channelCount)
            }
        } else {
            // Already mono
            monoSamples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(frameCount)))
        }

        // Then resample if needed
        if sourceRate != targetRate {
            let resampleRatio = targetRate / sourceRate
            let outputLength = Int(Float(monoSamples.count) * resampleRatio)
            var resampled = Array(repeating: Float(0), count: outputLength)

            // Simple linear interpolation resampling
            for i in 0..<outputLength {
                let sourceIndex = Float(i) / resampleRatio
                let index = Int(sourceIndex)
                let fraction = sourceIndex - Float(index)

                if index < monoSamples.count - 1 {
                    resampled[i] = monoSamples[index] * (1 - fraction) + monoSamples[index + 1] * fraction
                } else if index < monoSamples.count {
                    resampled[i] = monoSamples[index]
                }
            }

            return resampled
        } else {
            return monoSamples
        }
    }

    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            print("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var configType = "default"
        var useBatchMode = false

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--batch":
                useBatchMode = true
            case "--config":
                if i + 1 < arguments.count {
                    configType = arguments[i + 1]
                    i += 1
                }
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

        if useBatchMode {
            // Use batch processing with sliding window support
            await runBatchTranscription(audioFile: audioFile)
        } else {
            // Test loading audio at different sample rates
            await testAudioConversion(audioFile: audioFile)

            // Test transcription with StreamingAsrManager
            await testStreamingTranscription(
                audioFile: audioFile,
                configType: configType
            )
        }
    }

    /// Run batch transcription using AsrManager directly (with sliding window support)
    private static func runBatchTranscription(audioFile: String) async {
        print("🎙️ Batch Transcription Mode (with sliding window)")
        print("--------------------------------------------------")

        // Show sliding window configuration
        let overlapRatio = ProcessInfo.processInfo.environment["OVERLAP_RATIO"] ?? "0.1"
        let enableDebug = ProcessInfo.processInfo.environment["LOG_LEVEL"] == "DEBUG"
        print("Configuration:")
        print("  Sliding window: ENABLED")
        print("  Overlap ratio: \(overlapRatio) (\(Int(Double(overlapRatio)! * 100))%)")
        print("  Chunk size: 10 seconds")
        print("  Debug mode: \(enableDebug)")
        print()

        do {
            // Load audio file
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFile = try AVAudioFile(forReading: audioFileURL)
            let format = audioFile.processingFormat
            let frameCount = UInt32(audioFile.length)

            print("Audio Info:")
            print("  Sample rate: \(format.sampleRate) Hz")
            print("  Duration: \(String(format: "%.2f", Double(frameCount) / format.sampleRate)) seconds")
            print()

            // Read audio data
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                print("Failed to create audio buffer")
                return
            }

            try audioFile.read(into: buffer)

            // Convert to 16kHz mono if needed
            let audioSamples: [Float]
            if format.sampleRate != 16000 || format.channelCount != 1 {
                print("Converting to 16kHz mono...")
                audioSamples = try await convertTo16kHzMono(buffer)
            } else {
                guard let floatData = buffer.floatChannelData else {
                    print("Failed to access audio data")
                    return
                }
                audioSamples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(frameCount)))
            }

            // Load models
            print("Loading ASR models...")
            let models = try await AsrModels.downloadAndLoad()

            // Create ASR manager with debug config if needed
            let asrConfig = ASRConfig(
                sampleRate: 16000,
                maxSymbolsPerFrame: 3,
                enableDebug: enableDebug
            )
            let asrManager = AsrManager(config: asrConfig)
            try await asrManager.initialize(models: models)

            // Process with AsrManager.transcribe (uses ChunkProcessor with sliding window)
            print("Starting transcription with sliding window...")
            let startTime = Date()

            let result = try await asrManager.transcribe(audioSamples)

            let processingTime = Date().timeIntervalSince(startTime)
            let audioDuration = Double(audioSamples.count) / 16000.0
            let rtfx = audioDuration / processingTime

            print()
            print("==================================================")
            print("📝 TRANSCRIPTION RESULT (Batch Mode)")
            print("==================================================")
            print()
            print(result.text)
            print()
            print("==================================================")
            print("Performance:")
            print("  Audio duration: \(String(format: "%.2f", audioDuration))s")
            print("  Processing time: \(String(format: "%.2f", processingTime))s")
            print("  RTFx: \(String(format: "%.2f", rtfx))x")
            print("  Confidence: \(String(format: "%.3f", result.confidence))")

        } catch {
            print("Error: \(error)")
        }
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
    private static func testStreamingTranscription(
        audioFile: String,
        configType: String
    ) async {
        print("🎙️ Testing Streaming Transcription")
        print("----------------------------------")

        // Select configuration
        let config: StreamingAsrConfig
        switch configType {
        case "low-latency":
            config = .lowLatency
            print("Using low-latency configuration")
        case "high-accuracy":
            config = .highAccuracy
            print("Using high-accuracy configuration")
        default:
            config = .default
            print("Using default configuration")
        }

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
            let startTime = Date()

            // Listen for updates
            let updateTask = Task {
                for await update in await streamingAsr.transcriptionUpdates {
                    if update.isConfirmed {
                        print(
                            "✓ Confirmed: '\(update.text)' (confidence: \(String(format: "%.3f", update.confidence)))"
                        )
                        await tracker.addConfirmedUpdate(update.text)
                    } else {
                        print(
                            "~ Volatile: '\(update.text)' (confidence: \(String(format: "%.3f", update.confidence)))"
                        )
                        await tracker.addVolatileUpdate(update.text)
                    }
                }
            }

            // Simulate streaming by feeding audio in chunks
            let chunkDuration = 0.1  // 100ms chunks
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
            print("\(await streamingAsr.confirmedTranscript)")
            print("  Final volatile text: \(await streamingAsr.volatileTranscript)")

            let processingTime = Date().timeIntervalSince(startTime)
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
                --batch            Use batch mode with sliding window (better accuracy)
                --config <type>    Configuration type: default, low-latency, high-accuracy
                --help, -h         Show this help message

            Environment Variables (for batch mode):
                OVERLAP_RATIO=0.3  Set sliding window overlap (0.0-0.5, default 0.1)
                BLANK_SKIP=0       Control blank frame skipping (default 0)
                NON_BLANK_SKIP=0   Control non-blank frame skipping (default 1)

            Examples:
                fluidaudio transcribe audio.wav                    # Streaming mode
                fluidaudio transcribe audio.wav --batch            # Batch mode with sliding window
                OVERLAP_RATIO=0.2 fluidaudio transcribe audio.wav --batch

            Streaming mode uses StreamingAsrManager which provides:
            - Automatic audio format conversion to 16kHz mono
            - Real-time transcription updates
            - Volatile (unconfirmed) and confirmed text states
            - AsyncStream-based API for easy integration
            """
        )
    }
}
#endif
