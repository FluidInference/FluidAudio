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

    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            print("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var configType = "default"

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
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

        // Test loading audio at different sample rates
        await testAudioConversion(audioFile: audioFile)

        // Test transcription with StreamingAsrManager
        await testStreamingTranscription(
            audioFile: audioFile,
            configType: configType
        )
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

            // Create streaming display
            let display = StreamingDisplay()
            await display.start(confidenceThreshold: config.confirmationThreshold)

            // Listen for updates
            let updateTask = Task {
                for await update in await streamingAsr.transcriptionUpdates {
                    // Update display with new transcription data
                    if update.isConfirmed {
                        await tracker.addConfirmedUpdate(update.text)
                        await display.update(
                            confirmed: await streamingAsr.confirmedTranscript,
                            volatile: await streamingAsr.volatileTranscript,
                            confidence: update.confidence,
                            isNewChunk: true
                        )
                    } else {
                        await tracker.addVolatileUpdate(update.text)
                        await display.update(
                            confirmed: await streamingAsr.confirmedTranscript,
                            volatile: await streamingAsr.volatileTranscript,
                            confidence: update.confidence,
                            isNewChunk: false
                        )
                    }

                    // Update statistics
                    let processingTime = Date().timeIntervalSince(startTime)
                    let audioDuration = Double(audioFileHandle.length) / format.sampleRate
                    await display.updateStatistics(
                        processingTime: processingTime,
                        audioLength: audioDuration
                    )
                }
            }

            // Simulate streaming by feeding audio in chunks
            let chunkDuration = 0.1  // 100ms chunks
            let samplesPerChunk = Int(chunkDuration * format.sampleRate)
            var position = 0

            // Streaming audio (display handles the UI)
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

            // Finalize
            let finalText = try await streamingAsr.finish()

            // Cancel update task but keep display active for final view
            updateTask.cancel()

            // Update display one last time with final state
            await display.update(
                confirmed: await streamingAsr.confirmedTranscript,
                volatile: await streamingAsr.volatileTranscript,
                confidence: 1.0,  // Show as fully confident for final
                isNewChunk: false
            )

            let processingTime = Date().timeIntervalSince(startTime)
            let audioDuration = Double(audioFileHandle.length) / format.sampleRate
            await display.updateStatistics(
                processingTime: processingTime,
                audioLength: audioDuration
            )

            // Show completion message in display
            await display.showCompletion()

            // Wait for user input before showing final results
            print(
                "\n"
                    + TerminalUI.colored(
                        "✅ Transcription Complete! Press Enter to view full results...",
                        color: TerminalUI.Color.brightGreen))
            _ = readLine()

            // Now stop display and show full results
            await display.stop()

            // Clear screen for final results
            TerminalUI.clearScreen()

            // Print final results with scrollable output
            print(String(repeating: "═", count: 80))
            print("📝 FINAL TRANSCRIPTION RESULTS")
            print(String(repeating: "═", count: 80))

            print("\n" + TerminalUI.colored("Complete Transcription:", color: TerminalUI.Color.bold))
            print(String(repeating: "─", count: 80))

            // Word wrap the final text for better readability
            let wrappedLines = TerminalUI.wordWrap(text: finalText, width: 78)
            for line in wrappedLines {
                print(TerminalUI.colored(line, color: TerminalUI.Color.brightWhite))
            }

            print("\n" + String(repeating: "─", count: 80))
            print("\n" + TerminalUI.colored("Statistics:", color: TerminalUI.Color.bold))
            print("  Total volatile updates: \(await tracker.getVolatileCount())")
            print("  Total confirmed updates: \(await tracker.getConfirmedCount())")

            let confirmedText = await streamingAsr.confirmedTranscript
            let volatileText = await streamingAsr.volatileTranscript
            let wordCount = finalText.split(separator: " ").count

            print("  " + TerminalUI.colored("Total words: \(wordCount)", color: TerminalUI.Color.cyan))
            if !confirmedText.isEmpty {
                print(
                    "  "
                        + TerminalUI.colored(
                            "Final confirmed: \(confirmedText.split(separator: " ").count) words",
                            color: TerminalUI.Color.green))
            }
            if !volatileText.isEmpty {
                print(
                    "  "
                        + TerminalUI.colored(
                            "Final volatile: \(volatileText.split(separator: " ").count) words",
                            color: TerminalUI.Color.yellow))
            }

            let rtfx = audioDuration / processingTime

            print("\n" + TerminalUI.colored("Performance:", color: TerminalUI.Color.bold))
            print("  Audio duration: \(String(format: "%.2f", audioDuration))s")
            print("  Processing time: \(String(format: "%.2f", processingTime))s")
            print(
                "  "
                    + TerminalUI.colored(
                        "RTFx: \(String(format: "%.2fx", rtfx))",
                        color: rtfx > 10 ? TerminalUI.Color.green : TerminalUI.Color.yellow))

            print("\n" + TerminalUI.colored("Press Enter to exit...", color: TerminalUI.Color.dim))
            _ = readLine()

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
                --config <type>    Configuration type: default, low-latency, high-accuracy
                --help, -h         Show this help message

            Examples:
                fluidaudio transcribe audio.wav
                fluidaudio transcribe audio.wav --config low-latency

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
