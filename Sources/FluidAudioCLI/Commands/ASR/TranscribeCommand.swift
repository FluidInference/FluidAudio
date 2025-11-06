#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Terminal color codes for ANSI output
enum TerminalColor {
    static let green = "\u{001B}[32m"  // Confirmed text
    static let purple = "\u{001B}[35m"  // Volatile text (magenta)
    static let reset = "\u{001B}[0m"  // Reset color

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["TERM"] != nil
    }
}

/// Command to transcribe audio files using batch or streaming mode
enum TranscribeCommand {
    private static let logger = AppLogger(category: "Transcribe")

    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var streamingMode = false
        var showMetadata = false
        var modelVersion: AsrModelVersion = .v3  // Default to v3
        var realtimeChunkMs: Int = 500  // Default 500ms chunks for realistic streaming

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--streaming":
                streamingMode = true
            case "--metadata":
                showMetadata = true
            case "--model-version":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "v2", "2":
                        modelVersion = .v2
                    case "v3", "3":
                        modelVersion = .v3
                    default:
                        logger.error("Invalid model version: \(arguments[i + 1]). Use 'v2' or 'v3'")
                        exit(1)
                    }
                    i += 1
                }
            case "--realtime-chunk-size":
                if i + 1 < arguments.count {
                    let sizeStr = arguments[i + 1].lowercased()
                    if let ms = Int(sizeStr.replacingOccurrences(of: "ms", with: "")) {
                        realtimeChunkMs = max(10, min(5000, ms))  // Clamp to 10ms-5000ms
                    } else {
                        logger.error("Invalid chunk size: \(arguments[i + 1]). Use format like '500ms'")
                        exit(1)
                    }
                    i += 1
                }
            default:
                logger.warning("Warning: Unknown option: \(arguments[i])")
            }
            i += 1
        }

        if streamingMode {
            logger.info(
                "Streaming mode enabled: simulating real-time audio with \(realtimeChunkMs)ms chunks.\n"
            )
            await testStreamingTranscription(
                audioFile: audioFile,
                showMetadata: showMetadata,
                modelVersion: modelVersion,
                realtimeChunkMs: realtimeChunkMs
            )
        } else {
            logger.info("Using batch mode with direct processing\n")
            await testBatchTranscription(audioFile: audioFile, showMetadata: showMetadata, modelVersion: modelVersion)
        }
    }

    /// Test batch transcription using AsrManager directly
    private static func testBatchTranscription(
        audioFile: String, showMetadata: Bool, modelVersion: AsrModelVersion
    ) async {
        do {
            // Initialize ASR models
            let models = try await AsrModels.downloadAndLoad(version: modelVersion)
            let asrManager = AsrManager(config: .default)
            try await asrManager.initialize(models: models)

            logger.info("ASR Manager initialized successfully")

            // Load audio file
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
            else {
                logger.error("Failed to create audio buffer")
                return
            }

            try audioFileHandle.read(into: buffer)

            // Convert audio to the format expected by ASR (16kHz mono Float array)
            let samples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Double(audioFileHandle.length) / format.sampleRate
            logger.info("Processing \(String(format: "%.2f", duration))s of audio (\(samples.count) samples)\n")

            // Process with ASR Manager
            logger.info("Transcribing file: \(audioFileURL) ...")
            let startTime = Date()
            let result = try await asrManager.transcribe(audioFileURL)
            let processingTime = Date().timeIntervalSince(startTime)

            // Print results
            logger.info("" + String(repeating: "=", count: 50))
            logger.info("BATCH TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(result.text)

            if showMetadata {
                logger.info("Metadata:")
                logger.info("  Confidence: \(String(format: "%.3f", result.confidence))")
                logger.info("  Duration: \(String(format: "%.3f", result.duration))s")
                if let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty {
                    let startTime = tokenTimings.first?.startTime ?? 0.0
                    let endTime = tokenTimings.last?.endTime ?? result.duration
                    logger.info("  Start time: \(String(format: "%.3f", startTime))s")
                    logger.info("  End time: \(String(format: "%.3f", endTime))s")
                    logger.info("Token Timings:")
                    for (index, timing) in tokenTimings.enumerated() {
                        logger.info(
                            "    [\(index)] '\(timing.token)' (id: \(timing.tokenId), start: \(String(format: "%.3f", timing.startTime))s, end: \(String(format: "%.3f", timing.endTime))s, conf: \(String(format: "%.3f", timing.confidence)))"
                        )
                    }
                } else {
                    logger.info("  Start time: 0.000s")
                    logger.info("  End time: \(String(format: "%.3f", result.duration))s")
                    logger.info("  Token timings: Not available")
                }
            }

            let rtfx = duration / processingTime

            logger.info("Performance:")
            logger.info("  Audio duration: \(String(format: "%.2f", duration))s")
            logger.info("  Processing time: \(String(format: "%.2f", processingTime))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")
            if !showMetadata {
                logger.info("  Confidence: \(String(format: "%.3f", result.confidence))")
            }

            // Cleanup
            asrManager.cleanup()

        } catch {
            logger.error("Batch transcription failed: \(error)")
        }
    }

    /// Test streaming transcription using StreamingAsrManager with LocalAgreement-2 validation
    private static func testStreamingTranscription(
        audioFile: String,
        showMetadata: Bool,
        modelVersion: AsrModelVersion,
        realtimeChunkMs: Int
    ) async {
        // Use optimized streaming configuration
        let config = StreamingAsrConfig.streaming

        // Create StreamingAsrManager
        let streamingAsr = StreamingAsrManager(config: config)

        do {
            // Initialize ASR models
            let models = try await AsrModels.downloadAndLoad(version: modelVersion)

            // Start the engine with the models
            try await streamingAsr.start(models: models)

            // Load audio file
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
            else {
                logger.error("Failed to create audio buffer")
                return
            }

            try audioFileHandle.read(into: buffer)

            // Calculate streaming parameters - use requested chunk size for simulation
            let chunkDurationSeconds = Double(realtimeChunkMs) / 1000.0
            let samplesPerChunk = Int(chunkDurationSeconds * format.sampleRate)
            let totalDuration = Double(audioFileHandle.length) / format.sampleRate

            // Track transcription updates
            var updateCount = 0

            // Listen for updates in real-time
            let updateTask = Task {
                let timestampFormatter: DateFormatter = {
                    let formatter = DateFormatter()
                    formatter.dateFormat = "HH:mm:ss.SSS"
                    return formatter
                }()

                for await update in await streamingAsr.transcriptionUpdates {
                    updateCount += 1

                    // Color-coded output: green = confirmed, purple = volatile
                    let color = update.isConfirmed ? TerminalColor.green : TerminalColor.purple
                    let coloredText =
                        TerminalColor.enabled ? "\(color)\(update.text)\(TerminalColor.reset)" : update.text

                    if showMetadata {
                        let timestampString = timestampFormatter.string(from: update.timestamp)
                        logger.info(
                            "\(coloredText) (conf: \(String(format: "%.3f", update.confidence)), timestamp: \(timestampString))"
                        )
                    } else {
                        logger.info(
                            "\(coloredText) (conf: \(String(format: "%.2f", update.confidence)))")
                    }
                }
            }

            // Stream audio chunks with real-time simulation
            var position = 0

            logger.info("Streaming audio with real-time simulation (\(realtimeChunkMs)ms chunks)...")
            logger.info("Waiting \(realtimeChunkMs)ms between chunks to simulate real-time audio arrival")
            logger.info("Purple text = volatile (awaiting validation), Green text = confirmed by LocalAgreement-2\n")

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

                position += chunkSize

                // Simulate real-time audio arrival by waiting chunk duration
                let chunkDurationNanoseconds = UInt64(chunkDurationSeconds * 1_000_000_000)
                try await Task.sleep(nanoseconds: chunkDurationNanoseconds)
            }

            // Allow brief time for final processing
            try await Task.sleep(nanoseconds: 500_000_000)  // 0.5 seconds

            // Finalize transcription
            let finalText = try await streamingAsr.finish()

            // Cancel update task
            updateTask.cancel()

            // Show final results with actual processing performance
            let processingTime = totalDuration  // StreamingAsrManager handles its own timing

            logger.info("" + String(repeating: "=", count: 50))
            logger.info("STREAMING TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(finalText)
            logger.info("Performance:")
            logger.info("  Audio duration: \(String(format: "%.2f", totalDuration))s")
            logger.info("  Updates emitted: \(updateCount)")

        } catch {
            logger.error("Streaming transcription failed: \(error)")
        }
    }

    private static func printUsage() {
        let logger = AppLogger(category: "Transcribe")
        logger.info(
            """

            Transcribe Command Usage:
                fluidaudio transcribe <audio_file> [options]

            Options:
                --help, -h              Show this help message
                --streaming             Use streaming mode with chunk simulation
                --metadata              Show confidence, start time, and end time in results
                --model-version <ver>   ASR model version to use: v2 or v3 (default: v3)
                --realtime-chunk-size   Size of chunks to simulate real-time streaming (default: 500ms)
                <size>                  Format: e.g., "500ms", "100ms", "2000ms" (range: 10ms-5000ms)

            Examples:
                fluidaudio transcribe audio.wav                           # Batch mode (default)
                fluidaudio transcribe audio.wav --streaming               # Streaming mode with 500ms chunks
                fluidaudio transcribe audio.wav --streaming --metadata    # Streaming with metadata
                fluidaudio transcribe audio.wav --streaming --realtime-chunk-size 100ms   # Small chunks (more realistic)
                fluidaudio transcribe audio.wav --streaming --realtime-chunk-size 2000ms  # Larger chunks

            Batch mode (default):
            - Direct processing using AsrManager for fastest results
            - Processes entire audio file at once

            Streaming mode:
            - Immediate chunk-by-chunk processing using VAD-style API
            - Processes chunks as they arrive without buffering (true streaming)
            - Shows incremental transcription updates using LocalAgreement-2 validation
            - Color-coded output: Purple = provisional (awaiting validation), Green = confirmed
            - Confirmed text grows as LocalAgreement-2 validates tokens across chunks
            - Stateful decoder maintains context across chunks for better continuity
            - Default 500ms chunks simulate realistic microphone input at real-time speed

            Realtime chunk size:
            - Simulates how audio arrives from a microphone (e.g., 500ms at a time)
            - Smaller chunks (100-300ms) more closely simulate real microphones
            - Larger chunks (1000-5000ms) reduce processing frequency
            - Each chunk waits for its duration before the next arrives (e.g., 500ms wait for 500ms chunk)

            Metadata option:
            - Shows confidence score for transcription accuracy
            - Batch mode: Shows duration and token-based start/end times (if available)
            - Streaming mode: Shows timestamps for each transcription update
            - Works with both batch and streaming modes
            """
        )
    }
}
#endif
