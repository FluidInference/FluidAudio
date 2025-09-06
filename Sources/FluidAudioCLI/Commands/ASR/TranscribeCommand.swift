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

/// Command to transcribe audio files using batch or streaming mode
@available(macOS 13.0, *)
enum TranscribeCommand {

    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            print("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var streamingMode = false
        var showMetadata = false

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
            default:
                print("Warning: Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("Audio Transcription")
        print("===================\n")

        // Test loading audio at different sample rates
        await testAudioConversion(audioFile: audioFile)

        if streamingMode {
            print(
                "Streaming mode enabled: simulating real-time audio with 1-second chunks.\n"
            )
            await testStreamingTranscription(audioFile: audioFile, showMetadata: showMetadata)
        } else {
            print("Using batch mode with direct processing\n")
            await testBatchTranscription(audioFile: audioFile, showMetadata: showMetadata)
        }
    }

    /// Test audio conversion capabilities
    private static func testAudioConversion(audioFile: String) async {
        print("Testing Audio Conversion")
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

    /// Test batch transcription using AsrManager directly
    private static func testBatchTranscription(audioFile: String, showMetadata: Bool) async {
        print("Testing Batch Transcription")
        print("------------------------------")

        do {
            // Initialize ASR models
            let models = try await AsrModels.downloadAndLoad()
            let asrManager = AsrManager(config: .default)
            try await asrManager.initialize(models: models)

            print("ASR Manager initialized successfully")

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

            // Convert audio to the format expected by ASR (16kHz mono Float array)
            let samples = try await AudioProcessor.loadAudioFile(path: audioFile)

            let duration = Double(audioFileHandle.length) / format.sampleRate
            print("Processing \(String(format: "%.2f", duration))s of audio (\(samples.count) samples)\n")

            // Process with ASR Manager
            let startTime = Date()
            let result = try await asrManager.transcribe(samples, source: .system)
            let processingTime = Date().timeIntervalSince(startTime)

            // Print results
            print("\n" + String(repeating: "=", count: 50))
            print("BATCH TRANSCRIPTION RESULTS")
            print(String(repeating: "=", count: 50))
            print("\nFinal transcription:")
            print(result.text)

            if showMetadata {
                print("\nMetadata:")
                print("  Confidence: \(String(format: "%.3f", result.confidence))")
                print("  Duration: \(String(format: "%.3f", result.duration))s")
                if let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty {
                    let startTime = tokenTimings.first?.startTime ?? 0.0
                    let endTime = tokenTimings.last?.endTime ?? result.duration
                    print("  Start time: \(String(format: "%.3f", startTime))s")
                    print("  End time: \(String(format: "%.3f", endTime))s")
                    print("\nToken Timings:")
                    for (index, timing) in tokenTimings.enumerated() {
                        print(
                            "    [\(index)] '\(timing.token)' (id: \(timing.tokenId), start: \(String(format: "%.3f", timing.startTime))s, end: \(String(format: "%.3f", timing.endTime))s, conf: \(String(format: "%.3f", timing.confidence)))"
                        )
                    }
                } else {
                    print("  Start time: 0.000s")
                    print("  End time: \(String(format: "%.3f", result.duration))s")
                    print("  Token timings: Not available")
                }
            }

            let rtfx = duration / processingTime

            print("\nPerformance:")
            print("  Audio duration: \(String(format: "%.2f", duration))s")
            print("  Processing time: \(String(format: "%.2f", processingTime))s")
            print("  RTFx: \(String(format: "%.2f", rtfx))x")
            if !showMetadata {
                print("  Confidence: \(String(format: "%.3f", result.confidence))")
            }

            // Cleanup
            asrManager.cleanup()

        } catch {
            print("Batch transcription failed: \\(error)")
        }
    }

    /// Test streaming transcription
    private static func testStreamingTranscription(audioFile: String, showMetadata: Bool) async {
        // Use optimized streaming configuration
        let config = StreamingAsrConfig.streaming

        // Create StreamingAsrManager
        let streamingAsr = StreamingAsrManager(config: config)

        do {
            // Start the engine
            try await streamingAsr.start()

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

            // Calculate streaming parameters - align with StreamingAsrConfig chunk size
            let chunkDuration = config.chunkSeconds  // Use same chunk size as streaming config
            let samplesPerChunk = Int(chunkDuration * format.sampleRate)
            let totalFrames = Int(buffer.frameLength)
            let totalChunks = Int(ceil(Double(totalFrames) / Double(samplesPerChunk)))
            let totalDuration = Double(audioFileHandle.length) / format.sampleRate

            // Initialize UI
            let streamingUI = StreamingUI()
            await streamingUI.start(audioDuration: totalDuration, totalChunks: totalChunks)

            // Track transcription updates
            let tracker = TranscriptionTracker()
            var chunksProcessed = 0

            // Listen for updates in real-time
            let updateTask = Task {
                for await update in await streamingAsr.transcriptionUpdates {
                    // Debug: show transcription updates
                    let updateType = update.isConfirmed ? "CONFIRMED" : "VOLATILE"
                    if showMetadata {
                        let formatter = DateFormatter()
                        formatter.dateFormat = "HH:mm:ss.SSS"
                        let timestampString = formatter.string(from: update.timestamp)
                        print(
                            "[\(updateType)] '\(update.text)' (conf: \(String(format: "%.3f", update.confidence)), timestamp: \(timestampString))"
                        )
                    } else {
                        print("[\(updateType)] '\(update.text)' (conf: \(String(format: "%.2f", update.confidence)))")
                    }

                    if update.isConfirmed {
                        await streamingUI.addConfirmedUpdate(update.text)
                        await tracker.addConfirmedUpdate(update.text)
                    } else {
                        await streamingUI.updateVolatileText(update.text)
                        await tracker.addVolatileUpdate(update.text)
                    }
                }
            }

            // Stream audio chunks continuously - no artificial delays
            var position = 0
            let startTime = Date()

            print("Streaming audio continuously (no artificial delays)...")
            print(
                "Using \(String(format: "%.1f", chunkDuration))s chunks with \(String(format: "%.1f", config.leftContextSeconds))s left context, \(String(format: "%.1f", config.rightContextSeconds))s right context"
            )
            print("Watch for real-time hypothesis updates being replaced by confirmed text\n")

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

                // Stream the chunk immediately - no waiting
                await streamingAsr.streamAudio(chunkBuffer)

                // Update progress with actual processing time
                chunksProcessed += 1
                let elapsedTime = Date().timeIntervalSince(startTime)
                await streamingUI.updateProgress(chunksProcessed: chunksProcessed, elapsedTime: elapsedTime)

                position += chunkSize

                // Small yield to allow UI updates to show
                await Task.yield()
            }

            // Allow brief time for final processing
            try await Task.sleep(nanoseconds: 500_000_000)  // 0.5 seconds

            // Finalize transcription
            let finalText = try await streamingAsr.finish()

            // Cancel update task
            updateTask.cancel()

            // Show final results with actual processing performance
            let processingTime = await tracker.getElapsedProcessingTime()
            await streamingUI.showFinalResults(finalText: finalText, totalTime: processingTime)
            await streamingUI.finish()

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
                --streaming        Use streaming mode with chunk simulation
                --metadata         Show confidence, start time, and end time in results

            Examples:
                fluidaudio transcribe audio.wav                    # Batch mode (default)
                fluidaudio transcribe audio.wav --streaming        # Streaming mode
                fluidaudio transcribe audio.wav --metadata         # Batch mode with metadata
                fluidaudio transcribe audio.wav --streaming --metadata # Streaming mode with metadata

            Batch mode (default):
            - Direct processing using AsrManager for fastest results
            - Processes entire audio file at once

            Streaming mode:
            - Simulates real-time streaming with chunk processing
            - Shows incremental transcription updates
            - Uses StreamingAsrManager with sliding window processing

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
