#if os(macOS)
import AVFoundation
import CoreMedia
import FluidAudio
import Foundation

/// Command to transcribe audio files using batch or streaming mode
@available(macOS 13.0, *)
enum TranscribeCommand {
    /// Minimal ANSI color helper (avoid TerminalUI dependency)
    private static func green(_ s: String) -> String { "\u{001B}[32m\(s)\u{001B}[0m" }

    static func run(arguments: [String]) async {
        // Parse arguments - microphone mode doesn't require file
        var audioFile: String? = nil
        var streamingMode = false
        var microphoneMode = false
        var showMetadata = false

        // Parse options
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--streaming":
                streamingMode = true
            case "--microphone":
                microphoneMode = true
            case "--metadata":
                showMetadata = true
            default:
                // If not a flag, treat as audio file
                if !arguments[i].hasPrefix("--") {
                    audioFile = arguments[i]
                } else {
                    print("Warning: Unknown option: \(arguments[i])")
                }
            }
            i += 1
        }

        // Validate arguments
        if microphoneMode {
            if let file = audioFile {
                print("Error: Cannot specify both microphone mode and audio file: \(file)")
                printUsage()
                exit(1)
            }
        } else {
            guard let file = audioFile else {
                print("No audio file specified")
                printUsage()
                exit(1)
            }
            audioFile = file
        }

        print("Audio Transcription")
        print("===================\n")

        if microphoneMode {
            print("Real-time microphone transcription enabled.\n")
            await testMicrophoneTranscription()
        } else if streamingMode {
            print("Streaming mode enabled: volatile and finalized updates.\n")
            await testStreamingTranscription(audioFile: audioFile!, showMetadata: showMetadata)
        } else {
            print("Using batch mode with direct processing\n")
            await testBatchTranscription(audioFile: audioFile!, showMetadata: showMetadata)
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

        } catch {
            print("Batch transcription failed: \\(error)")
        }
    }

    /// Test streaming transcription
    private static func testStreamingTranscription(audioFile: String, showMetadata: Bool) async {
        // Simple, debuggable streaming output (no box UI)
        print("Preparing streaming transcription...")
        let streamingAsr = StreamingAsrManager()

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

            // Set up timestamped logging with accumulation
            let timeFormatter = DateFormatter()
            timeFormatter.dateFormat = "HH:mm:ss.SSS"

            // Accumulate finalized text; show a single evolving hypothesis
            let updateTask: Task<String, Never> = Task {
                var finalized = ""
                for await segmentUpdate in await streamingAsr.segmentUpdates {
                    let ts = timeFormatter.string(from: segmentUpdate.timestamp)
                    let text = segmentUpdate.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if text.isEmpty { continue }

                    if segmentUpdate.isVolatile {
                        // Show consolidated line: finalized + current hypothesis (green)
                        if finalized.isEmpty {
                            print("[\(ts)] \(Self.green(text))")
                        } else {
                            print("[\(ts)] \(finalized) \(Self.green(text))")
                        }
                    } else {
                        // Append to finalized and print current accumulation
                        if finalized.isEmpty {
                            finalized = text
                        } else {
                            finalized += " " + text
                        }
                        print("[\(ts)] \(finalized)")
                    }
                }
                return finalized
            }

            // Stream audio in real-time-like chunks
            var position = 0

            // Feed using Apple's typical audio engine default buffer size (frames)
            // On Apple platforms, the default I/O buffer is commonly 1024 frames.
            // Using this for file-streaming keeps behavior aligned with live engine taps.
            let samplesPerFeedChunk = 1024

            while position < Int(buffer.frameLength) {
                let remainingSamples = Int(buffer.frameLength) - position
                let chunkSize = min(samplesPerFeedChunk, remainingSamples)

                guard
                    let chunkBuffer: AVAudioPCMBuffer = AVAudioPCMBuffer(
                        pcmFormat: format,
                        frameCapacity: AVAudioFrameCount(chunkSize)
                    )
                else { break }

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

                await streamingAsr.streamAudio(chunkBuffer)

                position += chunkSize
            }

            // Stop streaming and wait for updates stream to close
            try await streamingAsr.stop()
            let finalText = await updateTask.value

            // Print final transcript summary
            print("\n" + String(repeating: "=", count: 50))
            print(String(repeating: "=", count: 50))
            print("Final transcription:")
            print(finalText)

        } catch {
            print("Streaming transcription failed: \(error)")
        }
    }

    /// Test real-time microphone transcription
    private static func testMicrophoneTranscription() async {

        // Create audio engine for microphone input
        let audioEngine = AVAudioEngine()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Create StreamingAsrManager for microphone source
        let streamingAsr = StreamingAsrManager()

        do {
            // Start the ASR engine with microphone source
            try await streamingAsr.start(source: .microphone)

            // Simple timestamped output with accumulation
            let timeFormatter = DateFormatter()
            timeFormatter.dateFormat = "HH:mm:ss.SSS"

            print("🎙️ Real-time microphone transcription - Press Enter to stop\n")

            // Listen for segment updates; show finalized prefix + evolving hypothesis
            let updateTask: Task<String, Never> = Task {
                var finalized = ""
                for await segmentUpdate in await streamingAsr.segmentUpdates {
                    let ts = timeFormatter.string(from: segmentUpdate.timestamp)
                    let text = segmentUpdate.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if text.isEmpty { continue }

                    if segmentUpdate.isVolatile {
                        if finalized.isEmpty {
                            print("[\(ts)] \(Self.green(text))")
                        } else {
                            print("[\(ts)] \(finalized) \(Self.green(text))")
                        }
                    } else {
                        if finalized.isEmpty {
                            finalized = text
                        } else {
                            finalized += " " + text
                        }
                        print("[\(ts)] \(finalized)")
                    }
                }
                return finalized
            }

            // Install tap on the input node to capture microphone audio
            inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { (buffer, time) in
                // Stream the audio buffer to ASR manager
                Task {
                    await streamingAsr.streamAudio(buffer)
                }
            }

            // Start the audio engine
            try audioEngine.start()

            _ = Date()

            // Use a shared actor to track completion state
            actor CompletionTracker {
                private var completed = false

                func markCompleted() {
                    completed = true
                }

                func isCompleted() -> Bool {
                    return completed
                }
            }

            let completionTracker = CompletionTracker()

            // Create a task to handle user input
            let inputTask = Task {
                // Wait for user to press Enter
                _ = readLine()
                print("\n🛑 Stopping microphone transcription...")
                await completionTracker.markCompleted()
            }

            // Keep running until user presses Enter
            while !(await completionTracker.isCompleted()) {
                // Check for completion (non-blocking)
                try await Task.sleep(nanoseconds: 100_000_000)  // 100ms
            }

            // Cancel the input task if still running
            inputTask.cancel()

            // Stop audio engine
            audioEngine.stop()
            inputNode.removeTap(onBus: 0)

            print("\n📝 Finalizing transcription...")

            // Allow brief time for final processing
            try await Task.sleep(nanoseconds: 1000_000_000)  // 1 second delay

            // Stop streaming and wait for updates stream to close
            try await streamingAsr.stop()
            let finalText = await updateTask.value

            // Show final results accumulated from updates
            print("\n" + String(repeating: "=", count: 50))
            print(String(repeating: "=", count: 50))
            print("Final transcription:")
            print(finalText)

        } catch {
            print("Microphone transcription failed: \(error)")
        }
    }

    private static func printUsage() {
        print(
            """

            Transcribe Command Usage:
                fluidaudio transcribe <audio_file> [options]
                fluidaudio transcribe --microphone

            Options:
                --help, -h         Show this help message
                --streaming        Use streaming mode with chunk simulation
                --microphone       Real-time microphone transcription
                --metadata         Show confidence, start time, and end time in results

            Examples:
                fluidaudio transcribe audio.wav                    # Batch mode (default)
                fluidaudio transcribe audio.wav --streaming        # Streaming mode
                fluidaudio transcribe --microphone                 # Real-time microphone
                fluidaudio transcribe audio.wav --metadata         # Batch mode with metadata
                fluidaudio transcribe audio.wav --streaming --metadata # Streaming mode with metadata

            Batch mode (default):
            - Direct processing using AsrManager for fastest results
            - Processes entire audio file at once

            Streaming mode:
            - Simulates real-time streaming with chunk processing
            - Shows incremental transcription updates
            - Uses StreamingAsrManager with sliding window processing

            Microphone mode:
            - Real-time transcription from microphone input
            - Shows live volatile and finalized text updates
            - Uses StreamingAsrManager with AVAudioEngine
            - Press Enter to stop and get final results

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
