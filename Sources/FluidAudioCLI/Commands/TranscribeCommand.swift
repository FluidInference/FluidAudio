#if os(macOS)
import AVFoundation
import CoreMedia
import FluidAudio
import Foundation

/// Debug utility for tracking transcription metrics and performance
///
/// **Note**: This is only used by the CLI for debugging and metrics collection.
/// For UI applications, use `StreamingAsrManager.snapshots` or `StreamingAsrManager.results` directly.
///
/// Example:
/// ```swift
/// let tracker = TranscriptionTracker()
///
/// // Track metrics during streaming
/// await tracker.addVolatileUpdate("Partial text...")
/// await tracker.addFinalizedUpdate("Final text")
/// await tracker.updateAudioPosition(15.3) // seconds
///
/// // Get performance stats
/// let stats = await tracker.getPerformanceStats()
/// print("Processed \(stats.volatileCount) volatile updates in \(stats.elapsedTime)s")
/// ```
@available(macOS 13.0, *)
actor TranscriptionTracker {
    private var volatileUpdates: [String] = []
    private var finalizedUpdates: [String] = []
    private var currentAudioPosition: Double = 0.0
    private let startTime: Date

    init() {
        self.startTime = Date()
    }

    func addVolatileUpdate(_ text: String) {
        volatileUpdates.append(text)
    }

    func addFinalizedUpdate(_ text: String) {
        finalizedUpdates.append(text)
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

    func getFinalizedCount() -> Int {
        return finalizedUpdates.count
    }

    /// Get comprehensive performance statistics
    func getPerformanceStats() -> PerformanceStats {
        return PerformanceStats(
            volatileCount: volatileUpdates.count,
            finalizedCount: finalizedUpdates.count,
            elapsedTime: Date().timeIntervalSince(startTime),
            audioPosition: currentAudioPosition
        )
    }
}

/// Performance statistics from transcription tracking
struct PerformanceStats {
    let volatileCount: Int
    let finalizedCount: Int
    let elapsedTime: TimeInterval
    let audioPosition: TimeInterval
}

/// Thread-safe file logger for streaming events
@available(macOS 13.0, *)
actor StreamingLogWriter {
    private var handle: FileHandle?
    private(set) var url: URL

    init(fileName: String) {
        let cwd = FileManager.default.currentDirectoryPath
        self.url = URL(fileURLWithPath: cwd).appendingPathComponent(fileName)
    }

    func open(with header: String) async {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        do {
            handle = try FileHandle(forWritingTo: url)
            if let data = (header + "\n").data(using: .utf8) {
                try handle?.write(contentsOf: data)
            }
        } catch {
            Swift.print("⚠️ Failed to open log file: \(error)")
        }
    }

    func write(_ line: String) async {
        guard let h = handle else { return }
        do {
            if let data = (line + "\n").data(using: .utf8) {
                try h.write(contentsOf: data)
            }
        } catch {}
    }

    func close() async {
        do {
            try handle?.synchronize()
            try handle?.close()
        } catch {}
        handle = nil
    }
}

/// Command to transcribe audio files using batch or streaming mode
@available(macOS 13.0, *)
enum TranscribeCommand {

    static func run(arguments: [String]) async {
        // Parse arguments - microphone mode doesn't require file
        var audioFile: String? = nil
        var streamingMode = false
        var microphoneMode = false

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
            await testStreamingTranscription(audioFile: audioFile!)
        } else {
            print("Using batch mode with direct processing\n")
            await testBatchTranscription(audioFile: audioFile!)
        }
    }

    /// Test batch transcription using AsrManager directly
    private static func testBatchTranscription(audioFile: String) async {
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

            let rtfx = duration / processingTime

            print("\nPerformance:")
            print("  Audio duration: \(String(format: "%.2f", duration))s")
            print("  Processing time: \(String(format: "%.2f", processingTime))s")
            print("  RTFx: \(String(format: "%.2f", rtfx))x")
            print("  Confidence: \(String(format: "%.3f", result.confidence))")

        } catch {
            print("Batch transcription failed: \\(error)")
        }
    }

    /// Test streaming transcription
    private static func testStreamingTranscription(audioFile: String) async {

        // Create StreamingAsrManager
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

            // Calculate streaming parameters - align with StreamingAsrConfig chunk size
            let chunkDuration = StreamingAsrConfig.default.mode.chunkSeconds  // Use same chunk size as streaming config
            let samplesPerChunk = Int(chunkDuration * format.sampleRate)
            let totalFrames = Int(buffer.frameLength)
            let totalChunks = Int(ceil(Double(totalFrames) / Double(samplesPerChunk)))
            let totalDuration = Double(audioFileHandle.length) / format.sampleRate

            // Initialize UI
            let streamingUI = StreamingUI()
            await streamingUI.start(audioDuration: totalDuration, totalChunks: totalChunks)

            // Track transcription updates
            let tracker = TranscriptionTracker()
            // Logging setup
            let iso = ISO8601DateFormatter()
            iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            let ts = iso.string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let logName = "asr_stream_\(ts).log"
            let logger = StreamingLogWriter(fileName: logName)
            await logger.open(
                with: "# FluidAudio Streaming Log (volatile + final)\n# Created: \(iso.string(from: Date()))")
            var chunksProcessed = 0

            // Listen for snapshot updates in real-time (finalized + volatile)
            let updateTask = Task {
                for await snapshot in await streamingAsr.snapshots {
                    let finalized = String(snapshot.finalized.characters)
                    let volatile = snapshot.volatile.map { String($0.characters) } ?? ""
                    await streamingUI.updateTranscription(finalized: finalized, volatile: volatile)
                    if !volatile.isEmpty {
                        await tracker.addVolatileUpdate(volatile)
                    }
                    if !finalized.isEmpty {
                        await tracker.addFinalizedUpdate(finalized)
                    }
                }
            }

            // Log precise volatile/final events from segment results
            let loggingTask = Task {
                for await result in await streamingAsr.results {
                    let now = iso.string(from: Date())
                    let kind = result.isFinal ? "FINAL" : "VOLATILE"
                    let start = result.audioTimeRange.start.seconds
                    let dur = result.audioTimeRange.duration.seconds
                    let conf = String(format: "%.3f", result.confidence)
                    let text = String(result.attributedText.characters)
                    let line =
                        "[\(now)] [\(kind)] seg=\(result.segmentID.uuidString.prefix(8)) rev=\(result.revision) t=\(String(format: "%.2f", start)) dur=\(String(format: "%.2f", dur)) conf=\(conf) text=\(text)"
                    await logger.write(line)
                }
            }

            var position = 0
            let startTime = Date()

            while position < Int(buffer.frameLength) {
                let remainingSamples = Int(buffer.frameLength) - position
                let chunkSize = min(samplesPerChunk, remainingSamples)

                guard
                    let chunkBuffer: AVAudioPCMBuffer = AVAudioPCMBuffer(
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
                try await Task.sleep(nanoseconds: 100_000_000)  // 100ms yield
                await Task.yield()

            }

            // Allow brief time for final processing
            try await Task.sleep(nanoseconds: 1000_000_000)  // 1 second delay

            // Finalize transcription
            let finalText = try await streamingAsr.finish()

            // Cancel background tasks
            updateTask.cancel()
            loggingTask.cancel()

            // Show final results with actual processing performance
            let processingTime = await tracker.getElapsedProcessingTime()
            await streamingUI.showFinalResults(finalText: finalText, totalTime: processingTime)
            await streamingUI.finish()
            await logger.write("# Final transcription:\n\(finalText)")
            await logger.close()

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

            // Initialize UI for real-time transcription
            let streamingUI = StreamingUI()
            await streamingUI.showInitialization()

            // Show that we're doing live transcription (set duration to -1 to indicate microphone mode)
            await streamingUI.start(audioDuration: -1, totalChunks: 0)

            // Track transcription updates
            let tracker = TranscriptionTracker()

            // Logging setup
            let iso = ISO8601DateFormatter()
            iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            let ts = iso.string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let logName = "asr_microphone_\(ts).log"
            let logger = StreamingLogWriter(fileName: logName)
            await logger.open(
                with: "# FluidAudio Real-time Microphone Log\n# Created: \(iso.string(from: Date()))")

            // Listen for snapshot updates in real-time (finalized + volatile)
            let updateTask = Task {
                for await snapshot in await streamingAsr.snapshots {
                    let finalized = String(snapshot.finalized.characters)
                    let volatile = snapshot.volatile.map { String($0.characters) } ?? ""
                    await streamingUI.updateTranscription(finalized: finalized, volatile: volatile)
                    if !volatile.isEmpty {
                        await tracker.addVolatileUpdate(volatile)
                    }
                    if !finalized.isEmpty {
                        await tracker.addFinalizedUpdate(finalized)
                    }
                }
            }

            // Log precise volatile/final events from segment results
            let loggingTask = Task {
                for await result in await streamingAsr.results {
                    let now = iso.string(from: Date())
                    let kind = result.isFinal ? "FINAL" : "VOLATILE"
                    let start = result.audioTimeRange.start.seconds
                    let dur = result.audioTimeRange.duration.seconds
                    let conf = String(format: "%.3f", result.confidence)
                    let text = String(result.attributedText.characters)
                    let line =
                        "[\(now)] [\(kind)] seg=\(result.segmentID.uuidString.prefix(8)) rev=\(result.revision) t=\(String(format: "%.2f", start)) dur=\(String(format: "%.2f", dur)) conf=\(conf) text=\(text)"
                    await logger.write(line)
                }
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

            let startTime = Date()
            var lastUpdateTime = Date()

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
                let elapsedTime = Date().timeIntervalSince(startTime)

                // Update elapsed time display every second
                if Date().timeIntervalSince(lastUpdateTime) >= 1.0 {
                    await streamingUI.updateProgress(chunksProcessed: 0, elapsedTime: elapsedTime)
                    lastUpdateTime = Date()
                }

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

            // Finalize transcription
            let finalText = try await streamingAsr.finish()

            // Cancel background tasks
            updateTask.cancel()
            loggingTask.cancel()

            // Show final results
            let processingTime = await tracker.getElapsedProcessingTime()
            await streamingUI.showFinalResults(finalText: finalText, totalTime: processingTime)
            await streamingUI.finish()
            await logger.write("# Final transcription:\n\(finalText)")
            await logger.close()

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

            Examples:
                fluidaudio transcribe audio.wav                    # Batch mode (default)
                fluidaudio transcribe audio.wav --streaming        # Streaming mode
                fluidaudio transcribe --microphone                 # Real-time microphone

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
            """
        )
    }
}
#endif
