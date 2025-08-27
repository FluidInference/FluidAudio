#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
struct DiarizedTranscribeCommand {

    private static let logger = Logger(
        subsystem: "com.fluidinfluence.cli",
        category: "DiarizedTranscribe"
    )

    enum ConfigPreset: String {
        case `default`
        case accurate
    }

    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            print("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var configPreset = ConfigPreset.default
        var debug = false
        var streaming = false
        var alignmentTolerance: Double = 0.5
        var output: String?

        // Parse optional arguments
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--config":
                if i + 1 < arguments.count {
                    if let preset = ConfigPreset(rawValue: arguments[i + 1]) {
                        configPreset = preset
                    } else {
                        print("Invalid config preset: \(arguments[i + 1])")
                        exit(1)
                    }
                    i += 2
                } else {
                    print("--config requires a value")
                    exit(1)
                }
            case "--debug", "-d":
                debug = true
                i += 1
            case "--streaming":
                streaming = true
                i += 1
            case "--alignment-tolerance":
                if i + 1 < arguments.count {
                    if let tolerance = Double(arguments[i + 1]) {
                        alignmentTolerance = tolerance
                    } else {
                        print("Invalid alignment tolerance: \(arguments[i + 1])")
                        exit(1)
                    }
                    i += 2
                } else {
                    print("--alignment-tolerance requires a value")
                    exit(1)
                }
            case "--output", "-o":
                if i + 1 < arguments.count {
                    output = arguments[i + 1]
                    i += 2
                } else {
                    print("--output requires a value")
                    exit(1)
                }
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                print("Unknown option: \(arguments[i])")
                printUsage()
                exit(1)
            }
        }

        do {
            try await runDiarizedTranscription(
                inputFile: audioFile,
                config: configPreset,
                debug: debug,
                streaming: streaming,
                alignmentTolerance: alignmentTolerance,
                output: output
            )
        } catch {
            print("Error: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func printUsage() {
        print(
            """
            FluidAudio Diarized Transcribe

            Usage: fluidaudio diarized-transcribe <audio-file> [options]

            Options:
                --config <preset>           Configuration preset: 'default' or 'accurate' (default: default)
                --debug, -d                 Enable debug logging
                --streaming                 Enable real-time streaming mode (for testing)
                --alignment-tolerance <sec> Alignment tolerance in seconds (default: 0.5)
                --output, -o <file>         Output file for results (JSON format)
                --help, -h                  Show this help message

            Examples:
                fluidaudio diarized-transcribe meeting.wav
                fluidaudio diarized-transcribe interview.wav --config accurate --debug
                fluidaudio diarized-transcribe podcast.wav --output results.json
            """)
    }

    private static func runDiarizedTranscription(
        inputFile: String,
        config: ConfigPreset,
        debug: Bool,
        streaming: Bool,
        alignmentTolerance: Double,
        output: String?
    ) async throws {
        logger.info("Starting diarized transcription of: \(inputFile)")

        // Validate input file
        let inputURL = URL(fileURLWithPath: inputFile)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw RuntimeError("Input file does not exist: \(inputFile)")
        }

        // Create configuration with debug enabled if requested
        let diarizedConfig = createConfiguration(
            preset: config,
            debug: debug,
            alignmentTolerance: alignmentTolerance
        )

        if debug {
            print("Configuration:")
            print("  - Config preset: \(config)")
            print("  - Alignment tolerance: \(alignmentTolerance)s")
            print("  - Streaming mode: \(streaming)")
            print("  - Debug logging: \(debug)")
            print()
        }

        if streaming {
            try await runStreamingMode(inputURL: inputURL, config: diarizedConfig, debug: debug, output: output)
        } else {
            try await runBatchMode(inputURL: inputURL, config: diarizedConfig, debug: debug, output: output)
        }
    }

    private static func createConfiguration(
        preset: ConfigPreset,
        debug: Bool,
        alignmentTolerance: Double
    ) -> StreamingDiarizedAsrConfig {
        let asrConfig: StreamingAsrConfig
        let diarizerConfig: DiarizerConfig

        switch preset {
        case .default:
            asrConfig = .realtime
            diarizerConfig = .default
        case .accurate:
            asrConfig = .default
            diarizerConfig = DiarizerConfig(
                clusteringThreshold: 0.7,
                minSpeechDuration: 0.5,
                minEmbeddingUpdateDuration: 1.0,
                debugMode: debug,
                chunkDuration: 10.0,
                chunkOverlap: 1.0
            )
        }

        return StreamingDiarizedAsrConfig(
            asrConfig: asrConfig,
            diarizerConfig: diarizerConfig,
            enableDebug: debug,
            alignmentTolerance: alignmentTolerance
        )
    }

    private static func runBatchMode(
        inputURL: URL, config: StreamingDiarizedAsrConfig, debug: Bool, output: String?
    ) async throws {
        print("Processing audio file: \(inputURL.lastPathComponent)")
        print("Mode: Real-time Streaming")
        print()

        // Load audio file
        guard let audioFile = try? AVAudioFile(forReading: inputURL) else {
            throw RuntimeError("Failed to load audio file: \(inputURL.path)")
        }

        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw RuntimeError("Failed to create audio buffer")
        }

        try audioFile.read(into: buffer)

        print("Audio info:")
        print("  - Duration: \(String(format: "%.1f", Double(audioFile.length) / format.sampleRate))s")
        print("  - Sample rate: \(format.sampleRate) Hz")
        print("  - Channels: \(format.channelCount)")
        print()

        // Initialize diarized ASR manager
        let manager = StreamingDiarizedAsrManager(config: config)

        // Start processing
        print("Starting diarized transcription...")
        let startTime = Date()

        try await manager.start()

        // Monitor results in background - show all segments
        let resultsTask = Task {
            var segmentCount = 0
            for await result in await manager.results {
                segmentCount += 1
                let timeStart = String(format: "%.1f", result.audioTimeRange.start.seconds)
                let timeEnd = String(format: "%.1f", result.audioTimeRange.end.seconds)
                let confidence = String(format: "%.1f", result.combinedConfidence * 100)

                print(
                    "[\(timeStart)-\(timeEnd)s] Speaker \(result.speakerId): \(result.attributedText) (confidence: \(confidence)%)"
                )

                if debug {
                    print(
                        "  -> Transcription conf: \(String(format: "%.1f", result.transcriptionConfidence * 100))%, Speaker conf: \(String(format: "%.1f", result.speakerConfidence * 100))%"
                    )
                }
            }
            if segmentCount == 0 {
                print("No results received from the diarized ASR pipeline")
            }
        }

        // Stream audio data in chunks for real streaming behavior
        let chunkDuration = 2.0  // 2 second chunks
        let chunkSize = Int(format.sampleRate * chunkDuration * Double(format.channelCount))

        guard let channelData = buffer.floatChannelData?[0] else {
            throw RuntimeError("Failed to access audio data")
        }

        let totalFrames = Int(buffer.frameLength)
        var currentFrame = 0

        print("Streaming audio in chunks...")
        while currentFrame < totalFrames {
            let remainingFrames = totalFrames - currentFrame
            let framesToProcess = min(chunkSize, remainingFrames)

            // Create chunk buffer
            guard
                let chunkBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(framesToProcess))
            else {
                break
            }

            chunkBuffer.frameLength = AVAudioFrameCount(framesToProcess)
            let chunkData = chunkBuffer.floatChannelData![0]

            // Copy data
            for i in 0..<framesToProcess {
                chunkData[i] = channelData[currentFrame + i]
            }

            // Stream the chunk
            await manager.streamAudio(chunkBuffer)

            // Small delay to simulate real-time streaming
            try await Task.sleep(nanoseconds: UInt64(0.1 * 1_000_000_000))  // 100ms delay

            currentFrame += framesToProcess

            if debug {
                let progress = Double(currentFrame) / Double(totalFrames) * 100
                print("Streaming progress: \(String(format: "%.1f", progress))%")
            }
        }

        // Finish and get results
        let finalTranscripts = try await manager.finish()
        resultsTask.cancel()

        let processingTime = Date().timeIntervalSince(startTime)
        let audioDuration = Double(audioFile.length) / format.sampleRate
        let rtfx = audioDuration / processingTime

        // Display results
        print()
        print("Diarized transcription completed!")
        print("Processing time: \(String(format: "%.2f", processingTime))s")
        print("Real-time factor: \(String(format: "%.1f", rtfx))x")
        print("Speakers identified: \(finalTranscripts.count)")
        print()

        // Show speaker statistics
        let stats = await manager.sessionStatistics
        if !stats.isEmpty {
            print("Speaker Statistics:")
            for stat in stats.sorted(by: { $0.speakerId < $1.speakerId }) {
                print("  Speaker \(stat.speakerId):")
                print("    - Speaking time: \(String(format: "%.1f", stat.totalSpeakingTime))s")
                print("    - Segments: \(stat.segmentCount)")
                print("    - Avg confidence: \(String(format: "%.1f", stat.averageConfidence * 100))%")
            }
            print()
        }

        // Display transcripts
        print("Speaker Transcripts:")
        print(String.stringRepeat(string: "=", count: 50))
        for (speakerId, transcript) in finalTranscripts.sorted(by: { $0.key < $1.key }) {
            print("Speaker \(speakerId):")
            print("   \(transcript.trimmingCharacters(in: .whitespacesAndNewlines))")
            print()
        }

        // Save to output file if specified
        if let outputPath = output {
            try await saveResults(
                transcripts: finalTranscripts,
                stats: stats,
                processingTime: processingTime,
                rtfx: rtfx,
                to: outputPath
            )
            print("Results saved to: \(outputPath)")
        }
    }

    private static func runStreamingMode(
        inputURL: URL, config: StreamingDiarizedAsrConfig, debug: Bool, output: String?
    ) async throws {
        print("Processing audio file: \(inputURL.lastPathComponent)")
        print("Mode: Streaming simulation")
        print()

        // Load audio file
        guard let audioFile = try? AVAudioFile(forReading: inputURL) else {
            throw RuntimeError("Failed to load audio file: \(inputURL.path)")
        }

        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw RuntimeError("Failed to create audio buffer")
        }

        try audioFile.read(into: buffer)

        // Initialize diarized ASR manager
        let manager = StreamingDiarizedAsrManager(config: config)

        print("Starting streaming diarized transcription...")
        try await manager.start()

        // Monitor snapshots for real-time updates
        let snapshotTask = Task {
            var lastUpdate = Date()
            for await snapshot in await manager.snapshots {
                let now = Date()
                if now.timeIntervalSince(lastUpdate) >= 2.0 {  // Update every 2 seconds
                    print("\nLive Transcript Snapshot:")
                    print("Active speakers: \(snapshot.activeSpeakers.sorted().joined(separator: ", "))")

                    // Show volatile updates
                    for (speakerId, text) in snapshot.volatileBySpeaker.sorted(by: { $0.key < $1.key }) {
                        print("Speaker \(speakerId) (live): \(text)")
                    }

                    // Show finalized updates
                    for (speakerId, text) in snapshot.finalizedBySpeaker.sorted(by: { $0.key < $1.key }) {
                        let preview = String(text.characters).suffix(100)
                        print("Speaker \(speakerId) (final): ...\(preview)")
                    }
                    print(String.stringRepeat(string: "-", count: 40))
                    lastUpdate = now
                }
            }
        }

        // Simulate streaming by feeding audio in chunks
        let chunkDuration = 2.0  // 2 second chunks
        let chunkSize = Int(format.sampleRate * chunkDuration * Double(format.channelCount))

        guard let channelData = buffer.floatChannelData?[0] else {
            throw RuntimeError("Failed to access audio data")
        }

        let totalFrames = Int(buffer.frameLength)
        var currentFrame = 0

        while currentFrame < totalFrames {
            let remainingFrames = totalFrames - currentFrame
            let framesToProcess = min(chunkSize, remainingFrames)

            // Create chunk buffer
            guard
                let chunkBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(framesToProcess))
            else {
                break
            }

            chunkBuffer.frameLength = AVAudioFrameCount(framesToProcess)
            let chunkData = chunkBuffer.floatChannelData![0]

            // Copy data
            for i in 0..<framesToProcess {
                chunkData[i] = channelData[currentFrame + i]
            }

            // Stream the chunk
            await manager.streamAudio(chunkBuffer)

            // Simulate real-time delay
            try await Task.sleep(nanoseconds: UInt64(chunkDuration * 0.5 * 1_000_000_000))  // Half speed for demo

            currentFrame += framesToProcess

            if debug {
                let progress = Double(currentFrame) / Double(totalFrames) * 100
                print("Streaming progress: \(String(format: "%.1f", progress))%")
            }
        }

        // Finish processing
        let finalTranscripts = try await manager.finish()
        snapshotTask.cancel()

        print("\nStreaming diarized transcription completed!")
        print("Final speakers identified: \(finalTranscripts.count)")

        // Show final results
        for (speakerId, transcript) in finalTranscripts.sorted(by: { $0.key < $1.key }) {
            print("\nSpeaker \(speakerId) Final:")
            print("   \(transcript.trimmingCharacters(in: .whitespacesAndNewlines))")
        }
    }

    private static func saveResults(
        transcripts: [String: String],
        stats: [SpeakerSessionStats],
        processingTime: TimeInterval,
        rtfx: Double,
        to outputPath: String
    ) async throws {
        let output = DiarizedTranscriptOutput(
            speakers: transcripts,
            statistics: stats.map { stat in
                SpeakerStatOutput(
                    speakerId: stat.speakerId,
                    totalSpeakingTimeSeconds: stat.totalSpeakingTime,
                    segmentCount: stat.segmentCount,
                    averageConfidence: stat.averageConfidence
                )
            },
            processingMetrics: ProcessingMetricsOutput(
                processingTimeSeconds: processingTime,
                realTimeFactorX: rtfx
            ),
            timestamp: Date()
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(output)
        try data.write(to: URL(fileURLWithPath: outputPath))
    }
}

// MARK: - Output Models

struct DiarizedTranscriptOutput: Codable {
    let speakers: [String: String]
    let statistics: [SpeakerStatOutput]
    let processingMetrics: ProcessingMetricsOutput
    let timestamp: Date
}

struct SpeakerStatOutput: Codable {
    let speakerId: String
    let totalSpeakingTimeSeconds: TimeInterval
    let segmentCount: Int
    let averageConfidence: Float
}

struct ProcessingMetricsOutput: Codable {
    let processingTimeSeconds: TimeInterval
    let realTimeFactorX: Double
}

// MARK: - Helper Extensions

extension String {
    static func stringRepeat(string: String, count: Int) -> String {
        return String(repeating: string, count: count)
    }
}

// MARK: - Error Types

struct RuntimeError: Error, LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}

#endif
