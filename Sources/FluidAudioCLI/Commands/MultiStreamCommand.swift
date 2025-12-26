#if os(macOS)
@preconcurrency import AVFoundation
import FluidAudio
import Foundation

/// Command to demonstrate multi-stream ASR with shared model loading
enum MultiStreamCommand {
    private static let logger = AppLogger(category: "MultiStream")

    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            logger.error("No audio files specified")
            printUsage()
            exit(1)
        }

        let audioFile1 = arguments[0]
        var audioFile2: String? = nil

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                // Check if it's a second audio file
                if audioFile2 == nil && arguments[i].hasSuffix(".wav") {
                    audioFile2 = arguments[i]
                } else {
                    logger.warning("Unknown option: \(arguments[i])")
                }
            }
            i += 1
        }

        // Use same file for both streams if only one provided
        let micAudioFile = audioFile1
        let systemAudioFile = audioFile2 ?? audioFile1

        logger.info("🎤 Multi-Stream ASR Test\n========================\n")

        if audioFile2 != nil {
            logger.info(
                "📁 Processing two different files:\n  Microphone: \(micAudioFile)\n  System: \(systemAudioFile)\n")
        } else {
            logger.info("📁 Processing single file on both streams: \(audioFile1)\n")
        }

        do {
            // Load first audio file (microphone)
            let micFileURL = URL(fileURLWithPath: micAudioFile)
            let micFileHandle = try AVAudioFile(forReading: micFileURL)
            let micFormat = micFileHandle.processingFormat
            let micFrameCount = AVAudioFrameCount(micFileHandle.length)

            guard
                let micBuffer = AVAudioPCMBuffer(
                    pcmFormat: micFormat, frameCapacity: micFrameCount)
            else {
                logger.error("Failed to create microphone audio buffer")
                return
            }
            try micFileHandle.read(into: micBuffer)

            // Load second audio file (system)
            let systemFileURL = URL(fileURLWithPath: systemAudioFile)
            let systemFileHandle = try AVAudioFile(forReading: systemFileURL)
            let systemFormat = systemFileHandle.processingFormat
            let systemFrameCount = AVAudioFrameCount(systemFileHandle.length)

            guard
                let systemBuffer = AVAudioPCMBuffer(
                    pcmFormat: systemFormat, frameCapacity: systemFrameCount)
            else {
                logger.error("Failed to create system audio buffer")
                return
            }
            try systemFileHandle.read(into: systemBuffer)

            logger.info(
                """
                📊 Audio file info:
                🎙️ Microphone file:
                  Sample rate: \(micFormat.sampleRate) Hz
                  Channels: \(micFormat.channelCount)
                  Duration: \(String(format: "%.2f", Double(micFileHandle.length) / micFormat.sampleRate)) seconds

                System audio file:
                  Sample rate: \(systemFormat.sampleRate) Hz
                  Channels: \(systemFormat.channelCount)
                  Duration: \(String(format: "%.2f", Double(systemFileHandle.length) / systemFormat.sampleRate)) seconds
                """
            )

            // Create a streaming session
            logger.info("Creating streaming session...")
            let session = StreamingAsrSession()

            // Initialize models once
            logger.info("Loading ASR models (shared across streams)...")
            let startTime = Date()
            try await session.initialize()
            let loadTime = Date().timeIntervalSince(startTime)
            logger.info("Models loaded in \(String(format: "%.2f", loadTime))s\n")

            // Create streams for different sources
            logger.info("Creating streams for different audio sources...")
            let micStream = try await session.createStream(
                source: .microphone,
                config: .default
            )
            logger.info("Created microphone stream")

            let systemStream = try await session.createStream(
                source: .system,
                config: .default
            )
            logger.info("Created system audio stream\n")

            // Listen for updates from both streams (only if debug enabled)
            let micTask = Task {
                for await update in await micStream.transcriptionUpdates {
                    logger.info("[MIC] \(update.isConfirmed ? "✓" : "~") \(update.text)")
                }
            }

            let systemTask = Task {
                for await update in await systemStream.transcriptionUpdates {
                    logger.info("[SYS] \(update.isConfirmed ? "✓" : "~") \(update.text)")
                }
            }

            logger.info("Streaming audio files in parallel...\n  Both streams using default config (10.0s chunks)\n")

            // Process both files in parallel
            let micProcessingTask = Task {
                await streamAudioFile(
                    buffer: micBuffer,
                    format: micFormat,
                    to: micStream,
                    label: "MIC"
                )
            }

            let systemProcessingTask = Task {
                await streamAudioFile(
                    buffer: systemBuffer,
                    format: systemFormat,
                    to: systemStream,
                    label: "SYS"
                )
            }

            // Wait for both to complete
            await micProcessingTask.value
            await systemProcessingTask.value

            logger.info("Finalizing transcriptions...")

            // Get final results
            let micFinal = try await micStream.finish()
            let systemFinal = try await systemStream.finish()

            // Cancel update tasks
            micTask.cancel()
            systemTask.cancel()

            // Print results
            logger.info(
                "" + String(repeating: "=", count: 60) + "TRANSCRIPTION RESULTS\n"
                    + String(repeating: "=", count: 60) + "")
            logger.info("MICROPHONE STREAM:\n\(micFinal)")
            logger.info("SYSTEM AUDIO STREAM:\n\(systemFinal)")
            logger.info("Session info:")
            let activeStreams = await session.activeStreams
            logger.info("Active streams: \(activeStreams.count)")
            for (source, stream) in activeStreams {
                logger.info("  - \(source): \(await stream.source)")
            }

            await session.cleanup()
        } catch {
            logger.error("Error: \(error)")
        }
    }

    /// Helper function to stream an audio file to a stream
    private static func streamAudioFile(
        buffer: AVAudioPCMBuffer,
        format: AVAudioFormat,
        to stream: StreamingAsrManager,
        label: String
    ) async {
        let chunkDuration = 0.5  // 500ms chunks
        let samplesPerChunk = Int(chunkDuration * format.sampleRate)
        var position = 0

        while position < Int(buffer.frameLength) {
            let remainingSamples = Int(buffer.frameLength) - position
            let chunkSize = min(samplesPerChunk, remainingSamples)

            // Create chunk buffer
            guard
                let chunkBuffer = AVAudioPCMBuffer(
                    pcmFormat: format,
                    frameCapacity: AVAudioFrameCount(chunkSize)
                )
            else {
                break
            }

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

            await stream.streamAudio(chunkBuffer)

            position += chunkSize
        }

        logger.info("[\(label)] Streaming complete")
    }

    private static func printUsage() {
        logger.info(
            """

            Multi-Stream Command Usage:
                fluidaudio multi-stream <audio_file1> [audio_file2] [options]

            Options:
                --help, -h         Show this help message

            Examples:
                # Process same file on both streams
                fluidaudio multi-stream audio.wav

                # Process two different files in parallel
                fluidaudio multi-stream mic_audio.wav system_audio.wav


            This command demonstrates:
            - Loading ASR models once and sharing across streams
            - Creating separate streams for microphone and system audio
            - Parallel transcription with shared resources
            """
        )
    }
}
#endif
