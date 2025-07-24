import AVFoundation
import FluidAudio

/// Handler for the 'process' command - processes a single audio file
enum ProcessCommand {
    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            print("❌ No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var threshold: Float = 0.7
        var debugMode = false
        var outputFile: String?

        // Parse remaining arguments
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.7
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            default:
                print("⚠️ Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("🎵 Processing audio file: \(audioFile)")
        print("   Clustering threshold: \(threshold)")

        let config = DiarizerConfig(
            clusteringThreshold: threshold,
            debugMode: debugMode
        )

        let manager = DiarizerManager(config: config)

        do {
            try await manager.initialize()
            print("✅ Models initialized")
        } catch {
            print("❌ Failed to initialize models: \(error)")
            exit(1)
        }

        // Load and process audio file
        do {
            let audioSamples = try await AudioProcessor.loadAudioFile(path: audioFile)
            print("✅ Loaded audio: \(audioSamples.count) samples")

            let startTime = Date()
            let result = try await manager.performCompleteDiarization(
                audioSamples, sampleRate: 16000)
            let processingTime = Date().timeIntervalSince(startTime)

            let duration = Float(audioSamples.count) / 16000.0
            let rtf = Float(processingTime) / duration

            print("✅ Diarization completed in \(String(format: "%.1f", processingTime))s")
            print("   Real-time factor: \(String(format: "%.2f", rtf))x")
            print("   Found \(result.segments.count) segments")
            print("   Detected \(result.speakerDatabase.count) speakers (total), mapped: TBD")

            // Create output
            let output = ProcessingResult(
                audioFile: audioFile,
                durationSeconds: duration,
                processingTimeSeconds: processingTime,
                realTimeFactor: rtf,
                segments: result.segments,
                speakerCount: result.speakerDatabase.count,
                config: config
            )

            // Output results
            if let outputFile = outputFile {
                try await ResultsFormatter.saveResults(output, to: outputFile)
                print("💾 Results saved to: \(outputFile)")
            } else {
                await ResultsFormatter.printResults(output)
            }

        } catch {
            print("❌ Failed to process audio file: \(error)")
            exit(1)
        }
    }
    
    private static func printUsage() {
        print(
            """
            
            Process Command Usage:
                fluidaudio process <audio_file> [options]
            
            Options:
                --threshold <float>    Clustering threshold (default: 0.7)
                --debug               Enable debug mode
                --output <file>       Save results to file instead of stdout
            
            Example:
                fluidaudio process audio.wav --threshold 0.5 --output results.json
            """
        )
    }
}