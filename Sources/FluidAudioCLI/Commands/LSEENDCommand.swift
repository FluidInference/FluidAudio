#if os(macOS)
import FluidAudio
import Foundation

/// Handler for the 'lseend' command - LS-EEND streaming diarization
enum LSEENDCommand {
    private static let logger = AppLogger(category: "LSEEND")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            fputs("ERROR: No audio file specified\n", stderr)
            fflush(stderr)
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var outputFile: String?
        var variant: LSEENDVariant = .dihard3
        var threshold: Float = 0.5

        // Post-processing parameters
        var onset: Float?
        var offset: Float?
        var padOnset: Float?
        var padOffset: Float?
        var minDurationOn: Float?
        var minDurationOff: Float?

        // Parse remaining arguments
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--variant":
                if i + 1 < arguments.count {
                    let v = arguments[i + 1].lowercased()
                    switch v {
                    case "ami":
                        variant = .ami
                    case "callhome":
                        variant = .callhome
                    case "dihard2":
                        variant = .dihard2
                    case "dihard3":
                        variant = .dihard3
                    default:
                        logger.warning("Unknown variant: \(arguments[i + 1]), using dihard3")
                    }
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    threshold = v
                    i += 1
                }
            case "--onset":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    onset = v
                    i += 1
                }
            case "--offset":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    offset = v
                    i += 1
                }
            case "--pad-onset":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    padOnset = v
                    i += 1
                }
            case "--pad-offset":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    padOffset = v
                    i += 1
                }
            case "--min-duration-on":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    minDurationOn = v
                    i += 1
                }
            case "--min-duration-off":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    minDurationOff = v
                    i += 1
                }
            case "--help":
                printUsage()
                return
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("LS-EEND Diarization")
        print("   Audio: \(audioFile)")
        print("   Variant: \(variant.rawValue)")
        print("   Threshold: \(threshold)")

        var timelineConfig = DiarizerTimelineConfig(onsetThreshold: threshold, onsetPadFrames: 0)
        if let v = onset { timelineConfig.onsetThreshold = v }
        if let v = offset { timelineConfig.offsetThreshold = v }
        if let v = padOnset { timelineConfig.onsetPadSeconds = v }
        if let v = padOffset { timelineConfig.offsetPadSeconds = v }
        if let v = minDurationOn { timelineConfig.minDurationOn = v }
        if let v = minDurationOff { timelineConfig.minDurationOff = v }

        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly, timelineConfig: timelineConfig)

        do {
            let loadStart = Date()
            print("Loading models from HuggingFace...")
            try await diarizer.initialize(variant: variant)
            let loadTime = Date().timeIntervalSince(loadStart)
            print("Models loaded in \(String(format: "%.2f", loadTime))s")

            guard let sampleRate = diarizer.targetSampleRate,
                let frameHz = diarizer.modelFrameHz,
                let numSpeakers = diarizer.numSpeakers
            else {
                print("ERROR: Failed to read model parameters after initialization")
                exit(1)
            }
            print("   Sample rate: \(sampleRate) Hz")
            print("   Frame rate: \(String(format: "%.1f", frameHz)) Hz")
            print("   Speakers: \(numSpeakers)")
        } catch {
            print("ERROR: Failed to initialize LS-EEND: \(error)")
            exit(1)
        }

        do {
            print("Processing...")
            fflush(stdout)
            let startTime = Date()
            let audioURL = URL(fileURLWithPath: audioFile)
            let timeline = try diarizer.processComplete(audioFileURL: audioURL)

            let processingTime = Date().timeIntervalSince(startTime)
            let duration = timeline.finalizedDuration
            let rtfx = duration / Float(processingTime)

            print("Processing completed in \(String(format: "%.2f", processingTime))s")
            print("   Duration: \(String(format: "%.1f", duration))s")
            print("   Real-time factor (RTFx): \(String(format: "%.1f", rtfx))x")
            print("   Total frames: \(timeline.numFinalizedFrames)")
            print("   Frame duration: \(String(format: "%.3f", timeline.config.frameDurationSeconds))s")

            // Collect all segments across speakers
            var allSegments: [DiarizerSegment] = []
            for (_, speaker) in timeline.speakers {
                allSegments.append(contentsOf: speaker.finalizedSegments)
            }
            allSegments.sort()

            print("   Found \(allSegments.count) segments")

            // Print segments
            print("\n--- Speaker Segments ---")
            for segment in allSegments {
                let start = String(format: "%.2f", segment.startTime)
                let end = String(format: "%.2f", segment.endTime)
                let dur = String(format: "%.2f", segment.duration)
                print("\(segment.speakerLabel): \(start)s - \(end)s (\(dur)s)")
            }

            // Print speaker activity summary
            let numSpeakers = timeline.config.numSpeakers
            print("\n--- Speaker Activity Summary ---")
            let predictions = timeline.finalizedPredictions
            let numFrames = timeline.numFinalizedFrames
            var speakerActivity = [Float](repeating: 0, count: numSpeakers)
            let activityThreshold = timeline.config.onsetThreshold
            for frame in 0..<numFrames {
                for spk in 0..<numSpeakers {
                    let idx = frame * numSpeakers + spk
                    if idx < predictions.count, predictions[idx] > activityThreshold {
                        speakerActivity[spk] += timeline.config.frameDurationSeconds
                    }
                }
            }
            for spk in 0..<numSpeakers {
                let activeTime = String(format: "%.1f", speakerActivity[spk])
                let percent = String(format: "%.1f", (speakerActivity[spk] / duration) * 100)
                print("Speaker \(spk): \(activeTime)s active (\(percent)%)")
            }

            // Save output if requested
            if let outputFile = outputFile {
                var output: [String: Any] = [
                    "audioFile": audioFile,
                    "variant": variant.rawValue,
                    "durationSeconds": duration,
                    "processingTimeSeconds": processingTime,
                    "rtfx": rtfx,
                    "totalFrames": numFrames,
                    "frameDurationSeconds": timeline.config.frameDurationSeconds,
                    "segmentCount": allSegments.count,
                ]

                var segmentDicts: [[String: Any]] = []
                for segment in allSegments {
                    segmentDicts.append([
                        "speaker": segment.speakerLabel,
                        "speakerIndex": segment.speakerIndex,
                        "startTimeSeconds": segment.startTime,
                        "endTimeSeconds": segment.endTime,
                        "durationSeconds": segment.duration,
                    ])
                }
                output["segments"] = segmentDicts

                let jsonData = try JSONSerialization.data(
                    withJSONObject: output,
                    options: [.prettyPrinted, .sortedKeys]
                )
                try jsonData.write(to: URL(fileURLWithPath: outputFile))
                print("Results saved to: \(outputFile)")
            }

        } catch {
            print("ERROR: Failed to process audio: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        print(
            """

            LS-EEND Command Usage:
                fluidaudio lseend <audio_file> [options]

            Options:
                --variant <name>        Model variant: ami, callhome, dihard2, dihard3 (default: dihard3)
                --threshold <value>     Speaker activity threshold (default: 0.5)
                --onset <value>         Onset threshold for speech detection (default: 0.5)
                --offset <value>        Offset threshold for speech detection (default: 0.5)
                --pad-onset <value>     Padding before speech segments in seconds
                --pad-offset <value>    Padding after speech segments in seconds
                --min-duration-on <v>   Minimum speech segment duration in seconds
                --min-duration-off <v>  Minimum silence duration in seconds
                --output <file>         Save results to JSON file
                --help                  Show this help message

            Examples:
                # Basic usage (downloads model from HuggingFace)
                fluidaudio lseend audio.wav

                # With specific variant
                fluidaudio lseend audio.wav --variant ami

                # Save results to file
                fluidaudio lseend audio.wav --output results.json
            """)
    }
}
#endif
