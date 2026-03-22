#if os(macOS)
import FluidAudio
import Foundation

/// LS-EEND diarization benchmark for evaluating performance on standard corpora
enum LSEENDBenchmark {
    private static let logger = AppLogger(category: "LSEENDBench")

    typealias Dataset = DiarizationBenchmarkUtils.Dataset
    typealias BenchmarkResult = DiarizationBenchmarkUtils.BenchmarkResult

    static func printUsage() {
        print(
            """
            LS-EEND Benchmark Command

            Evaluates LS-EEND speaker diarization on various corpora.

            Usage: fluidaudio lseend-benchmark [options]

            Options:
                --dataset <name>         Dataset to use: ami, voxconverse, callhome (default: ami)
                --variant <name>         Model variant: ami, callhome, dihard2, dihard3 (default: dihard3)
                --single-file <name>     Process a specific meeting (e.g., ES2004a)
                --max-files <n>          Maximum number of files to process
                --threshold <value>      Speaker activity threshold (default: 0.5)
                --median-width <value>   Median filter width for post-processing (default: 1)
                --collar <value>         Collar duration in seconds (default: 0.25)
                --onset <value>          Onset threshold for speech detection (default: 0.5)
                --offset <value>         Offset threshold for speech detection (default: 0.5)
                --pad-onset <value>      Padding before speech segments in seconds
                --pad-offset <value>     Padding after speech segments in seconds
                --min-duration-on <v>    Minimum speech segment duration in seconds
                --min-duration-off <v>   Minimum silence duration in seconds
                --output <file>          Output JSON file for results
                --progress <file>        Progress file for resuming (default: .lseend_progress.json)
                --resume                 Resume from previous progress file
                --verbose                Enable verbose output
                --auto-download          Auto-download AMI dataset if missing
                --help                   Show this help message

            Examples:
                # Quick test on one file
                fluidaudio lseend-benchmark --single-file ES2004a

                # Full AMI benchmark with auto-download
                fluidaudio lseend-benchmark --auto-download --output results.json

                # Benchmark with CALLHOME variant on CALLHOME dataset
                fluidaudio lseend-benchmark --dataset callhome --variant callhome
            """)
    }

    static func run(arguments: [String]) async {
        // Parse arguments
        var singleFile: String?
        var maxFiles: Int?
        var threshold: Float = 0.5
        var medianWidth: Int = 1
        var collarSeconds: Double = 0.25
        var outputFile: String?
        var verbose = false
        var autoDownload = false

        // Post-processing parameters
        var onset: Float?
        var offset: Float?
        var padOnset: Float?
        var padOffset: Float?
        var minDurationOn: Float?
        var minDurationOff: Float?
        var progressFile: String = ".lseend_progress.json"
        var resumeFromProgress = false
        var dataset: Dataset = .ami
        var variant: LSEENDVariant = .dihard3

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    if let d = Dataset(rawValue: arguments[i + 1].lowercased()) {
                        dataset = d
                    } else {
                        print("Unknown dataset: \(arguments[i + 1]). Using ami.")
                    }
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
                        print("Unknown variant: \(arguments[i + 1]). Using dihard3.")
                    }
                    i += 1
                }
            case "--single-file":
                if i + 1 < arguments.count {
                    singleFile = arguments[i + 1]
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1])
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.5
                    i += 1
                }
            case "--median-width":
                if i + 1 < arguments.count {
                    medianWidth = Int(arguments[i + 1]) ?? 1
                    i += 1
                }
            case "--collar":
                if i + 1 < arguments.count {
                    collarSeconds = Double(arguments[i + 1]) ?? 0.25
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--progress":
                if i + 1 < arguments.count {
                    progressFile = arguments[i + 1]
                    i += 1
                }
            case "--resume":
                resumeFromProgress = true
            case "--verbose":
                verbose = true
            case "--onset":
                if i + 1 < arguments.count {
                    onset = Float(arguments[i + 1])
                    i += 1
                }
            case "--offset":
                if i + 1 < arguments.count {
                    offset = Float(arguments[i + 1])
                    i += 1
                }
            case "--pad-onset":
                if i + 1 < arguments.count {
                    padOnset = Float(arguments[i + 1])
                    i += 1
                }
            case "--pad-offset":
                if i + 1 < arguments.count {
                    padOffset = Float(arguments[i + 1])
                    i += 1
                }
            case "--min-duration-on":
                if i + 1 < arguments.count {
                    minDurationOn = Float(arguments[i + 1])
                    i += 1
                }
            case "--min-duration-off":
                if i + 1 < arguments.count {
                    minDurationOff = Float(arguments[i + 1])
                    i += 1
                }
            case "--auto-download":
                autoDownload = true
            case "--help":
                printUsage()
                return
            default:
                logger.warning("Unknown argument: \(arguments[i])")
            }
            i += 1
        }

        print("Starting LS-EEND Benchmark")
        fflush(stdout)
        print("   Dataset: \(dataset.rawValue)")
        print("   Variant: \(variant.rawValue)")
        print("   Threshold: \(threshold)")
        print("   Median width: \(medianWidth)")
        print("   Collar: \(collarSeconds)s")

        // Download dataset if needed
        if autoDownload && dataset == .ami {
            print("Downloading AMI dataset if needed...")
            await DatasetDownloader.downloadAMIDataset(
                variant: .sdm,
                force: false,
                singleFile: singleFile
            )
            await DatasetDownloader.downloadAMIAnnotations(force: false)
        }

        // Get list of files to process
        let filesToProcess: [String]
        if let meeting = singleFile {
            filesToProcess = [meeting]
        } else {
            filesToProcess = DiarizationBenchmarkUtils.getFiles(for: dataset, maxFiles: maxFiles)
        }

        if filesToProcess.isEmpty {
            print("No files found to process")
            fflush(stdout)
            return
        }

        print("Processing \(filesToProcess.count) file(s)")
        print("   Progress file: \(progressFile)")
        fflush(stdout)

        // Load previous progress if resuming
        var completedResults: [BenchmarkResult] = []
        var completedMeetings: Set<String> = []
        if resumeFromProgress {
            if let loaded = DiarizationBenchmarkUtils.loadProgress(from: progressFile) {
                completedResults = loaded
                completedMeetings = Set(loaded.map { $0.meetingName })
                print("Resuming: loaded \(completedResults.count) previous results")
                for result in completedResults {
                    print("   \(result.meetingName): \(String(format: "%.1f", result.der))% DER")
                }
            } else {
                print("No previous progress found, starting fresh")
            }
        }
        print("")
        fflush(stdout)

        // Initialize LS-EEND
        print("Loading LS-EEND models...")
        fflush(stdout)
        let modelLoadStart = Date()

        var timelineConfig = DiarizerTimelineConfig(onsetThreshold: threshold, onsetPadFrames: 0)
        if let v = onset { timelineConfig.onsetThreshold = v }
        if let v = offset { timelineConfig.offsetThreshold = v }
        if let v = padOnset { timelineConfig.onsetPadSeconds = v }
        if let v = padOffset { timelineConfig.offsetPadSeconds = v }
        if let v = minDurationOn { timelineConfig.minDurationOn = v }
        if let v = minDurationOff { timelineConfig.minDurationOff = v }

        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly, timelineConfig: timelineConfig)

        do {
            try await diarizer.initialize(variant: variant)
        } catch {
            print("Failed to initialize LS-EEND: \(error)")
            return
        }

        let modelLoadTime = Date().timeIntervalSince(modelLoadStart)

        guard let frameHz = diarizer.modelFrameHz,
            let numSpeakers = diarizer.numSpeakers
        else {
            print("Failed to read model parameters after initialization")
            return
        }

        print("Models loaded in \(String(format: "%.2f", modelLoadTime))s")
        print("   Frame rate: \(String(format: "%.1f", frameHz)) Hz, Speakers: \(numSpeakers)\n")
        fflush(stdout)

        // Process each file
        var allResults: [BenchmarkResult] = completedResults

        for (fileIndex, meetingName) in filesToProcess.enumerated() {
            // Skip already completed files
            if completedMeetings.contains(meetingName) {
                print("[\(fileIndex + 1)/\(filesToProcess.count)] Skipping (already done): \(meetingName)")
                fflush(stdout)
                continue
            }

            print(String(repeating: "=", count: 60))
            print("[\(fileIndex + 1)/\(filesToProcess.count)] Processing: \(meetingName)")
            print(String(repeating: "=", count: 60))
            fflush(stdout)

            let result = await processMeeting(
                meetingName: meetingName,
                dataset: dataset,
                diarizer: diarizer,
                modelLoadTime: modelLoadTime,
                threshold: threshold,
                medianWidth: medianWidth,
                collarSeconds: collarSeconds,
                frameHz: frameHz,
                numSpeakers: numSpeakers,
                verbose: verbose
            )

            if let result = result {
                allResults.append(result)

                print("Results for \(meetingName):")
                print("   DER: \(String(format: "%.1f", result.der))%")
                print("   RTFx: \(String(format: "%.1f", result.rtfx))x")
                print("   Speakers: \(result.detectedSpeakers) detected / \(result.groundTruthSpeakers) truth")

                // Save progress after each file
                DiarizationBenchmarkUtils.saveProgress(results: allResults, to: progressFile)
                print("Progress saved (\(allResults.count) files complete)")
            }
            fflush(stdout)

            // Reset diarizer state for next file
            diarizer.reset()
        }

        // Print final summary
        DiarizationBenchmarkUtils.printFinalSummary(
            results: allResults,
            title: "LS-EEND BENCHMARK SUMMARY",
            derTargets: [15, 25]
        )

        // Save results
        if let outputPath = outputFile {
            DiarizationBenchmarkUtils.saveJSONResults(results: allResults, to: outputPath)
        }
    }

    private static func processMeeting(
        meetingName: String,
        dataset: Dataset,
        diarizer: LSEENDDiarizer,
        modelLoadTime: Double,
        threshold: Float,
        medianWidth: Int,
        collarSeconds: Double,
        frameHz: Double,
        numSpeakers: Int,
        verbose: Bool
    ) async -> BenchmarkResult? {
        let audioPath = DiarizationBenchmarkUtils.getAudioPath(for: meetingName, dataset: dataset)
        guard FileManager.default.fileExists(atPath: audioPath) else {
            print("Audio file not found: \(audioPath)")
            fflush(stdout)
            return nil
        }

        do {
            // Load and process audio
            let audioURL = URL(fileURLWithPath: audioPath)
            let startTime = Date()
            let timeline = try diarizer.processComplete(audioFileURL: audioURL)
            let processingTime = Date().timeIntervalSince(startTime)

            let duration = timeline.finalizedDuration
            let rtfx = duration / Float(processingTime)
            let numFrames = timeline.numFinalizedFrames

            if verbose {
                print("   Processing time: \(String(format: "%.2f", processingTime))s")
                print("   RTFx: \(String(format: "%.1f", rtfx))x")
                print("   Total frames: \(numFrames)")
            }

            // Load ground truth RTTM (or fall back to AMI XML annotations)
            let rttmEntries: [LSEENDRTTMEntry]
            let rttmSpeakers: [String]

            let rttmURL = DiarizationBenchmarkUtils.getRTTMURL(for: meetingName, dataset: dataset)
            if let rttmURL = rttmURL, FileManager.default.fileExists(atPath: rttmURL.path) {
                let parsed = try LSEENDEvaluation.parseRTTM(url: rttmURL)
                rttmEntries = parsed.entries
                rttmSpeakers = parsed.speakers
            } else if dataset == .ami {
                // Fall back to AMI XML annotations (same as SortformerBenchmark)
                print("   [RTTM] No RTTM file, falling back to AMI annotations")
                let groundTruth = await AMIParser.loadAMIGroundTruth(
                    for: meetingName,
                    duration: duration
                )
                guard !groundTruth.isEmpty else {
                    print("No ground truth found for \(meetingName)")
                    return nil
                }
                // Convert TimedSpeakerSegment to LSEENDRTTMEntry
                var speakers: [String] = []
                var entries: [LSEENDRTTMEntry] = []
                for segment in groundTruth {
                    if !speakers.contains(segment.speakerId) {
                        speakers.append(segment.speakerId)
                    }
                    entries.append(
                        LSEENDRTTMEntry(
                            recordingID: meetingName,
                            start: Double(segment.startTimeSeconds),
                            duration: Double(segment.endTimeSeconds - segment.startTimeSeconds),
                            speaker: segment.speakerId
                        )
                    )
                }
                rttmEntries = entries
                rttmSpeakers = speakers
            } else {
                print("No RTTM ground truth found for \(meetingName)")
                return nil
            }

            let referenceBinary = LSEENDEvaluation.rttmToFrameMatrix(
                entries: rttmEntries,
                speakers: rttmSpeakers,
                numFrames: numFrames,
                frameRate: frameHz
            )

            print("   [RTTM] Loaded \(rttmEntries.count) segments, speakers: \(rttmSpeakers)")

            // Build probability matrix from timeline predictions
            let predictions = timeline.finalizedPredictions
            let probMatrix = LSEENDMatrix(
                validatingRows: numFrames,
                columns: numSpeakers,
                values: predictions
            )

            // Compute DER using the built-in evaluation
            let settings = LSEENDEvaluationSettings(
                threshold: threshold,
                medianWidth: medianWidth,
                collarSeconds: collarSeconds,
                frameRate: frameHz
            )
            let evalResult = LSEENDEvaluation.computeDER(
                probabilities: probMatrix,
                referenceBinary: referenceBinary,
                settings: settings
            )

            let derPercent = Float(evalResult.der * 100)
            let missPercent =
                evalResult.speakerScored > 0
                ? Float(evalResult.speakerMiss / evalResult.speakerScored * 100) : 0
            let faPercent =
                evalResult.speakerScored > 0
                ? Float(evalResult.speakerFalseAlarm / evalResult.speakerScored * 100) : 0
            let sePercent =
                evalResult.speakerScored > 0
                ? Float(evalResult.speakerError / evalResult.speakerScored * 100) : 0

            print(
                "   DER breakdown: miss=\(String(format: "%.1f", missPercent))%, "
                    + "FA=\(String(format: "%.1f", faPercent))%, "
                    + "SE=\(String(format: "%.1f", sePercent))%"
            )
            fflush(stdout)

            // Count detected speakers from segments
            var detectedSpeakerIndices = Set<Int>()
            for (_, speaker) in timeline.speakers {
                if !speaker.finalizedSegments.isEmpty {
                    detectedSpeakerIndices.insert(speaker.index)
                }
            }

            return BenchmarkResult(
                meetingName: meetingName,
                der: derPercent,
                missRate: missPercent,
                falseAlarmRate: faPercent,
                speakerErrorRate: sePercent,
                rtfx: rtfx,
                processingTime: processingTime,
                totalFrames: numFrames,
                detectedSpeakers: detectedSpeakerIndices.count,
                groundTruthSpeakers: rttmSpeakers.count,
                modelLoadTime: modelLoadTime,
                audioLoadTime: nil
            )

        } catch {
            print("Error processing \(meetingName): \(error)")
            return nil
        }
    }

}
#endif
