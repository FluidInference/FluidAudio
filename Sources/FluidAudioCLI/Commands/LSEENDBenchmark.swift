#if os(macOS)
import FluidAudio
import Foundation

/// LS-EEND diarization benchmark for evaluating performance on standard corpora
enum LSEENDBenchmark {
    private static let logger = AppLogger(category: "LSEENDBench")
    private static let frameStepForDER: Double = 0.01

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
                --ami-split <name>       AMI split: dev, test, train (default: test)
                --variant <name>         Model variant: ami, callhome, dihard2, dihard3 (default: ami)
                --step-size <name>       Model step size: 100ms, 200ms, 300ms, 400ms, 500ms (default: 500ms)
                --single-file <name>     Process a specific meeting (e.g., ES2004a)
                --max-files <n>          Maximum number of files to process
                --threshold <value>      Speaker activity threshold (default: 0.5)
                --median-width <value>   Median filter width for post-processing (default: 1)
                --collar <value>         Collar duration in seconds (default: 0.0 for AMI, 0.25 otherwise)
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

                # Benchmark with AMI 500ms model
                fluidaudio lseend-benchmark --variant ami --step-size 500ms
            """)
    }

    static func run(arguments: [String]) async {
        // Parse arguments
        var singleFile: String?
        var maxFiles: Int?
        var threshold: Float = 0.5
        var medianWidth: Int = 1
        var collarSeconds: Double = 0.25
        var collarWasProvided = false
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
        var amiSplit: DiarizationBenchmarkUtils.AMISplit = .test
        var variant: LSEENDVariant = .ami
        var stepSize: LSEENDStepSize = .step500ms

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
            case "--ami-split":
                if i + 1 < arguments.count {
                    if let split = DiarizationBenchmarkUtils.AMISplit(rawValue: arguments[i + 1].lowercased()) {
                        amiSplit = split
                    } else {
                        print("Unknown AMI split: \(arguments[i + 1]). Using test.")
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
            case "--step-size":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "100", "100ms":
                        stepSize = .step100ms
                    case "200", "200ms":
                        stepSize = .step200ms
                    case "300", "300ms":
                        stepSize = .step300ms
                    case "400", "400ms":
                        stepSize = .step400ms
                    case "500", "500ms":
                        stepSize = .step500ms
                    default:
                        print("Unknown step size: \(arguments[i + 1]). Using 500ms.")
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
                    collarWasProvided = true
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

        if dataset == .ami && !collarWasProvided {
            collarSeconds = 0.0
        }

        print("Starting LS-EEND Benchmark")
        fflush(stdout)
        print("   Dataset: \(dataset.rawValue)")
        if dataset == .ami {
            print("   AMI split: \(amiSplit.rawValue)")
        }
        print("   Variant: \(variant.description)")
        print("   Step size: \(stepSize.description)")
        print("   Threshold: \(threshold)")
        print("   Median width: \(medianWidth)")
        print("   Collar: \(collarSeconds)s")

        // Download dataset if needed
        if autoDownload && dataset == .ami {
            print("Downloading AMI dataset if needed...")
            let meetingsToDownload =
                singleFile.map { [$0] } ?? DiarizationBenchmarkUtils.getAMIMeetings(split: amiSplit)
            await DatasetDownloader.downloadAMIDataset(
                variant: .sdm,
                force: false,
                singleFile: singleFile,
                meetingIds: meetingsToDownload
            )
            await DatasetDownloader.downloadAMIAnnotations(force: false)
        }

        let amiSplitDirectory: URL?
        if dataset == .ami {
            let splitDirectory = AMIKaldiData.splitDirectory(split: amiSplit)

            do {
                if autoDownload {
                    try AMIKaldiData.ensureSplitExists(split: amiSplit)
                } else if !AMIKaldiData.splitExists(split: amiSplit) {
                    print("AMI Kaldi split not found at \(splitDirectory.path)")
                    print(
                        "Run `fluidaudio lseend-benchmark --auto-download` to build Datasets/ami/mhs/data/\(amiSplit.rawValue)."
                    )
                    return
                }
            } catch {
                print("Failed to prepare AMI Kaldi data: \(error)")
                return
            }

            amiSplitDirectory = splitDirectory
        } else {
            amiSplitDirectory = nil
        }

        // Get list of files to process
        let filesToProcess: [String]
        if let meeting = singleFile {
            filesToProcess = [meeting]
        } else if dataset == .ami {
            guard let amiSplitDirectory else {
                print("AMI Kaldi split directory was not initialized.")
                return
            }

            do {
                filesToProcess = try AMIKaldiData.recordingIDs(in: amiSplitDirectory, maxFiles: maxFiles)
            } catch {
                print("Failed to enumerate AMI Kaldi recordings: \(error)")
                return
            }
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

        let diarizer: LSEENDDiarizer

        do {
            let model = try await LSEENDModel.loadFromHuggingFace(
                variant: variant,
                stepSize: stepSize,
                computeUnits: .cpuOnly
            )
            diarizer = try LSEENDDiarizer(model: model)
            diarizer.timeline = DiarizerTimeline(
                config: configuredTimelineConfig(
                    base: timelineConfig,
                    diarizer: diarizer
                )
            )
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
                amiSplitDirectory: amiSplitDirectory,
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
        amiSplitDirectory: URL?,
        diarizer: LSEENDDiarizer,
        modelLoadTime: Double,
        threshold: Float,
        medianWidth: Int,
        collarSeconds: Double,
        frameHz: Double,
        numSpeakers: Int,
        verbose: Bool
    ) async -> BenchmarkResult? {
        do {
            let audioPath: String
            if dataset == .ami {
                guard let amiSplitDirectory else {
                    print("AMI Kaldi split directory was not initialized.")
                    return nil
                }
                guard let path = try AMIKaldiData.audioPath(for: meetingName, in: amiSplitDirectory) else {
                    print("AMI Kaldi wav.scp has no entry for \(meetingName)")
                    return nil
                }
                audioPath = path
            } else {
                audioPath = DiarizationBenchmarkUtils.getAudioPath(for: meetingName, dataset: dataset)
            }

            guard FileManager.default.fileExists(atPath: audioPath) else {
                print("Audio file not found: \(audioPath)")
                fflush(stdout)
                return nil
            }

            // Load and process audio
            let audioURL = URL(fileURLWithPath: audioPath)
            let startTime = Date()
            let timeline = try diarizer.processComplete(
                audioFileURL: audioURL,
                keepingEnrolledSpeakers: nil,
                finalizeOnCompletion: true,
                progressCallback: nil
            )
            let processingTime = Date().timeIntervalSince(startTime)

            let duration = timeline.finalizedDuration
            let rtfx = duration / Float(processingTime)
            let numFrames = timeline.numFinalizedFrames

            if verbose {
                print("   Processing time: \(String(format: "%.2f", processingTime))s")
                print("   RTFx: \(String(format: "%.1f", rtfx))x")
                print("   Total frames: \(numFrames)")
            }

            let referenceSegments: [DERSpeakerSegment]
            let groundTruthSpeakers: Int

            if dataset == .ami {
                print("   [REF] Using AMI word-aligned annotations")
                referenceSegments = await AMIParser.loadWordAlignedDERReference(
                    for: meetingName,
                    duration: duration
                )
                groundTruthSpeakers = Set(referenceSegments.map(\.speaker)).count
            } else if let rttmURL = DiarizationBenchmarkUtils.getRTTMURL(for: meetingName, dataset: dataset),
                FileManager.default.fileExists(atPath: rttmURL.path)
            {
                let groundTruth = try RTTMParser.loadSegments(from: rttmURL.path)
                referenceSegments = groundTruth.map {
                    DERSpeakerSegment(
                        speaker: $0.speakerId,
                        start: Double($0.startTimeSeconds),
                        end: Double($0.endTimeSeconds)
                    )
                }
                groundTruthSpeakers = Set(groundTruth.map(\.speakerId)).count
            } else {
                print("No RTTM ground truth found for \(meetingName)")
                return nil
            }

            print(
                "   [REF] Loaded \(referenceSegments.count) segments, speakers: \(groundTruthSpeakers)"
            )

            let hypothesisSegments = timelineToDERSegments(
                timeline,
                numSpeakers: numSpeakers,
                threshold: threshold,
                medianWidth: medianWidth
            )

            let evalResult = DiarizationDER.compute(
                ref: referenceSegments,
                hyp: hypothesisSegments,
                frameStep: frameStepForDER,
                collar: collarSeconds
            )

            let totalRefSpeech = max(evalResult.totalRefSpeech, .leastNonzeroMagnitude)
            let derPercent = Float(evalResult.der * 100)
            let missPercent = Float(evalResult.miss / totalRefSpeech * 100)
            let faPercent = Float(evalResult.falseAlarm / totalRefSpeech * 100)
            let sePercent = Float(evalResult.confusion / totalRefSpeech * 100)

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
                groundTruthSpeakers: groundTruthSpeakers,
                modelLoadTime: modelLoadTime,
                audioLoadTime: nil
            )

        } catch {
            print("Error processing \(meetingName): \(error)")
            return nil
        }
    }

    private static func configuredTimelineConfig(
        base: DiarizerTimelineConfig,
        diarizer: LSEENDDiarizer
    ) -> DiarizerTimelineConfig {
        var config = base
        config.numSpeakers = diarizer.numSpeakers ?? config.numSpeakers
        if let frameHz = diarizer.modelFrameHz {
            config.frameDurationSeconds = Float(1.0 / frameHz)
        }
        return config
    }

    private static func timelineToDERSegments(
        _ timeline: DiarizerTimeline,
        numSpeakers: Int,
        threshold: Float,
        medianWidth: Int
    ) -> [DERSpeakerSegment] {
        let binary = probabilitiesToBinary(
            timeline.finalizedPredictions,
            numFrames: timeline.numFinalizedFrames,
            numSpeakers: numSpeakers,
            threshold: threshold,
            medianWidth: medianWidth
        )
        return binaryToSegments(
            binary,
            numFrames: timeline.numFinalizedFrames,
            numSpeakers: numSpeakers,
            frameStep: Double(timeline.config.frameDurationSeconds)
        )
    }

    private static func probabilitiesToBinary(
        _ predictions: [Float],
        numFrames: Int,
        numSpeakers: Int,
        threshold: Float,
        medianWidth: Int
    ) -> [Bool] {
        var out = [Bool](repeating: false, count: numFrames * numSpeakers)
        for frame in 0..<numFrames {
            for speaker in 0..<numSpeakers {
                let index = frame * numSpeakers + speaker
                out[index] = index < predictions.count && predictions[index] > threshold
            }
        }

        guard medianWidth > 1 else { return out }
        var filtered = out
        let halfWindow = medianWidth / 2
        for speaker in 0..<numSpeakers {
            for frame in 0..<numFrames {
                let start = max(0, frame - halfWindow)
                let end = min(numFrames, frame + halfWindow + 1)
                var active = 0
                for candidate in start..<end where out[candidate * numSpeakers + speaker] {
                    active += 1
                }
                filtered[frame * numSpeakers + speaker] = active * 2 >= (end - start)
            }
        }
        return filtered
    }

    private static func binaryToSegments(
        _ binary: [Bool],
        numFrames: Int,
        numSpeakers: Int,
        frameStep: Double
    ) -> [DERSpeakerSegment] {
        var segments: [DERSpeakerSegment] = []
        for speaker in 0..<numSpeakers {
            var runStart: Int? = nil
            for frame in 0..<numFrames {
                let isActive = binary[frame * numSpeakers + speaker]
                if isActive {
                    runStart = runStart ?? frame
                    continue
                }
                if let startFrame = runStart {
                    segments.append(
                        DERSpeakerSegment(
                            speaker: String(speaker),
                            start: Double(startFrame) * frameStep,
                            end: Double(frame) * frameStep
                        )
                    )
                    runStart = nil
                }
            }

            if let startFrame = runStart {
                segments.append(
                    DERSpeakerSegment(
                        speaker: String(speaker),
                        start: Double(startFrame) * frameStep,
                        end: Double(numFrames) * frameStep
                    )
                )
            }
        }
        return segments
    }

}
#endif
