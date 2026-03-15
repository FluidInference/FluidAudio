#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// LS-EEND diarization benchmark for evaluating performance on standard corpora
enum LSEENDBenchmark {
    private static let logger = AppLogger(category: "LSEENDBench")

    enum Dataset: String {
        case ami = "ami"
        case voxconverse = "voxconverse"
        case callhome = "callhome"
    }

    struct BenchmarkResult {
        let meetingName: String
        let der: Float
        let missRate: Float
        let falseAlarmRate: Float
        let speakerErrorRate: Float
        let rtfx: Float
        let processingTime: Double
        let totalFrames: Int
        let detectedSpeakers: Int
        let groundTruthSpeakers: Int
        let modelLoadTime: Double
        let audioLoadTime: Double
    }

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
                if !arguments[i].starts(with: "--") {
                    logger.warning("Unknown argument: \(arguments[i])")
                }
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
            switch dataset {
            case .ami:
                filesToProcess = getAMIFiles(maxFiles: maxFiles)
            case .voxconverse:
                filesToProcess = getVoxConverseFiles(maxFiles: maxFiles)
            case .callhome:
                filesToProcess = getCALLHOMEFiles(maxFiles: maxFiles)
            }
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
            if let loaded = loadProgress(from: progressFile) {
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

            let result = processMeeting(
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
                saveProgress(results: allResults, to: progressFile)
                print("Progress saved (\(allResults.count) files complete)")
            }
            fflush(stdout)

            // Reset diarizer state for next file
            diarizer.reset()
        }

        // Print final summary
        printFinalSummary(results: allResults)

        // Save results
        if let outputPath = outputFile {
            saveJSONResults(results: allResults, to: outputPath)
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
    ) -> BenchmarkResult? {
        let audioPath = getAudioPath(for: meetingName, dataset: dataset)
        guard FileManager.default.fileExists(atPath: audioPath) else {
            print("Audio file not found: \(audioPath)")
            fflush(stdout)
            return nil
        }

        do {
            // Load and process audio
            let audioLoadStart = Date()
            let audioURL = URL(fileURLWithPath: audioPath)
            let audioLoadTime = Date().timeIntervalSince(audioLoadStart)

            let startTime = Date()
            let timeline = try diarizer.processComplete(audioFileURL: audioURL)
            let processingTime = Date().timeIntervalSince(startTime)

            let duration = timeline.finalizedDuration
            let rtfx = duration / Float(processingTime)
            let numFrames = timeline.numFinalizedFrames

            if verbose {
                print("   Audio load time: \(String(format: "%.3f", audioLoadTime))s")
                print("   Processing time: \(String(format: "%.2f", processingTime))s")
                print("   RTFx: \(String(format: "%.1f", rtfx))x")
                print("   Total frames: \(numFrames)")
            }

            // Load ground truth RTTM
            let rttmURL = getRTTMURL(for: meetingName, dataset: dataset)
            guard let rttmURL = rttmURL, FileManager.default.fileExists(atPath: rttmURL.path) else {
                print("No RTTM ground truth found for \(meetingName)")
                return nil
            }

            let (rttmEntries, rttmSpeakers) = try LSEENDEvaluation.parseRTTM(url: rttmURL)
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
            let missPercent = evalResult.speakerScored > 0
                ? Float(evalResult.speakerMiss / evalResult.speakerScored * 100) : 0
            let faPercent = evalResult.speakerScored > 0
                ? Float(evalResult.speakerFalseAlarm / evalResult.speakerScored * 100) : 0
            let sePercent = evalResult.speakerScored > 0
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
                audioLoadTime: audioLoadTime
            )

        } catch {
            print("Error processing \(meetingName): \(error)")
            return nil
        }
    }

    // MARK: - File Paths

    private static func getAMIFiles(maxFiles: Int?) -> [String] {
        let allMeetings = [
            "EN2002a", "EN2002b", "EN2002c", "EN2002d",
            "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "IS1009a", "IS1009b", "IS1009c", "IS1009d",
            "TS3003a", "TS3003b", "TS3003c", "TS3003d",
        ]

        var availableMeetings: [String] = []
        for meeting in allMeetings {
            let path = getAudioPath(for: meeting, dataset: .ami)
            if FileManager.default.fileExists(atPath: path) {
                availableMeetings.append(meeting)
            }
        }

        if let max = maxFiles {
            return Array(availableMeetings.prefix(max))
        }
        return availableMeetings
    }

    private static func getAudioPath(for meeting: String, dataset: Dataset) -> String {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        switch dataset {
        case .ami:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/ami_official/sdm/\(meeting).Mix-Headset.wav"
            ).path
        case .voxconverse:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/voxconverse/voxconverse_test_wav/\(meeting).wav"
            ).path
        case .callhome:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/callhome_eng/\(meeting).wav"
            ).path
        }
    }

    private static func getRTTMURL(for meeting: String, dataset: Dataset) -> URL? {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        switch dataset {
        case .ami:
            // Try local RTTM first, then fall back to dataset directory
            let localPath = "Streaming-Sortformer-Conversion/\(meeting).rttm"
            if FileManager.default.fileExists(atPath: localPath) {
                return URL(fileURLWithPath: localPath)
            }
            let datasetPath = homeDir.appendingPathComponent(
                "FluidAudioDatasets/ami_official/rttm/\(meeting).rttm"
            )
            return datasetPath
        case .voxconverse:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/voxconverse/rttm_repo/test/\(meeting).rttm"
            )
        case .callhome:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/callhome_eng/rttm/\(meeting).rttm"
            )
        }
    }

    private static func getVoxConverseFiles(maxFiles: Int?) -> [String] {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let voxDir = homeDir.appendingPathComponent(
            "FluidAudioDatasets/voxconverse/voxconverse_test_wav"
        )

        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: voxDir,
                includingPropertiesForKeys: nil
            )
        else {
            return []
        }

        var availableMeetings: [String] = []
        for file in files where file.pathExtension == "wav" {
            let name = file.deletingPathExtension().lastPathComponent
            let rttmPath = homeDir.appendingPathComponent(
                "FluidAudioDatasets/voxconverse/rttm_repo/test/\(name).rttm"
            )
            if FileManager.default.fileExists(atPath: rttmPath.path) {
                availableMeetings.append(name)
            }
        }

        availableMeetings.sort()
        if let max = maxFiles {
            return Array(availableMeetings.prefix(max))
        }
        return availableMeetings
    }

    private static func getCALLHOMEFiles(maxFiles: Int?) -> [String] {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let callhomeDir = homeDir.appendingPathComponent("FluidAudioDatasets/callhome_eng")

        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: callhomeDir,
                includingPropertiesForKeys: nil
            )
        else {
            return []
        }

        var availableMeetings: [String] = []
        for file in files where file.pathExtension == "wav" {
            let name = file.deletingPathExtension().lastPathComponent
            let rttmPath = callhomeDir.appendingPathComponent("rttm/\(name).rttm")
            if FileManager.default.fileExists(atPath: rttmPath.path) {
                availableMeetings.append(name)
            }
        }

        availableMeetings.sort()
        if let max = maxFiles {
            return Array(availableMeetings.prefix(max))
        }
        return availableMeetings
    }

    // MARK: - Summary & Output

    private static func printFinalSummary(results: [BenchmarkResult]) {
        guard !results.isEmpty else { return }

        print("\n" + String(repeating: "=", count: 80))
        print("LS-EEND BENCHMARK SUMMARY")
        print(String(repeating: "=", count: 80))

        print("Results Sorted by DER:")
        print(String(repeating: "-", count: 70))
        print("Meeting        DER %    Miss %     FA %     SE %   Speakers     RTFx")
        print(String(repeating: "-", count: 70))

        for result in results.sorted(by: { $0.der < $1.der }) {
            let speakerInfo = "\(result.detectedSpeakers)/\(result.groundTruthSpeakers)"
            let meetingCol = result.meetingName.padding(toLength: 12, withPad: " ", startingAt: 0)
            let speakerCol = speakerInfo.padding(toLength: 10, withPad: " ", startingAt: 0)
            print(
                String(
                    format: "%@ %8.1f %8.1f %8.1f %8.1f %@ %8.1f",
                    meetingCol,
                    result.der,
                    result.missRate,
                    result.falseAlarmRate,
                    result.speakerErrorRate,
                    speakerCol,
                    result.rtfx))
        }
        print(String(repeating: "-", count: 70))

        let count = Float(results.count)
        let avgDER = results.map { $0.der }.reduce(0, +) / count
        let avgMiss = results.map { $0.missRate }.reduce(0, +) / count
        let avgFA = results.map { $0.falseAlarmRate }.reduce(0, +) / count
        let avgSE = results.map { $0.speakerErrorRate }.reduce(0, +) / count
        let avgRTFx = results.map { $0.rtfx }.reduce(0, +) / count

        print(
            String(
                format: "AVERAGE      %8.1f %8.1f %8.1f %8.1f         - %8.1f",
                avgDER, avgMiss, avgFA, avgSE, avgRTFx))
        print(String(repeating: "=", count: 70))

        print("\nTarget Check:")
        if avgDER < 15 {
            print("   DER < 15% (achieved: \(String(format: "%.1f", avgDER))%)")
        } else if avgDER < 25 {
            print("   DER < 25% (achieved: \(String(format: "%.1f", avgDER))%)")
        } else {
            print("   DER > 25% (achieved: \(String(format: "%.1f", avgDER))%)")
        }

        if avgRTFx > 1 {
            print("   RTFx > 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
        } else {
            print("   RTFx < 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
        }
    }

    private static func saveJSONResults(results: [BenchmarkResult], to path: String) {
        let jsonData = results.map { resultToDict($0) }
        do {
            let data = try JSONSerialization.data(withJSONObject: jsonData, options: .prettyPrinted)
            try data.write(to: URL(fileURLWithPath: path))
            print("JSON results saved to: \(path)")
        } catch {
            print("Failed to save JSON: \(error)")
        }
    }

    // MARK: - Progress Save/Load

    private static func resultToDict(_ result: BenchmarkResult) -> [String: Any] {
        return [
            "meeting": result.meetingName,
            "der": result.der,
            "missRate": result.missRate,
            "falseAlarmRate": result.falseAlarmRate,
            "speakerErrorRate": result.speakerErrorRate,
            "rtfx": result.rtfx,
            "processingTime": result.processingTime,
            "totalFrames": result.totalFrames,
            "detectedSpeakers": result.detectedSpeakers,
            "groundTruthSpeakers": result.groundTruthSpeakers,
            "modelLoadTime": result.modelLoadTime,
            "audioLoadTime": result.audioLoadTime,
        ]
    }

    private static func saveProgress(results: [BenchmarkResult], to path: String) {
        let jsonData = results.map { resultToDict($0) }
        do {
            let data = try JSONSerialization.data(withJSONObject: jsonData, options: .prettyPrinted)
            try data.write(to: URL(fileURLWithPath: path))
        } catch {
            print("Failed to save progress: \(error)")
        }
    }

    private static func loadProgress(from path: String) -> [BenchmarkResult]? {
        guard FileManager.default.fileExists(atPath: path) else { return nil }

        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            guard let jsonArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                return nil
            }

            return jsonArray.compactMap { dict -> BenchmarkResult? in
                guard let meeting = dict["meeting"] as? String,
                    let der = (dict["der"] as? NSNumber)?.floatValue,
                    let missRate = (dict["missRate"] as? NSNumber)?.floatValue,
                    let falseAlarmRate = (dict["falseAlarmRate"] as? NSNumber)?.floatValue,
                    let speakerErrorRate = (dict["speakerErrorRate"] as? NSNumber)?.floatValue,
                    let rtfx = (dict["rtfx"] as? NSNumber)?.floatValue,
                    let processingTime = (dict["processingTime"] as? NSNumber)?.doubleValue,
                    let totalFrames = (dict["totalFrames"] as? NSNumber)?.intValue,
                    let detectedSpeakers = (dict["detectedSpeakers"] as? NSNumber)?.intValue,
                    let groundTruthSpeakers = (dict["groundTruthSpeakers"] as? NSNumber)?.intValue,
                    let modelLoadTime = (dict["modelLoadTime"] as? NSNumber)?.doubleValue,
                    let audioLoadTime = (dict["audioLoadTime"] as? NSNumber)?.doubleValue
                else {
                    return nil
                }

                return BenchmarkResult(
                    meetingName: meeting,
                    der: der,
                    missRate: missRate,
                    falseAlarmRate: falseAlarmRate,
                    speakerErrorRate: speakerErrorRate,
                    rtfx: rtfx,
                    processingTime: processingTime,
                    totalFrames: totalFrames,
                    detectedSpeakers: detectedSpeakers,
                    groundTruthSpeakers: groundTruthSpeakers,
                    modelLoadTime: modelLoadTime,
                    audioLoadTime: audioLoadTime
                )
            }
        } catch {
            print("Failed to load progress: \(error)")
            return nil
        }
    }
}
#endif
