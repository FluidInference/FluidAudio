import AVFoundation
import FluidAudioSwift
import Foundation

@main
struct DiarizationCLI {

    static func main() async {
        let arguments = CommandLine.arguments

        guard arguments.count > 1 else {
            printUsage()
            exit(1)
        }

        let command = arguments[1]

        switch command {
        case "benchmark":
            await runBenchmark(arguments: Array(arguments.dropFirst(2)))
        case "process":
            await processFile(arguments: Array(arguments.dropFirst(2)))
        case "download":
            await downloadDataset(arguments: Array(arguments.dropFirst(2)))
        case "help", "--help", "-h":
            printUsage()
        default:
            print("❌ Unknown command: \(command)")
            printUsage()
            exit(1)
        }
    }

    static func printUsage() {
        print(
            """
            FluidAudioSwift Diarization CLI

            USAGE:
                fluidaudio <command> [options]

            COMMANDS:
                benchmark    Run AMI SDM benchmark evaluation with real annotations
                process      Process a single audio file
                download     Download datasets for benchmarking
                help         Show this help message

            BENCHMARK OPTIONS:
                --dataset <name>        Dataset to use (ami-sdm, ami-ihm) [default: ami-sdm]
                --threshold <float>     Clustering threshold 0.0-1.0 [default: 0.7]
                --min-duration-on <float>   Minimum speaker segment duration in seconds [default: 1.0]
                --min-duration-off <float>  Minimum silence between speakers in seconds [default: 0.5]
                --min-activity <float>      Minimum activity threshold in frames [default: 10.0]
                --single-file <name>    Test only one specific meeting file (e.g., ES2004a)
                --debug                 Enable debug mode
                --output <file>         Output results to JSON file
                --auto-download         Automatically download dataset if not found

            NOTE: Benchmark now uses real AMI manual annotations from Tests/ami_public_1.6.2/
                  If annotations are not found, falls back to simplified placeholder.

            PROCESS OPTIONS:
                <audio-file>         Audio file to process (.wav, .m4a, .mp3)
                --output <file>      Output results to JSON file [default: stdout]
                --threshold <float>  Clustering threshold 0.0-1.0 [default: 0.7]
                --debug             Enable debug mode

            DOWNLOAD OPTIONS:
                --dataset <name>     Dataset to download (ami-sdm, ami-ihm, all) [default: all]
                --force             Force re-download even if files exist

            EXAMPLES:
                # Download AMI datasets
                swift run fluidaudio download --dataset ami-sdm

                # Run AMI SDM benchmark with auto-download
                swift run fluidaudio benchmark --auto-download

                # Run benchmark with custom threshold and save results
                swift run fluidaudio benchmark --threshold 0.8 --output results.json

                # Process a single audio file
                swift run fluidaudio process meeting.wav

                # Process file with custom settings
                swift run fluidaudio process meeting.wav --threshold 0.6 --output output.json
            """)
    }

    static func runBenchmark(arguments: [String]) async {
        let benchmarkStartTime = Date()

        var dataset = "ami-sdm"
        var threshold: Float = 0.7
        var minDurationOn: Float = 1.0
        var minDurationOff: Float = 0.5
        var minActivityThreshold: Float = 10.0
        var singleFile: String?
        var debugMode = false
        var outputFile: String?
        var autoDownload = false

        // Parse arguments
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1]
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.7
                    i += 1
                }
            case "--min-duration-on":
                if i + 1 < arguments.count {
                    minDurationOn = Float(arguments[i + 1]) ?? 1.0
                    i += 1
                }
            case "--min-duration-off":
                if i + 1 < arguments.count {
                    minDurationOff = Float(arguments[i + 1]) ?? 0.5
                    i += 1
                }
            case "--min-activity":
                if i + 1 < arguments.count {
                    minActivityThreshold = Float(arguments[i + 1]) ?? 10.0
                    i += 1
                }
            case "--single-file":
                if i + 1 < arguments.count {
                    singleFile = arguments[i + 1]
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--auto-download":
                autoDownload = true
            default:
                print("⚠️ Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("🚀 Starting \(dataset.uppercased()) benchmark evaluation")
        print("   Clustering threshold: \(threshold)")
        print("   Min duration on: \(minDurationOn)s")
        print("   Min duration off: \(minDurationOff)s")
        print("   Min activity threshold: \(minActivityThreshold)")
        print("   Debug mode: \(debugMode ? "enabled" : "disabled")")
        print("   Auto-download: \(autoDownload ? "enabled" : "disabled")")

        let config = DiarizerConfig(
            clusteringThreshold: threshold,
            minDurationOn: minDurationOn,
            minDurationOff: minDurationOff,
            minActivityThreshold: minActivityThreshold,
            debugMode: debugMode
        )

        let manager = DiarizerManager(config: config)

        do {
            try await manager.initialize()
            print("✅ Models initialized successfully")
        } catch {
            print("❌ Failed to initialize models: \(error)")
            print("💡 Make sure you have network access for model downloads")
            exit(1)
        }

        // Run benchmark based on dataset
        switch dataset.lowercased() {
        case "ami-sdm":
            await runAMISDMBenchmark(
                manager: manager, outputFile: outputFile, autoDownload: autoDownload,
                singleFile: singleFile)
        case "ami-ihm":
            await runAMIIHMBenchmark(
                manager: manager, outputFile: outputFile, autoDownload: autoDownload,
                singleFile: singleFile)
        default:
            print("❌ Unsupported dataset: \(dataset)")
            print("💡 Supported datasets: ami-sdm, ami-ihm")
            exit(1)
        }

        let benchmarkElapsed = Date().timeIntervalSince(benchmarkStartTime)
        print("\n⏱️ Total benchmark execution time: \(String(format: "%.1f", benchmarkElapsed)) seconds")
    }

    static func downloadDataset(arguments: [String]) async {
        var dataset = "all"
        var forceDownload = false

        // Parse arguments
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1]
                    i += 1
                }
            case "--force":
                forceDownload = true
            default:
                print("⚠️ Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("📥 Starting dataset download")
        print("   Dataset: \(dataset)")
        print("   Force download: \(forceDownload ? "enabled" : "disabled")")

        switch dataset.lowercased() {
        case "ami-sdm":
            await downloadAMIDataset(variant: .sdm, force: forceDownload)
        case "ami-ihm":
            await downloadAMIDataset(variant: .ihm, force: forceDownload)
        case "all":
            await downloadAMIDataset(variant: .sdm, force: forceDownload)
            await downloadAMIDataset(variant: .ihm, force: forceDownload)
        default:
            print("❌ Unsupported dataset: \(dataset)")
            print("💡 Supported datasets: ami-sdm, ami-ihm, all")
            exit(1)
        }
    }

    static func processFile(arguments: [String]) async {
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
            let audioSamples = try await loadAudioFile(path: audioFile)
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
            print("   Detected \(result.speakerDatabase.count) speakers")

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
                try await saveResults(output, to: outputFile)
                print("💾 Results saved to: \(outputFile)")
            } else {
                await printResults(output)
            }

        } catch {
            print("❌ Failed to process audio file: \(error)")
            exit(1)
        }
    }

    // MARK: - AMI Benchmark Implementation

    static func runAMISDMBenchmark(
        manager: DiarizerManager, outputFile: String?, autoDownload: Bool, singleFile: String? = nil
    ) async {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let amiDirectory = homeDir.appendingPathComponent(
            "FluidAudioSwiftDatasets/ami_official/sdm")

        // Check if AMI dataset exists, download if needed
        if !FileManager.default.fileExists(atPath: amiDirectory.path) {
            if autoDownload {
                print("📥 AMI SDM dataset not found - downloading automatically...")
                await downloadAMIDataset(variant: .sdm, force: false)

                // Check again after download
                if !FileManager.default.fileExists(atPath: amiDirectory.path) {
                    print("❌ Failed to download AMI SDM dataset")
                    return
                }
            } else {
                print("⚠️ AMI SDM dataset not found")
                print("📥 Download options:")
                print("   Option 1: Use --auto-download flag")
                print("   Option 2: Download manually:")
                print("      1. Visit: https://groups.inf.ed.ac.uk/ami/download/")
                print(
                    "      2. Select test meetings: ES2002a, ES2003a, ES2004a, IS1000a, IS1001a")
                print("      3. Download 'Headset mix' (Mix-Headset.wav files)")
                print("      4. Place files in: \(amiDirectory.path)")
                print("   Option 3: Use download command:")
                print("      swift run fluidaudio download --dataset ami-sdm")
                return
            }
        }

        let commonMeetings: [String]
        if let singleFile = singleFile {
            commonMeetings = [singleFile]
            print("📋 Testing single file: \(singleFile)")
        } else {
            commonMeetings = [
                // Core AMI test set - smaller subset for initial benchmarking
                "ES2002a", "ES2003a", "ES2004a", "ES2005a",
                "IS1000a", "IS1001a", "IS1002b",
                "TS3003a", "TS3004a",
            ]
        }

        var benchmarkResults: [BenchmarkResult] = []
        var totalDER: Float = 0.0
        var totalJER: Float = 0.0
        var processedFiles = 0

        print("📊 Running AMI SDM Benchmark")
        print("   Looking for Mix-Headset.wav files in: \(amiDirectory.path)")

        for meetingId in commonMeetings {
            let audioFileName = "\(meetingId).Mix-Headset.wav"
            let audioPath = amiDirectory.appendingPathComponent(audioFileName)

            guard FileManager.default.fileExists(atPath: audioPath.path) else {
                print("   ⏭️ Skipping \(audioFileName) (not found)")
                continue
            }

            print("   🎵 Processing \(audioFileName)...")

            do {
                let audioSamples = try await loadAudioFile(path: audioPath.path)
                let duration = Float(audioSamples.count) / 16000.0

                let startTime = Date()
                let result = try await manager.performCompleteDiarization(
                    audioSamples, sampleRate: 16000)
                let processingTime = Date().timeIntervalSince(startTime)

                // Load ground truth from AMI annotations
                let groundTruth = await Self.loadAMIGroundTruth(for: meetingId, duration: duration)

                // Calculate metrics
                let metrics = calculateDiarizationMetrics(
                    predicted: result.segments,
                    groundTruth: groundTruth,
                    totalDuration: duration
                )

                totalDER += metrics.der
                totalJER += metrics.jer
                processedFiles += 1

                let rtf = Float(processingTime) / duration

                print(
                    "     ✅ DER: \(String(format: "%.1f", metrics.der))%, JER: \(String(format: "%.1f", metrics.jer))%, RTF: \(String(format: "%.2f", rtf))x"
                )

                benchmarkResults.append(
                    BenchmarkResult(
                        meetingId: meetingId,
                        durationSeconds: duration,
                        processingTimeSeconds: processingTime,
                        realTimeFactor: rtf,
                        der: metrics.der,
                        jer: metrics.jer,
                        segments: result.segments,
                        speakerCount: result.speakerDatabase.count
                    ))

            } catch {
                print("     ❌ Failed: \(error)")
            }
        }

        guard processedFiles > 0 else {
            print("❌ No files were processed successfully")
            return
        }

        let avgDER = totalDER / Float(processedFiles)
        let avgJER = totalJER / Float(processedFiles)

        // Print detailed results table
        printBenchmarkResults(benchmarkResults, avgDER: avgDER, avgJER: avgJER, dataset: "AMI-SDM")

        // Save results if requested
        if let outputFile = outputFile {
            let summary = BenchmarkSummary(
                dataset: "AMI-SDM",
                averageDER: avgDER,
                averageJER: avgJER,
                processedFiles: processedFiles,
                totalFiles: commonMeetings.count,
                results: benchmarkResults
            )

            do {
                try await saveBenchmarkResults(summary, to: outputFile)
                print("💾 Benchmark results saved to: \(outputFile)")
            } catch {
                print("⚠️ Failed to save results: \(error)")
            }
        }
    }

    static func runAMIIHMBenchmark(
        manager: DiarizerManager, outputFile: String?, autoDownload: Bool, singleFile: String? = nil
    ) async {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let amiDirectory = homeDir.appendingPathComponent(
            "FluidAudioSwiftDatasets/ami_official/ihm")

        // Check if AMI dataset exists, download if needed
        if !FileManager.default.fileExists(atPath: amiDirectory.path) {
            if autoDownload {
                print("📥 AMI IHM dataset not found - downloading automatically...")
                await downloadAMIDataset(variant: .ihm, force: false)

                // Check again after download
                if !FileManager.default.fileExists(atPath: amiDirectory.path) {
                    print("❌ Failed to download AMI IHM dataset")
                    return
                }
            } else {
                print("⚠️ AMI IHM dataset not found")
                print("📥 Download options:")
                print("   Option 1: Use --auto-download flag")
                print("   Option 2: Download manually:")
                print("      1. Visit: https://groups.inf.ed.ac.uk/ami/download/")
                print(
                    "      2. Select test meetings: ES2002a, ES2003a, ES2004a, IS1000a, IS1001a")
                print("      3. Download 'Individual headsets' (Headset-0.wav files)")
                print("      4. Place files in: \(amiDirectory.path)")
                print("   Option 3: Use download command:")
                print("      swift run fluidaudio download --dataset ami-ihm")
                return
            }
        }

        let commonMeetings = [
            // Core AMI test set - smaller subset for initial benchmarking
            "ES2002a", "ES2003a", "ES2004a", "ES2005a",
            "IS1000a", "IS1001a", "IS1002b",
            "TS3003a", "TS3004a",
        ]

        var benchmarkResults: [BenchmarkResult] = []
        var totalDER: Float = 0.0
        var totalJER: Float = 0.0
        var processedFiles = 0

        print("📊 Running AMI IHM Benchmark")
        print("   Looking for Headset-0.wav files in: \(amiDirectory.path)")

        for meetingId in commonMeetings {
            let audioFileName = "\(meetingId).Headset-0.wav"
            let audioPath = amiDirectory.appendingPathComponent(audioFileName)

            guard FileManager.default.fileExists(atPath: audioPath.path) else {
                print("   ⏭️ Skipping \(audioFileName) (not found)")
                continue
            }

            print("   🎵 Processing \(audioFileName)...")

            do {
                let audioSamples = try await loadAudioFile(path: audioPath.path)
                let duration = Float(audioSamples.count) / 16000.0

                let startTime = Date()
                let result = try await manager.performCompleteDiarization(
                    audioSamples, sampleRate: 16000)
                let processingTime = Date().timeIntervalSince(startTime)

                // Load ground truth from AMI annotations
                let groundTruth = await Self.loadAMIGroundTruth(for: meetingId, duration: duration)

                // Calculate metrics
                let metrics = calculateDiarizationMetrics(
                    predicted: result.segments,
                    groundTruth: groundTruth,
                    totalDuration: duration
                )

                totalDER += metrics.der
                totalJER += metrics.jer
                processedFiles += 1

                let rtf = Float(processingTime) / duration

                print(
                    "     ✅ DER: \(String(format: "%.1f", metrics.der))%, JER: \(String(format: "%.1f", metrics.jer))%, RTF: \(String(format: "%.2f", rtf))x"
                )

                benchmarkResults.append(
                    BenchmarkResult(
                        meetingId: meetingId,
                        durationSeconds: duration,
                        processingTimeSeconds: processingTime,
                        realTimeFactor: rtf,
                        der: metrics.der,
                        jer: metrics.jer,
                        segments: result.segments,
                        speakerCount: result.speakerDatabase.count
                    ))

            } catch {
                print("     ❌ Failed: \(error)")
            }
        }

        guard processedFiles > 0 else {
            print("❌ No files were processed successfully")
            return
        }

        let avgDER = totalDER / Float(processedFiles)
        let avgJER = totalJER / Float(processedFiles)

        // Print detailed results table
        printBenchmarkResults(benchmarkResults, avgDER: avgDER, avgJER: avgJER, dataset: "AMI-IHM")

        // Save results if requested
        if let outputFile = outputFile {
            let summary = BenchmarkSummary(
                dataset: "AMI-IHM",
                averageDER: avgDER,
                averageJER: avgJER,
                processedFiles: processedFiles,
                totalFiles: commonMeetings.count,
                results: benchmarkResults
            )

            do {
                try await saveBenchmarkResults(summary, to: outputFile)
                print("💾 Benchmark results saved to: \(outputFile)")
            } catch {
                print("⚠️ Failed to save results: \(error)")
            }
        }
    }

    // MARK: - Audio Processing

    static func loadAudioFile(path: String) async throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        let audioFile = try AVAudioFile(forReading: url)

        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw NSError(
                domain: "AudioError", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
        }

        try audioFile.read(into: buffer)

        guard let floatChannelData = buffer.floatChannelData else {
            throw NSError(
                domain: "AudioError", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to get float channel data"])
        }

        let actualFrameCount = Int(buffer.frameLength)
        var samples: [Float] = []

        if format.channelCount == 1 {
            samples = Array(
                UnsafeBufferPointer(start: floatChannelData[0], count: actualFrameCount))
        } else {
            // Mix stereo to mono
            let leftChannel = UnsafeBufferPointer(
                start: floatChannelData[0], count: actualFrameCount)
            let rightChannel = UnsafeBufferPointer(
                start: floatChannelData[1], count: actualFrameCount)

            samples = zip(leftChannel, rightChannel).map { (left, right) in
                (left + right) / 2.0
            }
        }

        // Resample to 16kHz if necessary
        if format.sampleRate != 16000 {
            samples = try await resampleAudio(samples, from: format.sampleRate, to: 16000)
        }

        return samples
    }

    static func resampleAudio(
        _ samples: [Float], from sourceSampleRate: Double, to targetSampleRate: Double
    ) async throws -> [Float] {
        if sourceSampleRate == targetSampleRate {
            return samples
        }

        let ratio = sourceSampleRate / targetSampleRate
        let outputLength = Int(Double(samples.count) / ratio)
        var resampled: [Float] = []
        resampled.reserveCapacity(outputLength)

        for i in 0..<outputLength {
            let sourceIndex = Double(i) * ratio
            let index = Int(sourceIndex)

            if index < samples.count - 1 {
                let fraction = sourceIndex - Double(index)
                let sample =
                    samples[index] * Float(1.0 - fraction) + samples[index + 1] * Float(fraction)
                resampled.append(sample)
            } else if index < samples.count {
                resampled.append(samples[index])
            }
        }

        return resampled
    }

    // MARK: - Ground Truth and Metrics

    static func generateSimplifiedGroundTruth(duration: Float, speakerCount: Int)
        -> [TimedSpeakerSegment]
    {
        let segmentDuration = duration / Float(speakerCount * 2)
        var segments: [TimedSpeakerSegment] = []
        let dummyEmbedding: [Float] = Array(repeating: 0.1, count: 512)

        for i in 0..<(speakerCount * 2) {
            let speakerId = "Speaker \((i % speakerCount) + 1)"
            let startTime = Float(i) * segmentDuration
            let endTime = min(startTime + segmentDuration, duration)

            segments.append(
                TimedSpeakerSegment(
                    speakerId: speakerId,
                    embedding: dummyEmbedding,
                    startTimeSeconds: startTime,
                    endTimeSeconds: endTime,
                    qualityScore: 1.0
                ))
        }

        return segments
    }

    static func calculateDiarizationMetrics(
        predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment], totalDuration: Float
    ) -> DiarizationMetrics {
        let frameSize: Float = 0.01
        let totalFrames = Int(totalDuration / frameSize)

        // Step 1: Find optimal speaker assignment using frame-based overlap
        let speakerMapping = findOptimalSpeakerMapping(
            predicted: predicted, groundTruth: groundTruth, totalDuration: totalDuration)

        print("🔍 SPEAKER MAPPING: \(speakerMapping)")

        var missedFrames = 0
        var falseAlarmFrames = 0
        var speakerErrorFrames = 0

        for frame in 0..<totalFrames {
            let frameTime = Float(frame) * frameSize

            let gtSpeaker = findSpeakerAtTime(frameTime, in: groundTruth)
            let predSpeaker = findSpeakerAtTime(frameTime, in: predicted)

            switch (gtSpeaker, predSpeaker) {
            case (nil, nil):
                continue
            case (nil, _):
                falseAlarmFrames += 1
            case (_, nil):
                missedFrames += 1
            case let (gt?, pred?):
                // Map predicted speaker ID to ground truth speaker ID
                let mappedPredSpeaker = speakerMapping[pred] ?? pred
                if gt != mappedPredSpeaker {
                    speakerErrorFrames += 1
                    // Debug first few mismatches
                    if speakerErrorFrames <= 5 {
                        print(
                            "🔍 DER DEBUG: Speaker mismatch at \(String(format: "%.2f", frameTime))s - GT: '\(gt)' vs Pred: '\(pred)' (mapped: '\(mappedPredSpeaker)')"
                        )
                    }
                }
            }
        }

        let der =
            Float(missedFrames + falseAlarmFrames + speakerErrorFrames) / Float(totalFrames) * 100
        let jer = calculateJaccardErrorRate(predicted: predicted, groundTruth: groundTruth)

        // Debug error breakdown
        print(
            "🔍 DER BREAKDOWN: Missed: \(missedFrames), FalseAlarm: \(falseAlarmFrames), SpeakerError: \(speakerErrorFrames), Total: \(totalFrames)"
        )
        print(
            "🔍 DER RATES: Miss: \(String(format: "%.1f", Float(missedFrames) / Float(totalFrames) * 100))%, FA: \(String(format: "%.1f", Float(falseAlarmFrames) / Float(totalFrames) * 100))%, SE: \(String(format: "%.1f", Float(speakerErrorFrames) / Float(totalFrames) * 100))%"
        )

        return DiarizationMetrics(
            der: der,
            jer: jer,
            missRate: Float(missedFrames) / Float(totalFrames) * 100,
            falseAlarmRate: Float(falseAlarmFrames) / Float(totalFrames) * 100,
            speakerErrorRate: Float(speakerErrorFrames) / Float(totalFrames) * 100
        )
    }

    static func calculateJaccardErrorRate(
        predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment]
    ) -> Float {
        // If no segments in either prediction or ground truth, return 100% error
        if predicted.isEmpty && groundTruth.isEmpty {
            return 0.0  // Perfect match - both empty
        } else if predicted.isEmpty || groundTruth.isEmpty {
            return 100.0  // Complete mismatch - one empty, one not
        }

        // Use the same frame size as DER calculation for consistency
        let frameSize: Float = 0.01
        let totalDuration = max(
            predicted.map { $0.endTimeSeconds }.max() ?? 0,
            groundTruth.map { $0.endTimeSeconds }.max() ?? 0
        )
        let totalFrames = Int(totalDuration / frameSize)

        // Get optimal speaker mapping using existing Hungarian algorithm
        let speakerMapping = findOptimalSpeakerMapping(
            predicted: predicted,
            groundTruth: groundTruth,
            totalDuration: totalDuration
        )

        var intersectionFrames = 0
        var unionFrames = 0

        // Calculate frame-by-frame Jaccard
        for frame in 0..<totalFrames {
            let frameTime = Float(frame) * frameSize

            let gtSpeaker = findSpeakerAtTime(frameTime, in: groundTruth)
            let predSpeaker = findSpeakerAtTime(frameTime, in: predicted)

            // Map predicted speaker to ground truth speaker using optimal mapping
            let mappedPredSpeaker = predSpeaker.flatMap { speakerMapping[$0] }

            switch (gtSpeaker, mappedPredSpeaker) {
            case (nil, nil):
                // Both silent - no contribution to intersection or union
                continue
            case (nil, _):
                // Ground truth silent, prediction has speaker
                unionFrames += 1
            case (_, nil):
                // Ground truth has speaker, prediction silent
                unionFrames += 1
            case let (gt?, pred?):
                // Both have speakers
                unionFrames += 1
                if gt == pred {
                    // Same speaker - contributes to intersection
                    intersectionFrames += 1
                }
                // Different speakers - only contributes to union
            }
        }

        // Calculate Jaccard Index
        let jaccardIndex = unionFrames > 0 ? Float(intersectionFrames) / Float(unionFrames) : 0.0

        // Convert to error rate: JER = 1 - Jaccard Index
        let jer = (1.0 - jaccardIndex) * 100.0

        // Debug logging for first few calculations
        if predicted.count > 0 && groundTruth.count > 0 {
            print("🔍 JER DEBUG: Intersection: \(intersectionFrames), Union: \(unionFrames), Jaccard Index: \(String(format: "%.3f", jaccardIndex)), JER: \(String(format: "%.1f", jer))%")
        }

        return jer
    }

    static func findSpeakerAtTime(_ time: Float, in segments: [TimedSpeakerSegment]) -> String? {
        for segment in segments {
            if time >= segment.startTimeSeconds && time < segment.endTimeSeconds {
                return segment.speakerId
            }
        }
        return nil
    }

    /// Find optimal speaker mapping using frame-by-frame overlap analysis
    static func findOptimalSpeakerMapping(
        predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment], totalDuration: Float
    ) -> [String: String] {
        let frameSize: Float = 0.01
        let totalFrames = Int(totalDuration / frameSize)

        // Get all unique speaker IDs
        let predSpeakers = Set(predicted.map { $0.speakerId })
        let gtSpeakers = Set(groundTruth.map { $0.speakerId })

        // Build overlap matrix: [predSpeaker][gtSpeaker] = overlap_frames
        var overlapMatrix: [String: [String: Int]] = [:]

        for predSpeaker in predSpeakers {
            overlapMatrix[predSpeaker] = [:]
            for gtSpeaker in gtSpeakers {
                overlapMatrix[predSpeaker]![gtSpeaker] = 0
            }
        }

        // Calculate frame-by-frame overlaps
        for frame in 0..<totalFrames {
            let frameTime = Float(frame) * frameSize

            let gtSpeaker = findSpeakerAtTime(frameTime, in: groundTruth)
            let predSpeaker = findSpeakerAtTime(frameTime, in: predicted)

            if let gt = gtSpeaker, let pred = predSpeaker {
                overlapMatrix[pred]![gt]! += 1
            }
        }

        // Find optimal assignment using Hungarian Algorithm for globally optimal solution
        let predSpeakerArray = Array(predSpeakers).sorted()  // Consistent ordering
        let gtSpeakerArray = Array(gtSpeakers).sorted()      // Consistent ordering

        // Build numerical overlap matrix for Hungarian algorithm
        var numericalOverlapMatrix: [[Int]] = []
        for predSpeaker in predSpeakerArray {
            var row: [Int] = []
            for gtSpeaker in gtSpeakerArray {
                row.append(overlapMatrix[predSpeaker]![gtSpeaker]!)
            }
            numericalOverlapMatrix.append(row)
        }

        // Convert overlap matrix to cost matrix (higher overlap = lower cost)
        let costMatrix = HungarianAlgorithm.overlapToCostMatrix(numericalOverlapMatrix)

        // Solve optimal assignment
        let assignments = HungarianAlgorithm.minimumCostAssignment(costs: costMatrix)

        // Create speaker mapping from Hungarian result
        var mapping: [String: String] = [:]
        var totalAssignmentCost: Float = 0
        var totalOverlap = 0

        for (predIndex, gtIndex) in assignments.assignments.enumerated() {
            if gtIndex != -1 && predIndex < predSpeakerArray.count && gtIndex < gtSpeakerArray.count {
                let predSpeaker = predSpeakerArray[predIndex]
                let gtSpeaker = gtSpeakerArray[gtIndex]
                let overlap = overlapMatrix[predSpeaker]![gtSpeaker]!

                if overlap > 0 {  // Only assign if there's actual overlap
                    mapping[predSpeaker] = gtSpeaker
                    totalOverlap += overlap
                    print("🔍 HUNGARIAN MAPPING: '\(predSpeaker)' → '\(gtSpeaker)' (overlap: \(overlap) frames)")
                }
            }
        }

        totalAssignmentCost = assignments.totalCost
        print("🔍 HUNGARIAN RESULT: Total assignment cost: \(String(format: "%.1f", totalAssignmentCost)), Total overlap: \(totalOverlap) frames")

        // Handle unassigned predicted speakers
        for predSpeaker in predSpeakerArray {
            if mapping[predSpeaker] == nil {
                print("🔍 HUNGARIAN MAPPING: '\(predSpeaker)' → NO_MATCH (no beneficial assignment)")
            }
        }

        return mapping
    }

    // MARK: - Output and Results

    static func printResults(_ result: ProcessingResult) async {
        print("\n📊 Diarization Results:")
        print("   Audio File: \(result.audioFile)")
        print("   Duration: \(String(format: "%.1f", result.durationSeconds))s")
        print("   Processing Time: \(String(format: "%.1f", result.processingTimeSeconds))s")
        print("   Real-time Factor: \(String(format: "%.2f", result.realTimeFactor))x")
        print("   Detected Speakers: \(result.speakerCount)")
        print("\n🎤 Speaker Segments:")

        for (index, segment) in result.segments.enumerated() {
            let startTime = formatTime(segment.startTimeSeconds)
            let endTime = formatTime(segment.endTimeSeconds)
            let duration = segment.endTimeSeconds - segment.startTimeSeconds

            print(
                "   \(index + 1). \(segment.speakerId): \(startTime) - \(endTime) (\(String(format: "%.1f", duration))s)"
            )
        }
    }

    static func saveResults(_ result: ProcessingResult, to file: String) async throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(result)
        try data.write(to: URL(fileURLWithPath: file))
    }

    static func saveBenchmarkResults(_ summary: BenchmarkSummary, to file: String) async throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(summary)
        try data.write(to: URL(fileURLWithPath: file))
    }

    static func formatTime(_ seconds: Float) -> String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return String(format: "%02d:%02d", minutes, remainingSeconds)
    }

    static func printBenchmarkResults(
        _ results: [BenchmarkResult], avgDER: Float, avgJER: Float, dataset: String
    ) {
        print("\n🏆 \(dataset) Benchmark Results")
        let separator = String(repeating: "=", count: 75)
        print("\(separator)")

        // Print table header
        print("│ Meeting ID    │  DER   │  JER   │  RTF   │ Duration │ Speakers │")
        let headerSep = "├───────────────┼────────┼────────┼────────┼──────────┼──────────┤"
        print("\(headerSep)")

        // Print individual results
        for result in results.sorted(by: { $0.meetingId < $1.meetingId }) {
            let meetingDisplay = String(result.meetingId.prefix(13)).padding(
                toLength: 13, withPad: " ", startingAt: 0)
            let derStr = String(format: "%.1f%%", result.der).padding(
                toLength: 6, withPad: " ", startingAt: 0)
            let jerStr = String(format: "%.1f%%", result.jer).padding(
                toLength: 6, withPad: " ", startingAt: 0)
            let rtfStr = String(format: "%.2fx", result.realTimeFactor).padding(
                toLength: 6, withPad: " ", startingAt: 0)
            let durationStr = formatTime(result.durationSeconds).padding(
                toLength: 8, withPad: " ", startingAt: 0)
            let speakerStr = String(result.speakerCount).padding(
                toLength: 8, withPad: " ", startingAt: 0)

            print(
                "│ \(meetingDisplay) │ \(derStr) │ \(jerStr) │ \(rtfStr) │ \(durationStr) │ \(speakerStr) │"
            )
        }

        // Print summary section
        let midSep = "├───────────────┼────────┼────────┼────────┼──────────┼──────────┤"
        print("\(midSep)")

        let avgDerStr = String(format: "%.1f%%", avgDER).padding(
            toLength: 6, withPad: " ", startingAt: 0)
        let avgJerStr = String(format: "%.1f%%", avgJER).padding(
            toLength: 6, withPad: " ", startingAt: 0)
        let avgRtf = results.reduce(0.0) { $0 + $1.realTimeFactor } / Float(results.count)
        let avgRtfStr = String(format: "%.2fx", avgRtf).padding(
            toLength: 6, withPad: " ", startingAt: 0)
        let totalDuration = results.reduce(0.0) { $0 + $1.durationSeconds }
        let avgDurationStr = formatTime(totalDuration).padding(
            toLength: 8, withPad: " ", startingAt: 0)
        let avgSpeakers = results.reduce(0) { $0 + $1.speakerCount } / results.count
        let avgSpeakerStr = String(format: "%.1f", Float(avgSpeakers)).padding(
            toLength: 8, withPad: " ", startingAt: 0)

        print(
            "│ AVERAGE       │ \(avgDerStr) │ \(avgJerStr) │ \(avgRtfStr) │ \(avgDurationStr) │ \(avgSpeakerStr) │"
        )
        let bottomSep = "└───────────────┴────────┴────────┴────────┴──────────┴──────────┘"
        print("\(bottomSep)")

        // Print statistics
        if results.count > 1 {
            let derValues = results.map { $0.der }
            let jerValues = results.map { $0.jer }
            let derStdDev = calculateStandardDeviation(derValues)
            let jerStdDev = calculateStandardDeviation(jerValues)

            print("\n📊 Statistical Analysis:")
            print(
                "   DER: \(String(format: "%.1f", avgDER))% ± \(String(format: "%.1f", derStdDev))% (min: \(String(format: "%.1f", derValues.min()!))%, max: \(String(format: "%.1f", derValues.max()!))%)"
            )
            print(
                "   JER: \(String(format: "%.1f", avgJER))% ± \(String(format: "%.1f", jerStdDev))% (min: \(String(format: "%.1f", jerValues.min()!))%, max: \(String(format: "%.1f", jerValues.max()!))%)"
            )
            print("   Files Processed: \(results.count)")
            print(
                "   Total Audio: \(formatTime(totalDuration)) (\(String(format: "%.1f", totalDuration/60)) minutes)"
            )
        }

        // Print research comparison
        print("\n📝 Research Comparison:")
        print("   Your Results:          \(String(format: "%.1f", avgDER))% DER")
        print("   Powerset BCE (2023):   18.5% DER")
        print("   EEND (2019):           25.3% DER")
        print("   x-vector clustering:   28.7% DER")

        if dataset == "AMI-IHM" {
            print("   Note: IHM typically achieves 5-10% lower DER than SDM")
        }

        // Performance assessment
        if avgDER < 20.0 {
            print("\n🎉 EXCELLENT: Competitive with state-of-the-art research!")
        } else if avgDER < 30.0 {
            print("\n✅ GOOD: Above research baseline, room for optimization")
        } else if avgDER < 50.0 {
            print("\n⚠️  NEEDS WORK: Significant room for parameter tuning")
        } else {
            print("\n🚨 CRITICAL: Check configuration - results much worse than expected")
        }
    }

    static func calculateStandardDeviation(_ values: [Float]) -> Float {
        guard values.count > 1 else { return 0.0 }
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Float(values.count - 1)
        return sqrt(variance)
    }

    // MARK: - Dataset Downloading

    enum AMIVariant: String, CaseIterable {
        case sdm = "sdm"  // Single Distant Microphone (Mix-Headset.wav)
        case ihm = "ihm"  // Individual Headset Microphones (Headset-0.wav)

        var displayName: String {
            switch self {
            case .sdm: return "Single Distant Microphone"
            case .ihm: return "Individual Headset Microphones"
            }
        }

        var filePattern: String {
            switch self {
            case .sdm: return "Mix-Headset.wav"
            case .ihm: return "Headset-0.wav"
            }
        }
    }

    static func downloadAMIDataset(variant: AMIVariant, force: Bool) async {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let baseDir = homeDir.appendingPathComponent("FluidAudioSwiftDatasets")
        let amiDir = baseDir.appendingPathComponent("ami_official")
        let variantDir = amiDir.appendingPathComponent(variant.rawValue)

        // Create directories if needed
        do {
            try FileManager.default.createDirectory(
                at: variantDir, withIntermediateDirectories: true)
        } catch {
            print("❌ Failed to create directory: \(error)")
            return
        }

        print("📥 Downloading AMI \(variant.displayName) dataset...")
        print("   Target directory: \(variantDir.path)")

        // Core AMI test set - smaller subset for initial benchmarking
        let commonMeetings = [
            "ES2002a",
            "ES2003a",
            "ES2004a",
            "ES2005a",
            "IS1000a",
            "IS1001a",
            "IS1002b",
            "TS3003a",
            "TS3004a",
        ]

        var downloadedFiles = 0
        var skippedFiles = 0

        for meetingId in commonMeetings {
            let fileName = "\(meetingId).\(variant.filePattern)"
            let filePath = variantDir.appendingPathComponent(fileName)

            // Skip if file exists and not forcing download
            if !force && FileManager.default.fileExists(atPath: filePath.path) {
                print("   ⏭️ Skipping \(fileName) (already exists)")
                skippedFiles += 1
                continue
            }

            // Try to download from AMI corpus mirror
            let success = await downloadAMIFile(
                meetingId: meetingId,
                variant: variant,
                outputPath: filePath
            )

            if success {
                downloadedFiles += 1
                print("   ✅ Downloaded \(fileName)")
            } else {
                print("   ❌ Failed to download \(fileName)")
            }
        }

        print("🎉 AMI \(variant.displayName) download completed")
        print("   Downloaded: \(downloadedFiles) files")
        print("   Skipped: \(skippedFiles) files")
        print("   Total files: \(downloadedFiles + skippedFiles)/\(commonMeetings.count)")

        if downloadedFiles == 0 && skippedFiles == 0 {
            print("⚠️ No files were downloaded. You may need to download manually from:")
            print("   https://groups.inf.ed.ac.uk/ami/download/")
        }
    }

    static func downloadAMIFile(meetingId: String, variant: AMIVariant, outputPath: URL) async
        -> Bool
    {
        // Try multiple URL patterns - the AMI corpus mirror structure has some variations
        let baseURLs = [
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus",  // Double slash pattern (from user's working example)
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus",  // Single slash pattern
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus",  // Alternative with extra slash
        ]

        for (_, baseURL) in baseURLs.enumerated() {
            let urlString = "\(baseURL)/\(meetingId)/audio/\(meetingId).\(variant.filePattern)"

            guard let url = URL(string: urlString) else {
                print("     ⚠️ Invalid URL: \(urlString)")
                continue
            }

            do {
                print("     📥 Downloading from: \(urlString)")
                let (data, response) = try await URLSession.shared.data(from: url)

                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        try data.write(to: outputPath)

                        // Verify it's a valid audio file
                        if await isValidAudioFile(outputPath) {
                            let fileSizeMB = Double(data.count) / (1024 * 1024)
                            print(
                                "     ✅ Downloaded \(String(format: "%.1f", fileSizeMB)) MB")
                            return true
                        } else {
                            print("     ⚠️ Downloaded file is not valid audio")
                            try? FileManager.default.removeItem(at: outputPath)
                            // Try next URL
                            continue
                        }
                    } else if httpResponse.statusCode == 404 {
                        print("     ⚠️ File not found (HTTP 404) - trying next URL...")
                        continue
                    } else {
                        print(
                            "     ⚠️ HTTP error: \(httpResponse.statusCode) - trying next URL...")
                        continue
                    }
                }
            } catch {
                print(
                    "     ⚠️ Download error: \(error.localizedDescription) - trying next URL...")
                continue
            }
        }

        print("     ❌ Failed to download from all available URLs")
        return false
    }

    static func isValidAudioFile(_ url: URL) async -> Bool {
        do {
            let _ = try AVAudioFile(forReading: url)
            return true
        } catch {
            return false
        }
    }

    // MARK: - AMI Annotation Loading

    /// Load AMI ground truth annotations for a specific meeting
    static func loadAMIGroundTruth(for meetingId: String, duration: Float) async
        -> [TimedSpeakerSegment]
    {
        // Try to find the AMI annotations directory in several possible locations
        let possiblePaths = [
            // Current working directory
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(
                "Tests/ami_public_1.6.2"),
            // Relative to source file
            URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Tests/ami_public_1.6.2"),
            // Home directory
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
                "code/FluidAudioSwift/Tests/ami_public_1.6.2"),
        ]

        var amiDir: URL?
        for path in possiblePaths {
            let segmentsDir = path.appendingPathComponent("segments")
            let meetingsFile = path.appendingPathComponent("corpusResources/meetings.xml")

            if FileManager.default.fileExists(atPath: segmentsDir.path)
                && FileManager.default.fileExists(atPath: meetingsFile.path)
            {
                amiDir = path
                break
            }
        }

        guard let validAmiDir = amiDir else {
            print("   ⚠️ AMI annotations not found in any expected location")
            print(
                "      Using simplified placeholder - real annotations expected in Tests/ami_public_1.6.2/"
            )
            return Self.generateSimplifiedGroundTruth(duration: duration, speakerCount: 4)
        }

        let segmentsDir = validAmiDir.appendingPathComponent("segments")
        let meetingsFile = validAmiDir.appendingPathComponent("corpusResources/meetings.xml")

        print("   📖 Loading AMI annotations for meeting: \(meetingId)")

        do {
            let parser = AMIAnnotationParser()

            // Get speaker mapping for this meeting
            guard
                let speakerMapping = try parser.parseSpeakerMapping(
                    for: meetingId, from: meetingsFile)
            else {
                print(
                    "      ⚠️ No speaker mapping found for meeting: \(meetingId), using placeholder")
                return Self.generateSimplifiedGroundTruth(duration: duration, speakerCount: 4)
            }

            print(
                "      Speaker mapping: A=\(speakerMapping.speakerA), B=\(speakerMapping.speakerB), C=\(speakerMapping.speakerC), D=\(speakerMapping.speakerD)"
            )

            var allSegments: [TimedSpeakerSegment] = []

            // Parse segments for each speaker (A, B, C, D)
            for speakerCode in ["A", "B", "C", "D"] {
                let segmentFile = segmentsDir.appendingPathComponent(
                    "\(meetingId).\(speakerCode).segments.xml")

                if FileManager.default.fileExists(atPath: segmentFile.path) {
                    let segments = try parser.parseSegmentsFile(segmentFile)

                    // Map to TimedSpeakerSegment with real participant ID
                    guard let participantId = speakerMapping.participantId(for: speakerCode) else {
                        continue
                    }

                    for segment in segments {
                        // Filter out very short segments (< 0.5 seconds) as done in research
                        guard segment.duration >= 0.5 else { continue }

                        let timedSegment = TimedSpeakerSegment(
                            speakerId: participantId,  // Use real AMI participant ID
                            embedding: Self.generatePlaceholderEmbedding(for: participantId),
                            startTimeSeconds: Float(segment.startTime),
                            endTimeSeconds: Float(segment.endTime),
                            qualityScore: 1.0
                        )

                        allSegments.append(timedSegment)
                    }

                    print(
                        "      Loaded \(segments.count) segments for speaker \(speakerCode) (\(participantId))"
                    )
                }
            }

            // Sort by start time
            allSegments.sort { $0.startTimeSeconds < $1.startTimeSeconds }

            print("      Total segments loaded: \(allSegments.count)")
            return allSegments

        } catch {
            print("      ❌ Failed to parse AMI annotations: \(error)")
            print("      Using simplified placeholder instead")
            return Self.generateSimplifiedGroundTruth(duration: duration, speakerCount: 4)
        }
    }

    /// Generate consistent placeholder embeddings for each speaker
    static func generatePlaceholderEmbedding(for participantId: String) -> [Float] {
        // Generate a consistent embedding based on participant ID
        let hash = participantId.hashValue
        let seed = abs(hash) % 1000

        var embedding: [Float] = []
        for i in 0..<512 {  // Match expected embedding size
            let value = Float(sin(Double(seed + i * 37))) * 0.5 + 0.5
            embedding.append(value)
        }
        return embedding
    }
}

// MARK: - Data Structures

struct ProcessingResult: Codable {
    let audioFile: String
    let durationSeconds: Float
    let processingTimeSeconds: TimeInterval
    let realTimeFactor: Float
    let segments: [TimedSpeakerSegment]
    let speakerCount: Int
    let config: DiarizerConfig
    let timestamp: Date

    init(
        audioFile: String, durationSeconds: Float, processingTimeSeconds: TimeInterval,
        realTimeFactor: Float, segments: [TimedSpeakerSegment], speakerCount: Int,
        config: DiarizerConfig
    ) {
        self.audioFile = audioFile
        self.durationSeconds = durationSeconds
        self.processingTimeSeconds = processingTimeSeconds
        self.realTimeFactor = realTimeFactor
        self.segments = segments
        self.speakerCount = speakerCount
        self.config = config
        self.timestamp = Date()
    }
}

struct BenchmarkResult: Codable {
    let meetingId: String
    let durationSeconds: Float
    let processingTimeSeconds: TimeInterval
    let realTimeFactor: Float
    let der: Float
    let jer: Float
    let segments: [TimedSpeakerSegment]
    let speakerCount: Int
}

struct BenchmarkSummary: Codable {
    let dataset: String
    let averageDER: Float
    let averageJER: Float
    let processedFiles: Int
    let totalFiles: Int
    let results: [BenchmarkResult]
    let timestamp: Date

    init(
        dataset: String, averageDER: Float, averageJER: Float, processedFiles: Int, totalFiles: Int,
        results: [BenchmarkResult]
    ) {
        self.dataset = dataset
        self.averageDER = averageDER
        self.averageJER = averageJER
        self.processedFiles = processedFiles
        self.totalFiles = totalFiles
        self.results = results
        self.timestamp = Date()
    }
}

struct DiarizationMetrics {
    let der: Float
    let jer: Float
    let missRate: Float
    let falseAlarmRate: Float
    let speakerErrorRate: Float
}

// Make DiarizerConfig Codable for output
extension DiarizerConfig: Codable {
    enum CodingKeys: String, CodingKey {
        case clusteringThreshold
        case minDurationOn
        case minDurationOff
        case numClusters
        case minActivityThreshold
        case debugMode
        case modelCacheDirectory
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(clusteringThreshold, forKey: .clusteringThreshold)
        try container.encode(minDurationOn, forKey: .minDurationOn)
        try container.encode(minDurationOff, forKey: .minDurationOff)
        try container.encode(numClusters, forKey: .numClusters)
        try container.encode(minActivityThreshold, forKey: .minActivityThreshold)
        try container.encode(debugMode, forKey: .debugMode)
        try container.encodeIfPresent(modelCacheDirectory, forKey: .modelCacheDirectory)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let clusteringThreshold = try container.decode(Float.self, forKey: .clusteringThreshold)
        let minDurationOn = try container.decode(Float.self, forKey: .minDurationOn)
        let minDurationOff = try container.decode(Float.self, forKey: .minDurationOff)
        let numClusters = try container.decode(Int.self, forKey: .numClusters)
        let minActivityThreshold = try container.decode(Float.self, forKey: .minActivityThreshold)
        let debugMode = try container.decode(Bool.self, forKey: .debugMode)
        let modelCacheDirectory = try container.decodeIfPresent(
            URL.self, forKey: .modelCacheDirectory)

        self.init(
            clusteringThreshold: clusteringThreshold,
            minDurationOn: minDurationOn,
            minDurationOff: minDurationOff,
            numClusters: numClusters,
            minActivityThreshold: minActivityThreshold,
            debugMode: debugMode,
            modelCacheDirectory: modelCacheDirectory
        )
    }
}

// Make TimedSpeakerSegment Codable for CLI output
extension TimedSpeakerSegment: Codable {
    enum CodingKeys: String, CodingKey {
        case speakerId
        case embedding
        case startTimeSeconds
        case endTimeSeconds
        case qualityScore
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(speakerId, forKey: .speakerId)
        try container.encode(embedding, forKey: .embedding)
        try container.encode(startTimeSeconds, forKey: .startTimeSeconds)
        try container.encode(endTimeSeconds, forKey: .endTimeSeconds)
        try container.encode(qualityScore, forKey: .qualityScore)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let speakerId = try container.decode(String.self, forKey: .speakerId)
        let embedding = try container.decode([Float].self, forKey: .embedding)
        let startTimeSeconds = try container.decode(Float.self, forKey: .startTimeSeconds)
        let endTimeSeconds = try container.decode(Float.self, forKey: .endTimeSeconds)
        let qualityScore = try container.decode(Float.self, forKey: .qualityScore)

        self.init(
            speakerId: speakerId,
            embedding: embedding,
            startTimeSeconds: startTimeSeconds,
            endTimeSeconds: endTimeSeconds,
            qualityScore: qualityScore
        )
    }
}

// MARK: - AMI Annotation Parser

/// Represents a single AMI speaker segment from NXT format
struct AMISpeakerSegment {
    let segmentId: String  // e.g., "EN2001a.sync.4"
    let participantId: String  // e.g., "FEE005" (mapped from A/B/C/D)
    let startTime: Double  // Start time in seconds
    let endTime: Double  // End time in seconds

    var duration: Double {
        return endTime - startTime
    }
}

/// Maps AMI speaker codes (A/B/C/D) to real participant IDs
struct AMISpeakerMapping {
    let meetingId: String
    let speakerA: String  // e.g., "MEE006"
    let speakerB: String  // e.g., "FEE005"
    let speakerC: String  // e.g., "MEE007"
    let speakerD: String  // e.g., "MEE008"

    func participantId(for speakerCode: String) -> String? {
        switch speakerCode.uppercased() {
        case "A": return speakerA
        case "B": return speakerB
        case "C": return speakerC
        case "D": return speakerD
        default: return nil
        }
    }
}

/// Parser for AMI NXT XML annotation files
class AMIAnnotationParser: NSObject {

    /// Parse segments.xml file and return speaker segments
    func parseSegmentsFile(_ xmlFile: URL) throws -> [AMISpeakerSegment] {
        let data = try Data(contentsOf: xmlFile)

        // Extract speaker code from filename (e.g., "EN2001a.A.segments.xml" -> "A")
        let speakerCode = extractSpeakerCodeFromFilename(xmlFile.lastPathComponent)

        let parser = XMLParser(data: data)
        let delegate = AMISegmentsXMLDelegate(speakerCode: speakerCode)
        parser.delegate = delegate

        guard parser.parse() else {
            throw NSError(
                domain: "AMIParser", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Failed to parse XML file: \(xmlFile.lastPathComponent)"
                ])
        }

        if let error = delegate.parsingError {
            throw error
        }

        return delegate.segments
    }

    /// Extract speaker code from AMI filename
    private func extractSpeakerCodeFromFilename(_ filename: String) -> String {
        // Filename format: "EN2001a.A.segments.xml" -> extract "A"
        let components = filename.components(separatedBy: ".")
        if components.count >= 3 {
            return components[1]  // The speaker code is the second component
        }
        return "UNKNOWN"
    }

    /// Parse meetings.xml to get speaker mappings for a specific meeting
    func parseSpeakerMapping(for meetingId: String, from meetingsFile: URL) throws
        -> AMISpeakerMapping?
    {
        let data = try Data(contentsOf: meetingsFile)

        let parser = XMLParser(data: data)
        let delegate = AMIMeetingsXMLDelegate(targetMeetingId: meetingId)
        parser.delegate = delegate

        guard parser.parse() else {
            throw NSError(
                domain: "AMIParser", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to parse meetings.xml"])
        }

        if let error = delegate.parsingError {
            throw error
        }

        return delegate.speakerMapping
    }
}

/// XML parser delegate for AMI segments files
private class AMISegmentsXMLDelegate: NSObject, XMLParserDelegate {
    var segments: [AMISpeakerSegment] = []
    var parsingError: Error?

    private let speakerCode: String

    init(speakerCode: String) {
        self.speakerCode = speakerCode
    }

    func parser(
        _ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?,
        qualifiedName qName: String?, attributes attributeDict: [String: String] = [:]
    ) {

        if elementName == "segment" {
            // Extract segment attributes
            guard let segmentId = attributeDict["nite:id"],
                let startTimeStr = attributeDict["transcriber_start"],
                let endTimeStr = attributeDict["transcriber_end"],
                let startTime = Double(startTimeStr),
                let endTime = Double(endTimeStr)
            else {
                return  // Skip invalid segments
            }

            let segment = AMISpeakerSegment(
                segmentId: segmentId,
                participantId: speakerCode,  // Use speaker code from filename
                startTime: startTime,
                endTime: endTime
            )

            segments.append(segment)
        }
    }

    func parser(_ parser: XMLParser, parseErrorOccurred parseError: Error) {
        parsingError = parseError
    }
}

/// XML parser delegate for AMI meetings.xml file
private class AMIMeetingsXMLDelegate: NSObject, XMLParserDelegate {
    let targetMeetingId: String
    var speakerMapping: AMISpeakerMapping?
    var parsingError: Error?

    private var currentMeetingId: String?
    private var speakersInCurrentMeeting: [String: String] = [:]  // agent code -> global_name
    private var isInTargetMeeting = false

    init(targetMeetingId: String) {
        self.targetMeetingId = targetMeetingId
    }

    func parser(
        _ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?,
        qualifiedName qName: String?, attributes attributeDict: [String: String] = [:]
    ) {

        if elementName == "meeting" {
            currentMeetingId = attributeDict["observation"]
            isInTargetMeeting = (currentMeetingId == targetMeetingId)
            speakersInCurrentMeeting.removeAll()
        }

        if elementName == "speaker" && isInTargetMeeting {
            guard let nxtAgent = attributeDict["nxt_agent"],
                let globalName = attributeDict["global_name"]
            else {
                return
            }
            speakersInCurrentMeeting[nxtAgent] = globalName
        }
    }

    func parser(
        _ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?,
        qualifiedName qName: String?
    ) {
        if elementName == "meeting" && isInTargetMeeting {
            // Create the speaker mapping for this meeting
            if let meetingId = currentMeetingId {
                speakerMapping = AMISpeakerMapping(
                    meetingId: meetingId,
                    speakerA: speakersInCurrentMeeting["A"] ?? "UNKNOWN",
                    speakerB: speakersInCurrentMeeting["B"] ?? "UNKNOWN",
                    speakerC: speakersInCurrentMeeting["C"] ?? "UNKNOWN",
                    speakerD: speakersInCurrentMeeting["D"] ?? "UNKNOWN"
                )
            }
            isInTargetMeeting = false
        }
    }

    func parser(_ parser: XMLParser, parseErrorOccurred parseError: Error) {
        parsingError = parseError
    }
}
