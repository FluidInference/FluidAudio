#if os(macOS)
    import AVFoundation
    import FluidAudio
    import Foundation

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
            case "vad-benchmark":
                await VadBenchmark.runVadBenchmark(arguments: Array(arguments.dropFirst(2)))
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
                FluidAudio Diarization CLI

                USAGE:
                    fluidaudio <command> [options]

                COMMANDS:
                    benchmark       Run AMI SDM benchmark evaluation with real annotations
                    vad-benchmark   Run VAD benchmark with real audio files
                    process         Process a single audio file
                    download        Download datasets for benchmarking
                    help            Show this help message

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
                    --iterations <num>      Run multiple iterations for consistency testing [default: 1]
                    --der-threshold <float> Custom DER threshold for pass/fail (job exits with failure if exceeded)
                    --jer-threshold <float> Custom JER threshold for pass/fail (job exits with failure if exceeded)
                    --rtf-threshold <float> Custom RTF threshold for pass/fail (job exits with failure if exceeded)

                NOTE: Benchmark now uses real AMI manual annotations from Tests/ami_public_1.6.2/
                      If annotations are not found, falls back to simplified placeholder.

                VAD-BENCHMARK OPTIONS:
                    --threshold <float>     VAD threshold 0.0-1.0 [default: 0.3]
                    --dataset <name>        Dataset to use: mini50 (default), mini100 [default: mini50]
                    --num-files <int>       Limit number of test files (default: all files)
                    --output <file>         Output results to JSON file [default: vad_benchmark_results.json]

                PROCESS OPTIONS:
                    <audio-file>         Audio file to process (.wav, .m4a, .mp3)
                    --output <file>      Output results to JSON file [default: stdout]
                    --threshold <float>  Clustering threshold 0.0-1.0 [default: 0.7]
                    --debug             Enable debug mode

                DOWNLOAD OPTIONS:
                    --dataset <name>     Dataset to download (ami-sdm, ami-ihm, ami-annotations, vad, vad-mini50, vad-mini100, all) [default: all]
                    --force             Force re-download even if files exist

                EXAMPLES:
                    # Download AMI datasets
                    swift run fluidaudio download --dataset ami-sdm

                    # Download VAD dataset from Hugging Face
                    swift run fluidaudio download --dataset vad

                    # Run AMI SDM benchmark with auto-download
                    swift run fluidaudio benchmark --auto-download

                    # Run benchmark with custom threshold and save results
                    swift run fluidaudio benchmark --threshold 0.8 --output results.json
                    
                    # Run benchmark with custom DER threshold for CI (fails if DER > 25%)
                    swift run fluidaudio benchmark --der-threshold 25.0 --auto-download
                    
                    # Run benchmark with strict thresholds (DER < 20%, JER < 23%, RTF < 0.1x)
                    swift run fluidaudio benchmark --der-threshold 20.0 --jer-threshold 23.0 --rtf-threshold 0.1 --auto-download

                    # Run VAD benchmark with mini50 dataset (default, all files)
                    swift run fluidaudio vad-benchmark
                    swift run fluidaudio vad-benchmark --threshold 0.5

                    # Run VAD benchmark with mini50 dataset (all files)
                    swift run fluidaudio vad-benchmark --dataset mini50

                    # Run VAD benchmark with custom threshold and dataset
                    swift run fluidaudio vad-benchmark --threshold 0.45 --dataset mini50

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
            var disableVad = false
            var iterations = 1
            var derThreshold: Float?
            var jerThreshold: Float?
            var rtfThreshold: Float?

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
                case "--disable-vad":
                    disableVad = true
                case "--iterations":
                    if i + 1 < arguments.count {
                        iterations = Int(arguments[i + 1]) ?? 1
                        i += 1
                    }
                case "--der-threshold":
                    if i + 1 < arguments.count {
                        derThreshold = Float(arguments[i + 1])
                        i += 1
                    }
                case "--jer-threshold":
                    if i + 1 < arguments.count {
                        jerThreshold = Float(arguments[i + 1])
                        i += 1
                    }
                case "--rtf-threshold":
                    if i + 1 < arguments.count {
                        rtfThreshold = Float(arguments[i + 1])
                        i += 1
                    }
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
            print("   VAD: \(disableVad ? "disabled" : "enabled")")
            if iterations > 1 {
                print("   Iterations: \(iterations) (consistency testing)")
            }

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
            let assessment: PerformanceAssessment
            switch dataset.lowercased() {
            case "ami-sdm":
                assessment = await BenchmarkRunner.runAMISDMBenchmark(
                    manager: manager, config: config, outputFile: outputFile,
                    autoDownload: autoDownload,
                    singleFile: singleFile, iterations: iterations, 
                    customThresholds: (derThreshold, jerThreshold, rtfThreshold))
            case "ami-ihm":
                assessment = await BenchmarkRunner.runAMIIHMBenchmark(
                    manager: manager, config: config, outputFile: outputFile,
                    autoDownload: autoDownload,
                    singleFile: singleFile, iterations: iterations, 
                    customThresholds: (derThreshold, jerThreshold, rtfThreshold))
            default:
                print("❌ Unsupported dataset: \(dataset)")
                print("💡 Supported datasets: ami-sdm, ami-ihm")
                exit(1)
            }

            let benchmarkElapsed = Date().timeIntervalSince(benchmarkStartTime)
            print(
                "\n⏱️ Total benchmark execution time: \(String(format: "%.1f", benchmarkElapsed)) seconds"
            )
            
            // Exit with appropriate code based on performance assessment
            if assessment.exitCode != 0 {
                print("\n❌ Benchmark failed to meet performance standards")
                print("💡 Exit code: \(assessment.exitCode)")
                exit(assessment.exitCode)
            } else {
                print("\n✅ Benchmark completed successfully")
            }
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
                await DatasetDownloader.downloadAMIDataset(variant: .sdm, force: forceDownload)
            case "ami-ihm":
                await DatasetDownloader.downloadAMIDataset(variant: .ihm, force: forceDownload)
            case "ami-annotations":
                await DatasetDownloader.downloadAMIAnnotations(force: forceDownload)
            case "vad":
                await DatasetDownloader.downloadVadDataset(force: forceDownload, dataset: "mini100")  // Default to mini100 for more test data
            case "vad-mini50":
                await DatasetDownloader.downloadVadDataset(force: forceDownload, dataset: "mini50")
            case "vad-mini100":
                await DatasetDownloader.downloadVadDataset(force: forceDownload, dataset: "mini100")
            case "all":
                await DatasetDownloader.downloadAMIDataset(variant: .sdm, force: forceDownload)
                await DatasetDownloader.downloadAMIDataset(variant: .ihm, force: forceDownload)
                await DatasetDownloader.downloadVadDataset(force: forceDownload, dataset: "mini100")
            default:
                print("❌ Unsupported dataset: \(dataset)")
                print("💡 Supported datasets: ami-sdm, ami-ihm, ami-annotations, vad, vad-mini50, vad-mini100, all")
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
    }

    // Main execution
    await DiarizationCLI.main()

#endif
