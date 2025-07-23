import AVFoundation
import FluidAudio

/// Handler for the 'diarization-benchmark' command - runs diarization benchmarks
enum DiarizationBenchmark {
    static func run(arguments: [String]) async {
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
            printUsage()
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
    
    private static func printUsage() {
        print(
            """
            
            Diarization Benchmark Command Usage:
                fluidaudio diarization-benchmark [options]
            
            Options:
                --dataset <name>           Dataset to benchmark (default: ami-sdm)
                --threshold <float>        Clustering threshold (default: 0.7)
                --min-duration-on <float>  Minimum speech duration (default: 1.0)
                --min-duration-off <float> Minimum silence duration (default: 0.5)
                --min-activity <float>     Minimum activity threshold (default: 10.0)
                --single-file <path>       Process single file instead of full dataset
                --debug                    Enable debug mode
                --output <file>           Save results to file
                --auto-download           Auto-download missing datasets
                --disable-vad             Disable VAD preprocessing
                --iterations <int>        Number of iterations (default: 1)
                --der-threshold <float>   Custom DER threshold
                --jer-threshold <float>   Custom JER threshold
                --rtf-threshold <float>   Custom RTF threshold
            
            Supported datasets:
                ami-sdm    AMI SDM dataset
                ami-ihm    AMI IHM dataset
            
            Example:
                fluidaudio diarization-benchmark --dataset ami-sdm --threshold 0.5 --debug
            """
        )
    }
}