#if os(macOS)
import FluidAudio
import Foundation

// MARK: - Buckeye Manifest Types

struct BuckeyeManifest: Codable {
    let dataset: String
    let speakers: Int
    let totalSegments: Int
    let totalWords: Int
    let samples: [BuckeyeSample]

    enum CodingKeys: String, CodingKey {
        case dataset, speakers
        case totalSegments = "total_segments"
        case totalWords = "total_words"
        case samples
    }
}

struct BuckeyeSample: Codable {
    let id: String
    let speaker: String
    let audio: String
    let transcript: String
    let durationS: Double
    let numWords: Int
    let words: [BuckeyeWord]

    enum CodingKeys: String, CodingKey {
        case id, speaker, audio, transcript
        case durationS = "duration_s"
        case numWords = "num_words"
        case words
    }
}

struct BuckeyeWord: Codable {
    let word: String
    let startMs: Double
    let endMs: Double

    enum CodingKeys: String, CodingKey {
        case word
        case startMs = "start_ms"
        case endMs = "end_ms"
    }
}

// MARK: - PyTorch Reference Types

struct PyTorchReference: Codable {
    let modelId: String
    let dataset: String
    let numSamples: Int
    let totalPytorchLatencyMs: Double
    let samples: [PyTorchSampleResult]

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case dataset
        case numSamples = "num_samples"
        case totalPytorchLatencyMs = "total_pytorch_latency_ms"
        case samples
    }
}

struct PyTorchSampleResult: Codable {
    let id: String
    let pytorchAlignments: [PyTorchAlignment]
    let pytorchLatencyMs: Double

    enum CodingKeys: String, CodingKey {
        case id
        case pytorchAlignments = "pytorch_alignments"
        case pytorchLatencyMs = "pytorch_latency_ms"
    }
}

struct PyTorchAlignment: Codable {
    let text: String
    let startTimeMs: Double
    let endTimeMs: Double

    enum CodingKeys: String, CodingKey {
        case text
        case startTimeMs = "start_time_ms"
        case endTimeMs = "end_time_ms"
    }
}

// MARK: - Benchmark Command

enum AlignBenchmark {
    private static let logger = AppLogger(category: "AlignBenchmark")

    static func run(arguments: [String]) async {
        var numFiles = 1000
        var modelDir: String?
        var pytorchRefPath: String?
        var outputPath: String?
        var autoDownload = false
        var datasetDir: String?

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--num-files":
                if i + 1 < arguments.count {
                    numFiles = Int(arguments[i + 1]) ?? 1000
                    i += 1
                }
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            case "--pytorch-ref":
                if i + 1 < arguments.count {
                    pytorchRefPath = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputPath = arguments[i + 1]
                    i += 1
                }
            case "--auto-download":
                autoDownload = true
            case "--dataset-dir":
                if i + 1 < arguments.count {
                    datasetDir = arguments[i + 1]
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        if #available(macOS 14, iOS 17, *) {
            await runBenchmark(
                numFiles: numFiles,
                modelDir: modelDir,
                pytorchRefPath: pytorchRefPath,
                outputPath: outputPath,
                autoDownload: autoDownload,
                datasetDir: datasetDir
            )
        } else {
            logger.error("ForcedAligner requires macOS 14 or later")
        }
    }

    @available(macOS 14, iOS 17, *)
    private static func runBenchmark(
        numFiles: Int,
        modelDir: String?,
        pytorchRefPath: String?,
        outputPath: String?,
        autoDownload: Bool,
        datasetDir: String?
    ) async {
        // 1. Resolve dataset directory
        let buckeyeDir: URL
        if let dir = datasetDir {
            buckeyeDir = URL(fileURLWithPath: dir)
        } else {
            buckeyeDir = DatasetDownloader.getBuckeyeDatasetDirectory()
        }

        let manifestPath = buckeyeDir.appendingPathComponent("manifest.json")

        // Auto-download if needed
        if !FileManager.default.fileExists(atPath: manifestPath.path) {
            if autoDownload {
                logger.info("Buckeye dataset not found, downloading...")
                await DatasetDownloader.downloadBuckeyeDataset(force: false)
            } else {
                logger.error(
                    "Buckeye dataset not found at \(buckeyeDir.path). "
                        + "Use --auto-download or --dataset-dir, or run: fluidaudio download --dataset buckeye"
                )
                return
            }
        }

        // 2. Load manifest
        guard let manifestData = try? Data(contentsOf: manifestPath),
            let manifest = try? JSONDecoder().decode(BuckeyeManifest.self, from: manifestData)
        else {
            logger.error("Failed to parse manifest.json at \(manifestPath.path)")
            return
        }

        let samples = Array(manifest.samples.prefix(numFiles))
        logger.info(
            "Loaded \(samples.count) samples from Buckeye (\(manifest.totalSegments) total)")

        // 3. Load PyTorch reference if provided
        var pytorchLookup: [String: PyTorchSampleResult] = [:]
        if let refPath = pytorchRefPath {
            do {
                let refData = try Data(contentsOf: URL(fileURLWithPath: refPath))
                let ref = try JSONDecoder().decode(PyTorchReference.self, from: refData)
                for sample in ref.samples {
                    pytorchLookup[sample.id] = sample
                }
                logger.info("Loaded PyTorch reference: \(ref.numSamples) samples")
            } catch {
                logger.error("Failed to load PyTorch reference: \(error)")
                return
            }
        }

        // 4. Load ForcedAligner models
        let manager = ForcedAlignerManager()
        do {
            if let dir = modelDir {
                logger.info("Loading ForcedAligner models from: \(dir)")
                try await manager.loadModels(from: URL(fileURLWithPath: dir))
            } else {
                logger.info("Downloading ForcedAligner models from HuggingFace...")
                try await manager.downloadAndLoadModels()
            }
        } catch {
            logger.error("Failed to load ForcedAligner: \(error)")
            return
        }
        logger.info("ForcedAligner ready\n")

        // 5. Warmup with first sample (CoreML compilation)
        if let firstSample = samples.first {
            let warmupPath = buckeyeDir.appendingPathComponent(firstSample.audio).path
            do {
                let warmupAudio = try AudioConverter().resampleAudioFile(path: warmupPath)
                logger.info("Warmup alignment...")
                let _ = try await manager.align(audioSamples: warmupAudio, text: firstSample.transcript)
                logger.info("Warmup done\n")
            } catch {
                logger.warning("Warmup failed: \(error)")
            }
        }

        // 6. Run benchmark
        let converter = AudioConverter()
        var allBoundaryErrors: [Double] = []
        var perSampleAAS: [Double] = []
        var totalLatencyMs: Double = 0
        var totalAudioDurationS: Double = 0
        var processedCount = 0
        var failedCount = 0

        var pytorchBoundaryErrors: [Double] = []
        var coremlVsPytorchDiffs: [Double] = []
        var sampleResults: [[String: Any]] = []

        for (idx, sample) in samples.enumerated() {
            let audioPath = buckeyeDir.appendingPathComponent(sample.audio).path

            do {
                let audioSamples = try converter.resampleAudioFile(path: audioPath)
                let result = try await manager.align(
                    audioSamples: audioSamples, text: sample.transcript)

                let predicted = result.alignments
                let truth = sample.words
                let matchCount = min(predicted.count, truth.count)
                var sampleErrors: [Double] = []

                for j in 0..<matchCount {
                    let startError = abs(predicted[j].startMs - truth[j].startMs)
                    let endError = abs(predicted[j].endMs - truth[j].endMs)
                    sampleErrors.append(startError)
                    sampleErrors.append(endError)
                }

                if predicted.count != truth.count {
                    logger.warning(
                        "  [\(idx + 1)/\(samples.count)] \(sample.id): "
                            + "word count mismatch (predicted=\(predicted.count), truth=\(truth.count))"
                    )
                }

                let sampleAAS =
                    sampleErrors.isEmpty
                    ? 0 : sampleErrors.reduce(0, +) / Double(sampleErrors.count)
                allBoundaryErrors.append(contentsOf: sampleErrors)
                perSampleAAS.append(sampleAAS)
                totalLatencyMs += result.latencyMs
                totalAudioDurationS += sample.durationS
                processedCount += 1

                // PyTorch comparison
                if let pytorchResult = pytorchLookup[sample.id] {
                    let ptAlignments = pytorchResult.pytorchAlignments
                    let ptMatchCount = min(ptAlignments.count, matchCount)

                    for j in 0..<ptMatchCount {
                        if j < truth.count {
                            let ptStartError = abs(
                                ptAlignments[j].startTimeMs - truth[j].startMs)
                            let ptEndError = abs(ptAlignments[j].endTimeMs - truth[j].endMs)
                            pytorchBoundaryErrors.append(ptStartError)
                            pytorchBoundaryErrors.append(ptEndError)
                        }
                        let startDiff = abs(
                            predicted[j].startMs - ptAlignments[j].startTimeMs)
                        let endDiff = abs(predicted[j].endMs - ptAlignments[j].endTimeMs)
                        coremlVsPytorchDiffs.append(startDiff)
                        coremlVsPytorchDiffs.append(endDiff)
                    }
                }

                // Per-sample result for JSON
                var sampleDict: [String: Any] = [
                    "id": sample.id,
                    "num_words": matchCount,
                    "aas_ms": round(sampleAAS * 10) / 10,
                    "latency_ms": round(result.latencyMs * 10) / 10,
                    "duration_s": sample.durationS,
                ]
                var alignmentDicts: [[String: Any]] = []
                for j in 0..<matchCount {
                    alignmentDicts.append([
                        "word": predicted[j].word,
                        "pred_start_ms": round(predicted[j].startMs * 10) / 10,
                        "pred_end_ms": round(predicted[j].endMs * 10) / 10,
                        "truth_start_ms": round(truth[j].startMs * 10) / 10,
                        "truth_end_ms": round(truth[j].endMs * 10) / 10,
                    ])
                }
                sampleDict["alignments"] = alignmentDicts
                sampleResults.append(sampleDict)

                if (idx + 1) % 10 == 0 || idx == 0 {
                    logger.info(
                        "  [\(idx + 1)/\(samples.count)] \(sample.id): "
                            + "\(matchCount) words, \(String(format: "%.0f", result.latencyMs))ms, "
                            + "AAS=\(String(format: "%.1f", sampleAAS))ms"
                    )
                }

            } catch {
                failedCount += 1
                logger.warning(
                    "  [\(idx + 1)/\(samples.count)] \(sample.id): FAILED - \(error)")
            }
        }

        // 7. Aggregate metrics
        guard processedCount > 0 else {
            logger.error("No samples processed successfully")
            return
        }

        let overallAAS = allBoundaryErrors.reduce(0, +) / Double(allBoundaryErrors.count)
        let sortedSampleAAS = perSampleAAS.sorted()
        let medianAAS = sortedSampleAAS[sortedSampleAAS.count / 2]
        let maxAAS = sortedSampleAAS.last ?? 0

        let within20ms =
            Double(allBoundaryErrors.filter { $0 <= 20 }.count)
            / Double(allBoundaryErrors.count) * 100
        let within50ms =
            Double(allBoundaryErrors.filter { $0 <= 50 }.count)
            / Double(allBoundaryErrors.count) * 100
        let within100ms =
            Double(allBoundaryErrors.filter { $0 <= 100 }.count)
            / Double(allBoundaryErrors.count) * 100

        let avgLatency = totalLatencyMs / Double(processedCount)
        let rtfx = totalAudioDurationS / (totalLatencyMs / 1000.0)

        // 8. Print results
        print("")
        print(String(repeating: "=", count: 60))
        print("BUCKEYE FORCED ALIGNMENT BENCHMARK")
        print(String(repeating: "=", count: 60))
        print("")
        print("CoreML int8 vs Human Ground Truth:")
        print("  AAS:            \(String(format: "%.1f", overallAAS)) ms")
        print("  Median AAS:     \(String(format: "%.1f", medianAAS)) ms")
        print("  Max AAS:        \(String(format: "%.1f", maxAAS)) ms")
        print("  Within 20ms:    \(String(format: "%.1f", within20ms))%")
        print("  Within 50ms:    \(String(format: "%.1f", within50ms))%")
        print("  Within 100ms:   \(String(format: "%.1f", within100ms))%")
        print("  Boundaries:     \(allBoundaryErrors.count)")
        print("")
        print("Performance:")
        print("  Processed:      \(processedCount)/\(samples.count) segments")
        if failedCount > 0 {
            print("  Failed:         \(failedCount)")
        }
        print(
            "  Total audio:    \(String(format: "%.1f", totalAudioDurationS / 3600))h"
        )
        print(
            "  Total time:     \(String(format: "%.1f", totalLatencyMs / 1000))s"
        )
        print("  Avg latency:    \(String(format: "%.0f", avgLatency))ms")
        print("  RTFx:           \(String(format: "%.1f", rtfx))x")

        if !pytorchBoundaryErrors.isEmpty {
            let ptAAS =
                pytorchBoundaryErrors.reduce(0, +)
                / Double(pytorchBoundaryErrors.count)
            let meanDiff =
                coremlVsPytorchDiffs.reduce(0, +)
                / Double(coremlVsPytorchDiffs.count)

            print("")
            print("PyTorch f32 vs Human Ground Truth:")
            print("  AAS:            \(String(format: "%.1f", ptAAS)) ms")
            print("  Boundaries:     \(pytorchBoundaryErrors.count)")
            print("")
            print("CoreML int8 vs PyTorch f32:")
            print("  Mean diff:      \(String(format: "%.1f", meanDiff)) ms")
            print("  Boundaries:     \(coremlVsPytorchDiffs.count)")
        }

        print(String(repeating: "=", count: 60))

        // 9. Save JSON output
        if let outPath = outputPath {
            var outputDict: [String: Any] = [
                "model": "Qwen3-ForcedAligner-0.6B (CoreML int8)",
                "dataset": "Buckeye Corpus v2.0",
                "num_samples": processedCount,
                "metrics": [
                    "aas_ms": round(overallAAS * 10) / 10,
                    "median_aas_ms": round(medianAAS * 10) / 10,
                    "max_aas_ms": round(maxAAS * 10) / 10,
                    "pct_within_20ms": round(within20ms * 10) / 10,
                    "pct_within_50ms": round(within50ms * 10) / 10,
                    "pct_within_100ms": round(within100ms * 10) / 10,
                    "total_boundaries": allBoundaryErrors.count,
                ] as [String: Any],
                "performance": [
                    "total_audio_hours": round(totalAudioDurationS / 3600 * 100) / 100,
                    "total_alignment_s": round(totalLatencyMs / 1000 * 10) / 10,
                    "avg_latency_ms": round(avgLatency * 10) / 10,
                    "rtfx": round(rtfx * 10) / 10,
                ] as [String: Any],
                "samples": sampleResults,
            ]

            if !pytorchBoundaryErrors.isEmpty {
                let ptAAS =
                    pytorchBoundaryErrors.reduce(0, +)
                    / Double(pytorchBoundaryErrors.count)
                let meanDiff =
                    coremlVsPytorchDiffs.reduce(0, +)
                    / Double(coremlVsPytorchDiffs.count)
                outputDict["pytorch_comparison"] =
                    [
                        "pytorch_aas_ms": round(ptAAS * 10) / 10,
                        "coreml_aas_ms": round(overallAAS * 10) / 10,
                        "mean_timestamp_diff_ms": round(meanDiff * 10) / 10,
                    ] as [String: Any]
            }

            do {
                let jsonData = try JSONSerialization.data(
                    withJSONObject: outputDict,
                    options: [.prettyPrinted, .sortedKeys]
                )
                let outURL = URL(fileURLWithPath: outPath)
                try? FileManager.default.createDirectory(
                    at: outURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try jsonData.write(to: outURL)
                logger.info("Results saved to \(outPath)")
            } catch {
                logger.error("Failed to save results: \(error)")
            }
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Forced Alignment Benchmark (Buckeye Corpus)

            Usage: fluidaudio align-benchmark [options]

            Options:
                --help, -h                  Show this help message
                --num-files <n>             Number of segments to process (default: 1000)
                --model-dir <path>          Path to local ForcedAligner model directory
                --dataset-dir <path>        Path to Buckeye dataset directory
                --pytorch-ref <file.json>   PyTorch reference results for comparison
                --output <file.json>        Save detailed results to JSON
                --auto-download             Auto-download Buckeye dataset if not found

            Examples:
                fluidaudio align-benchmark --num-files 5
                fluidaudio align-benchmark --pytorch-ref pytorch_buckeye_1000.json --output results.json
                fluidaudio align-benchmark --dataset-dir ./buckeye-benchmark --num-files 100
            """
        )
    }
}
#endif
