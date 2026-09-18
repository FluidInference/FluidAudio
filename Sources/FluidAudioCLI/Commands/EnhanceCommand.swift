#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

/// `enhance`: LocalVQE acoustic echo cancellation + noise suppression on a file.
enum EnhanceCommand {
    private static let logger = AppLogger(category: "Enhance")

    private struct Options {
        var micPath: String?
        var referencePath: String?
        var outputPath: String?
        var modelDirectory: String?
        var variant: LocalVqeVariant = .v13
        var chunk: LocalVqeChunk = .batch256ms
        var computeUnits: MLComputeUnits = .cpuOnly
        var streaming = false
        var bufferSamples = 256
    }

    static func run(arguments: [String]) async {
        var options = Options()
        var index = 0
        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--reference", "-r":
                options.referencePath = next(arguments, &index)
            case "--output", "-o":
                options.outputPath = next(arguments, &index)
            case "--model-dir":
                options.modelDirectory = next(arguments, &index)
            case "--variant":
                guard let raw = next(arguments, &index), let v = LocalVqeVariant(rawValue: raw) else {
                    logger.error("--variant must be one of \(LocalVqeVariant.allCases.map(\.rawValue))")
                    exit(1)
                }
                options.variant = v
            case "--chunk":
                guard let raw = next(arguments, &index), let c = LocalVqeChunk(rawValue: raw) else {
                    logger.error("--chunk must be one of \(LocalVqeChunk.allCases.map(\.rawValue))")
                    exit(1)
                }
                options.chunk = c
            case "--compute-units":
                switch next(arguments, &index)?.lowercased() {
                case "cpu-only", "cpu": options.computeUnits = .cpuOnly
                case "gpu", "cpu-and-gpu": options.computeUnits = .cpuAndGPU
                case "ane", "cpu-and-ne": options.computeUnits = .cpuAndNeuralEngine
                case "all": options.computeUnits = .all
                default:
                    logger.error("--compute-units must be cpu-only | gpu | ane | all")
                    exit(1)
                }
            case "--streaming":
                options.streaming = true
            case "--buffer-samples":
                options.bufferSamples = Int(next(arguments, &index) ?? "") ?? options.bufferSamples
            default:
                if arg.hasPrefix("--") {
                    logger.warning("Unknown option: \(arg)")
                } else if options.micPath == nil {
                    options.micPath = arg
                } else {
                    logger.warning("Ignoring extra argument: \(arg)")
                }
            }
            index += 1
        }

        guard let micPath = options.micPath else {
            logger.error("No mic audio file provided")
            printUsage()
            exit(1)
        }

        do {
            let converter = AudioConverter()
            let mic = try converter.resampleAudioFile(path: micPath)
            var reference: [Float]
            if let referencePath = options.referencePath {
                reference = try converter.resampleAudioFile(path: referencePath)
            } else {
                logger.warning("No --reference given: running noise suppression / dereverb only (silent far end)")
                reference = []
            }
            if reference.count < mic.count {
                reference.append(contentsOf: [Float](repeating: 0, count: mic.count - reference.count))
            } else if reference.count > mic.count {
                reference.removeLast(reference.count - mic.count)
            }

            let config = LocalVqeConfig(
                variant: options.variant, chunk: options.chunk, computeUnits: options.computeUnits)
            let loadStart = Date()
            let manager: LocalVqeManager
            if let dir = options.modelDirectory {
                manager = try LocalVqeManager(config: config, modelDirectory: URL(fileURLWithPath: dir))
            } else {
                manager = try await LocalVqeManager(config: config)
            }
            let loadTime = Date().timeIntervalSince(loadStart)
            report(
                "Loaded LocalVQE \(options.variant.rawValue) (\(options.chunk.rawValue) chunk, "
                    + "\(describe(options.computeUnits))) in \(String(format: "%.2f", loadTime))s")

            let audioSeconds = Double(mic.count) / Double(LocalVqeManager.sampleRate)
            let enhanced: [Float]
            let start = Date()
            if options.streaming {
                enhanced = try await runStreaming(manager: manager, mic: mic, reference: reference, options: options)
            } else {
                enhanced = try await manager.process(mic: mic, reference: reference)
            }
            let wall = Date().timeIntervalSince(start)

            let inRms = rms(mic)
            let outRms = rms(enhanced)
            report(
                String(
                    format: "Enhanced %.2fs of audio in %.3fs wall (RTFx %.1fx); RMS in %.4f -> out %.4f (%.1f dB)",
                    audioSeconds, wall, audioSeconds / max(wall, 1e-9), inRms, outRms,
                    20 * log10(max(outRms, 1e-9) / max(inRms, 1e-9))))

            if let outputPath = options.outputPath {
                try writeWav(samples: enhanced, sampleRate: LocalVqeManager.sampleRate, to: outputPath)
                report("Wrote \(outputPath)")
            }
        } catch {
            logger.error("Enhance failed: \(error)")
            exit(1)
        }
    }

    /// Push audio through a `LocalVqeStream` in `bufferSamples` pieces and
    /// report per-call latency.
    private static func runStreaming(
        manager: LocalVqeManager, mic: [Float], reference: [Float], options: Options
    ) async throws -> [Float] {
        let stream = try await manager.makeStream()
        let step = max(1, options.bufferSamples)
        var out: [Float] = []
        out.reserveCapacity(mic.count)
        var latencies: [Double] = []
        var offset = 0
        while offset < mic.count {
            let end = min(offset + step, mic.count)
            let t0 = DispatchTime.now().uptimeNanoseconds
            let hop = try await stream.enhance(
                mic: Array(mic[offset..<end]), reference: Array(reference[offset..<end]))
            if !hop.isEmpty {
                latencies.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
            }
            out.append(contentsOf: hop)
            offset = end
        }
        out.append(contentsOf: try await stream.flush())

        if !latencies.isEmpty {
            let sorted = latencies.sorted()
            let p50 = sorted[sorted.count / 2]
            let p99 = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.99))]
            let callMs = Double(options.chunk.samplesPerCall) / Double(LocalVqeManager.sampleRate) * 1000
            let summary = String(
                format: "Streaming: %d model calls, %.0f ms audio per call, "
                    + "latency p50 %.2f ms / p99 %.2f ms / max %.2f ms (p50 RTFx %.1fx)",
                latencies.count, callMs, p50, p99, sorted[sorted.count - 1], callMs / p50)
            report(summary)
        }
        return out
    }

    /// User-facing result lines go to stdout (the logger is silent in release builds).
    private static func report(_ line: String) {
        print(line)
        logger.info("\(line)")
    }

    private static func next(_ arguments: [String], _ index: inout Int) -> String? {
        guard index + 1 < arguments.count else { return nil }
        index += 1
        return arguments[index]
    }

    private static func describe(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: return "cpu-only"
        case .cpuAndGPU: return "gpu"
        case .cpuAndNeuralEngine: return "ane"
        case .all: return "all"
        @unknown default: return "unknown"
        }
    }

    private static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var acc: Float = 0
        for s in samples { acc += s * s }
        return (acc / Float(samples.count)).squareRoot()
    }

    private static func writeWav(samples: [Float], sampleRate: Int, to path: String) throws {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false),
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(max(samples.count, 1)))
        else {
            throw LocalVqeError.modelProcessingFailed("could not allocate output buffer")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard let channel = buffer.floatChannelData?[0] else {
            throw LocalVqeError.modelProcessingFailed("could not access output channel")
        }
        samples.withUnsafeBufferPointer { src in
            if let base = src.baseAddress { channel.update(from: base, count: samples.count) }
        }
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.removeItem(at: url)
        let file = try AVAudioFile(
            forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: buffer)
    }

    private static func printUsage() {
        logger.info(
            """
            Usage: fluidaudiocli enhance <mic.wav> [options]

            LocalVQE speech enhancement: acoustic echo cancellation + noise suppression + dereverberation.

            Options:
                --reference, -r <file>   Far-end reference (what the speaker played). Omit for NS/dereverb only.
                --output, -o <file>      Write the enhanced 16 kHz mono WAV here.
                --variant <v1.3|v1.2>    Checkpoint (default v1.3, 4.8M params; v1.2 is 1.3M).
                --chunk <256ms|16ms>     Samples per model call (default 256ms; 16ms for live use).
                --compute-units <cpu-only|gpu|ane|all>   Default cpu-only (fp32 models).
                --streaming              Drive the LocalVqeStream API buffer-by-buffer and report call latency.
                --buffer-samples <n>     Buffer size for --streaming (default 256).
                --model-dir <dir>        Load <dir>/<model>.mlmodelc instead of downloading from HuggingFace.
            """
        )
    }
}
#endif
