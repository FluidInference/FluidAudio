#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Multi-stream parallel benchmark for Nemotron multilingual.
///
/// Spawns N independent `StreamingNemotronMultilingualAsrManager` instances
/// (each loads its own copy of the CoreML models — accept N× model memory
/// in exchange for clean per-stream cache isolation), distributes a batch
/// of audio files across them via a shared work queue, and reports the
/// aggregate throughput.
///
/// The Apple Neural Engine has limited concurrent context support. The
/// expected curve is roughly:
///   N=1 → baseline (e.g. ~99.6 RTFx test-clean on LP [42,13]+B1+triple-stage)
///   N=2 → ~1.5-1.8× aggregate (some ANE contention, but triple-stage
///         within a single stream already overlaps CPU+ANE so the
///         per-stream parallel gain is smaller than naive 2×)
///   N=4 → flattens — ANE dispatch becomes the bottleneck
///
/// Use this command to validate the multi-stream model for batch
/// transcription workloads (podcast queues, meeting backlogs, etc.).
public enum NemotronMultilingualMultiStreamBench {
    private static let logger = AppLogger(category: "NemotronMultiStream")

    public struct Config {
        var modelDir: URL?
        var streams: Int = 2
        var samples: Int = 100
        var language: String = "en-US"
        var datasetSubset: String = "test-clean"
        public init() {}
    }

    public static func run(arguments: [String]) async {
        var config = Config()
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--model-dir", "-m":
                i += 1
                if i < arguments.count { config.modelDir = URL(fileURLWithPath: arguments[i]) }
            case "--streams", "-n":
                i += 1
                if i < arguments.count, let n = Int(arguments[i]) { config.streams = max(1, n) }
            case "--samples":
                i += 1
                if i < arguments.count, let s = Int(arguments[i]) { config.samples = s }
            case "--language", "-l":
                i += 1
                if i < arguments.count { config.language = arguments[i] }
            case "--subset":
                i += 1
                if i < arguments.count { config.datasetSubset = arguments[i] }
            case "--help", "-h":
                printUsage()
                return
            default:
                logger.warning("Unknown arg: \(arguments[i])")
            }
            i += 1
        }
        guard let modelDir = config.modelDir else {
            logger.error("--model-dir is required")
            printUsage()
            return
        }

        // Locate LS test-clean samples
        let cacheRoot = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("FluidAudio/Datasets/LibriSpeech")
        let subsetDir = cacheRoot.appendingPathComponent(config.datasetSubset)
        guard FileManager.default.fileExists(atPath: subsetDir.path) else {
            logger.error("LibriSpeech subset missing at \(subsetDir.path)")
            return
        }

        var flacs: [URL] = []
        if let walker = FileManager.default.enumerator(at: subsetDir, includingPropertiesForKeys: nil) {
            while let item = walker.nextObject() {
                if let url = item as? URL, url.pathExtension.lowercased() == "flac" {
                    flacs.append(url)
                }
            }
        }
        flacs.sort { $0.lastPathComponent < $1.lastPathComponent }
        let take = config.samples == Int.max ? flacs.count : min(config.samples, flacs.count)
        let files = Array(flacs.prefix(take))
        logger.info("Multi-stream bench: \(config.streams) streams × \(files.count) files (LS \(config.datasetSubset))")

        // Load CoreML models ONCE into a shared bundle, then initialize
        // N managers off that bundle. Each manager allocates only its
        // own per-stream state (~50 MB) rather than a fresh model copy
        // (~1.5 GB). Memory footprint: O(1) for models + O(N) for state,
        // down from O(N) total in the original MVP.
        let loadStart = Date()
        let shared: SharedNemotronMultilingualModels
        do {
            shared = try await StreamingNemotronMultilingualAsrManager.preloadShared(from: modelDir)
        } catch {
            logger.error("Shared preload failed: \(error)")
            return
        }
        var managers: [StreamingNemotronMultilingualAsrManager] = []
        managers.reserveCapacity(config.streams)
        let langForLoad = config.language
        for streamIdx in 0..<config.streams {
            let mgr = StreamingNemotronMultilingualAsrManager()
            do {
                try await mgr.loadFromShared(shared)
                await mgr.setLanguage(langForLoad)
                managers.append(mgr)
            } catch {
                print("Stream \(streamIdx) init failed: \(error)")
            }
        }
        guard managers.count == config.streams else {
            logger.error("Only \(managers.count)/\(config.streams) managers initialized; aborting")
            return
        }
        let loadTime = Date().timeIntervalSince(loadStart)
        logger.info("Loaded shared bundle + \(managers.count) managers in \(String(format: "%.1f", loadTime))s")

        // Distribute files across N work queues. Round-robin keeps the load
        // balanced for short clips with similar length (LS test-clean fits).
        var queues: [[URL]] = Array(repeating: [], count: config.streams)
        for (idx, url) in files.enumerated() {
            queues[idx % config.streams].append(url)
        }

        // Compute total audio duration + load LS-style trans.txt
        // references for WER verification. Each LS speaker subdir
        // contains a `<spk>-<chap>.trans.txt` mapping file_id → text.
        var totalDurationSec: Double = 0
        var references: [String: String] = [:]
        for url in files {
            if let f = try? AVAudioFile(forReading: url) {
                totalDurationSec += Double(f.length) / f.processingFormat.sampleRate
            }
        }
        // Walk subset dir for trans.txt to populate references
        if let walker = FileManager.default.enumerator(at: subsetDir, includingPropertiesForKeys: nil) {
            while let item = walker.nextObject() {
                guard let url = item as? URL, url.lastPathComponent.hasSuffix(".trans.txt") else { continue }
                let content = (try? String(contentsOf: url)) ?? ""
                for line in content.components(separatedBy: .newlines) where !line.isEmpty {
                    let parts = line.split(separator: " ", maxSplits: 1)
                    if parts.count == 2 { references[String(parts[0])] = String(parts[1]) }
                }
            }
        }
        logger.info("Total audio: \(String(format: "%.1f", totalDurationSec))s, \(references.count) references loaded")

        // Process all queues concurrently. Each task returns (idx,
        // processed, audio, [(file_id, hypothesis)]) so we can compute
        // per-stream WER vs reference at the end.
        let processStart = Date()
        var perStreamHyps: [Int: [(String, String)]] = [:]
        await withTaskGroup(of: (Int, Int, Double, [(String, String)]).self) { group in
            for (streamIdx, queue) in queues.enumerated() {
                let mgr = managers[streamIdx]
                let queueCopy = queue
                let idx = streamIdx
                group.addTask {
                    var processed = 0
                    var streamAudio: Double = 0
                    var hyps: [(String, String)] = []
                    let converter = AudioConverter()
                    for url in queueCopy {
                        do {
                            // Resample to 16 kHz [Float] up front (Sendable) and feed
                            // process(samples:) — passing a non-Sendable
                            // AVAudioPCMBuffer across the actor boundary fails Swift 6
                            // sending checks (the buffer's region is merged with the
                            // non-Sendable AVAudioFile via read(into:)).
                            let samples = try converter.resampleAudioFile(url)
                            streamAudio += Double(samples.count) / 16000.0
                            _ = try await mgr.process(samples: samples)
                            let result = try await mgr.finish()
                            let fileId = url.deletingPathExtension().lastPathComponent
                            hyps.append((fileId, result))
                            await mgr.reset()
                        } catch {
                            print("Stream \(idx) file \(url.lastPathComponent) failed: \(error)")
                        }
                        processed += 1
                    }
                    return (idx, processed, streamAudio, hyps)
                }
            }
            for await (streamIdx, processed, streamAudio, hyps) in group {
                perStreamHyps[streamIdx] = hyps
                logger.info(
                    "Stream \(streamIdx): processed \(processed) files (\(String(format: "%.1f", streamAudio))s audio)")
            }
        }
        let wallTime = Date().timeIntervalSince(processStart)
        let aggregateRtfx = totalDurationSec / wallTime

        // Compute aggregate WER across all streams. Use English
        // normalizer (LS test-clean is English audiobook).
        var totalWER: Double = 0
        var werSamples = 0
        for (_, hyps) in perStreamHyps {
            for (fileId, hyp) in hyps {
                guard let ref = references[fileId] else { continue }
                let m = WERCalculator.calculateWERAndCER(hypothesis: hyp, reference: ref)
                totalWER += m.wer
                werSamples += 1
            }
        }
        let avgWER = werSamples > 0 ? totalWER / Double(werSamples) : 0

        // Print summary to stdout (logger.info routes to os_log which
        // doesn't surface in shell). PROFILE counters already go to stderr.
        print(String(repeating: "=", count: 60))
        print("Streams: \(config.streams)")
        print("Files: \(files.count)")
        print("Total audio: \(String(format: "%.1f", totalDurationSec))s")
        print("Wall time: \(String(format: "%.1f", wallTime))s")
        print("Aggregate RTFx: \(String(format: "%.2f", aggregateRtfx))x")
        print("Per-stream RTFx (avg): \(String(format: "%.2f", aggregateRtfx / Double(config.streams)))x")
        print("Avg WER: \(String(format: "%.2f", avgWER * 100))% (\(werSamples) samples scored)")
        print(String(repeating: "=", count: 60))
    }

    private static func printUsage() {
        print(
            """
            Nemotron Multilingual — Multi-Stream Parallel Benchmark

            Usage: fluidaudio nemotron-multilingual-multi-stream-bench --model-dir <path> [options]

            Options:
                --model-dir, -m <path>   Path to multilingual CoreML models (required)
                --streams, -n <N>        Number of concurrent streams (default: 2)
                --samples <N>            Files per benchmark (default: 100)
                --subset <name>          LS subset: test-clean / test-other (default: test-clean)
                --language, -l <code>    Language hint (default: en-US)
                --help, -h               Show this help

            Memory: each stream loads its own copy of the encoder (~1.1 GB FP16
                / ~430 MB LAYERPOS). Plan ~1.5 GB per stream + ~1 GB baseline.

            Outputs aggregate RTFx = total_audio_seconds / wall_time. Compare
            against N=1 baseline (which equals the single-stream per-file RTFx).
            """
        )
    }
}
#endif
