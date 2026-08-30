#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// CLI for Nemotron 3 Diarization preview (local eval-license models — no HF download).
enum Nemotron3DiarizeCommand {
    private static let logger = AppLogger(category: "Nemotron3CLI")

    static func printUsage() {
        let usage = """
            Nemotron 3 Diarization (preview, internal evaluation only)

            Usage:
              fluidaudiocli nemotron3-diarize <audio.wav> --models <dir> [options]
              fluidaudiocli nemotron3-benchmark --models <dir> [options]

            Shared options:
                --models <dir>       Directory containing Nemotron3Diarizer_<variant>.mlmodelc
                                     and learnable_sil_emb.bin (REQUIRED)
                --variant <name>     offline | low | verylow | ultra (default: low)
                --threshold <t>      Speaker activity threshold (default: 0.5)

            nemotron3-diarize options:
                --dump-preds <file>  Write raw frame probabilities (float32 LE, [T, 8]) for parity checks
                --output <file>      Write RTTM hypothesis

            nemotron3-benchmark options:
                --dataset <name>     ami (default: ami)
                --single-file <name> Process one meeting (e.g. ES2004a)
                --max-files <n>      Limit number of files
                --collar <sec>       DER collar (default: 0)
                --output <file>      Output JSON results
            """
        fputs(usage, stderr)
        fflush(stderr)
    }

    struct CustomShape {
        var chunkLen: Int?
        var rightContext: Int?
        var fifoLen: Int?
        var spkcacheLen: Int?
        var updatePeriod: Int?
    }

    private static func loadDiarizer(
        modelsDir: String, variantName: String, custom: CustomShape = CustomShape(),
        computeUnits: MLComputeUnits = .all
    ) async throws -> (Nemotron3Diarizer, TimeInterval) {
        var config: Nemotron3Config
        if let preset = Nemotron3Config.preset(named: variantName) {
            config = preset
        } else {
            // Sweep variant: derive shape from flags, model file from the variant name.
            config = Nemotron3Config(
                chunkLen: custom.chunkLen ?? 9,
                chunkRightContext: custom.rightContext ?? 4,
                fifoLen: custom.fifoLen ?? 40,
                spkcacheLen: custom.spkcacheLen ?? 264,
                spkcacheUpdatePeriod: custom.updatePeriod ?? 40,
                modelFileName: "Nemotron3Diarizer_\(variantName).mlmodelc")
        }
        let start = Date()
        let models = try await Nemotron3Models.load(
            config: config,
            directory: URL(fileURLWithPath: modelsDir),
            computeUnits: computeUnits
        )
        return (Nemotron3Diarizer(config: config, models: models), Date().timeIntervalSince(start))
    }

    static func parseComputeUnits(_ s: String?) -> MLComputeUnits {
        switch s {
        case "ane": return .cpuAndNeuralEngine
        case "gpu": return .cpuAndGPU
        case "cpu": return .cpuOnly
        default: return .all
        }
    }

    /// Silero VAD -> per-10ms-frame speech mask, with speech regions padded by
    /// `padSeconds` on both sides to protect onsets/offsets at chunk granularity.
    /// Pass a shared `VadManager` when calling repeatedly — a fresh instance per file
    /// leaks IOSurfaces across a long benchmark run and eventually fails allocation.
    static func speechMask(
        audio: [Float], threshold: Float, padSeconds: Double = 1.0,
        vad existingVad: VadManager? = nil
    ) async throws -> [Bool] {
        let vad: VadManager
        if let existingVad {
            vad = existingVad
        } else {
            vad = try await VadManager(config: VadConfig(defaultThreshold: threshold))
        }
        let framesPerVadChunk = VadManager.chunkSize / 160  // 4096 samples -> 25.6 x 10ms frames
        let frameCount = (audio.count + 159) / 160
        var mask = [Bool](repeating: false, count: frameCount)
        // Process in bounded segments: one monolithic process() over a long meeting churns
        // thousands of MLMultiArrays without an autorelease drain and exhausts IOSurfaces
        // (same failure class as issue #752). Silero state resets per call anyway; the 1 s
        // padding below absorbs boundary effects.
        let segmentSamples = 300 * 16000
        var segmentStart = 0
        while segmentStart < audio.count {
            let segmentEnd = min(segmentStart + segmentSamples, audio.count)
            let results = try await vad.process(Array(audio[segmentStart..<segmentEnd]))
            let frameOffset = segmentStart / 160
            for (i, r) in results.enumerated() where r.probability >= threshold {
                let start = frameOffset + i * framesPerVadChunk
                let end = min(frameOffset + (i + 1) * framesPerVadChunk + 1, frameCount)
                if start < frameCount {
                    for f in start..<end { mask[f] = true }
                }
            }
            segmentStart = segmentEnd
            await Task.yield()
        }
        let pad = Int(padSeconds * 100)
        var padded = mask
        var lastSpeech = -pad - 1
        for f in 0..<frameCount {
            if mask[f] { lastSpeech = f }
            if f - lastSpeech <= pad { padded[f] = true }
        }
        lastSpeech = frameCount + pad + 1
        for f in stride(from: frameCount - 1, through: 0, by: -1) {
            if mask[f] { lastSpeech = f }
            if lastSpeech - f <= pad { padded[f] = true }
        }
        return padded
    }

    // MARK: - Single-file diarization

    static func runDiarize(arguments: [String]) async {
        var audioPath: String?
        var modelsDir: String?
        var variantName = "low"
        var threshold: Float = 0.5
        var dumpPredsPath: String?
        var outputPath: String?
        var showProfile = false
        var useVad = false
        var vadThreshold: Float = 0.85
        var custom = CustomShape()
        var computeUnits: MLComputeUnits = .all

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--models":
                i += 1
                modelsDir = arguments[safe: i]
            case "--variant":
                i += 1
                variantName = arguments[safe: i] ?? "low"
            case "--threshold":
                i += 1
                threshold = arguments[safe: i].flatMap(Float.init) ?? 0.5
            case "--dump-preds":
                i += 1
                dumpPredsPath = arguments[safe: i]
            case "--output":
                i += 1
                outputPath = arguments[safe: i]
            case "--profile":
                showProfile = true
            case "--vad":
                useVad = true
            case "--vad-threshold":
                i += 1
                vadThreshold = arguments[safe: i].flatMap(Float.init) ?? 0.85
            case "--chunk-len":
                i += 1
                custom.chunkLen = arguments[safe: i].flatMap(Int.init)
            case "--rc":
                i += 1
                custom.rightContext = arguments[safe: i].flatMap(Int.init)
            case "--fifo":
                i += 1
                custom.fifoLen = arguments[safe: i].flatMap(Int.init)
            case "--spkcache":
                i += 1
                custom.spkcacheLen = arguments[safe: i].flatMap(Int.init)
            case "--update-period":
                i += 1
                custom.updatePeriod = arguments[safe: i].flatMap(Int.init)
            case "--compute-units":
                i += 1
                computeUnits = parseComputeUnits(arguments[safe: i])
            case "--help", "-h":
                printUsage()
                return
            default:
                if !arg.hasPrefix("--"), audioPath == nil { audioPath = arg }
            }
            i += 1
        }

        guard let audioPath, let modelsDir else {
            printUsage()
            exit(1)
        }

        do {
            let (diarizer, loadTime) = try await loadDiarizer(
                modelsDir: modelsDir, variantName: variantName, custom: custom, computeUnits: computeUnits)
            print("Model loaded in \(String(format: "%.2f", loadTime))s (variant: \(variantName))")

            let audio = try AudioConverter().resampleAudioFile(path: audioPath)
            let duration = Float(audio.count) / 16000.0

            var mask: [Bool]? = nil
            if useVad {
                let vadStart = Date()
                mask = try await speechMask(audio: audio, threshold: vadThreshold)
                let speechRatio = Float(mask!.filter { $0 }.count) / Float(mask!.count)
                print(
                    "VAD: \(String(format: "%.1f", Date().timeIntervalSince(vadStart)))s, "
                        + "\(String(format: "%.0f", speechRatio * 100))% speech (padded)")
            }

            let start = Date()
            let (probs, frames) = try diarizer.processComplete(audio, speechMask: mask)
            let elapsed = Date().timeIntervalSince(start)
            let rtfx = duration / Float(elapsed)
            if useVad {
                let p = diarizer.lastProfile
                print("VAD gating: skipped \(p.skippedChunks)/\(p.chunkCount + p.skippedChunks) chunks")
            }

            print(
                "Processed \(String(format: "%.1f", duration))s in \(String(format: "%.2f", elapsed))s "
                    + "(RTFx \(String(format: "%.1f", rtfx))x), \(frames) x 10ms frames")

            if showProfile {
                let p = diarizer.lastProfile
                let n = Double(max(p.chunkCount, 1))
                func line(_ name: String, _ s: Double) {
                    let pct = p.totalSeconds > 0 ? s / p.totalSeconds * 100 : 0
                    let padded = name.padding(toLength: 14, withPad: " ", startingAt: 0)
                    print(
                        padded
                            + String(format: "%8.3fs  %5.1f%%  %8.3fms/chunk", s, pct, s / n * 1000))
                }
                print("Pipeline profile (\(p.chunkCount) chunks):")
                line("mel", p.melSeconds)
                line("chunk-slice", p.chunkSliceSeconds)
                line("input-prep", p.inputPrepSeconds)
                line("predict", p.predictSeconds)
                line("readback", p.readbackSeconds)
                line("state-update", p.stateUpdateSeconds)
                line("output-append", p.outputAppendSeconds)
                line("total", p.totalSeconds)
            }

            let segments = Nemotron3Diarizer.segments(
                probabilities: probs, frameCount: frames, threshold: threshold)
            let speakers = Set(segments.map(\.speakerIndex))
            print("Detected \(speakers.count) speakers, \(segments.count) segments")
            for seg in segments.prefix(20) {
                print(
                    "  spk\(seg.speakerIndex): \(String(format: "%7.2f", seg.startSeconds))s - "
                        + "\(String(format: "%7.2f", seg.endSeconds))s")
            }
            if segments.count > 20 { print("  ... (\(segments.count - 20) more)") }

            if let dumpPredsPath {
                var data = Data(capacity: probs.count * 4)
                probs.withUnsafeBytes { data.append(contentsOf: $0) }
                try data.write(to: URL(fileURLWithPath: dumpPredsPath))
                print("Dumped \(frames)x8 frame probabilities to \(dumpPredsPath)")
            }

            if let outputPath {
                let fileId = URL(fileURLWithPath: audioPath).deletingPathExtension().lastPathComponent
                var rttm = ""
                for seg in segments {
                    let dur = seg.endSeconds - seg.startSeconds
                    rttm +=
                        "SPEAKER \(fileId) 1 \(String(format: "%.3f", seg.startSeconds)) "
                        + "\(String(format: "%.3f", dur)) <NA> <NA> speaker_\(seg.speakerIndex) <NA> <NA>\n"
                }
                try rttm.write(toFile: outputPath, atomically: true, encoding: .utf8)
                print("Wrote RTTM to \(outputPath)")
            }
        } catch {
            print("Error: \(error)")
            exit(1)
        }
    }

    // MARK: - Batch mode (concurrent GPU streams)

    /// Process many files with N concurrent workers, each owning its own model instance.
    /// Multi-stream measurement: the M5 Pro GPU takes exactly one extra concurrent stream
    /// (+43% aggregate) before saturating, so the default is 2 workers on the GPU route.
    static func runBatch(arguments: [String]) async {
        var modelsDir: String?
        var variantName = "fast32"
        var workers = 2
        var computeUnits: MLComputeUnits = .cpuAndGPU
        var files: [String] = []
        var dataset: DiarizationBenchmarkUtils.Dataset = .ami
        var threshold: Float = 0.5
        var collar: Double = 0
        var maxFiles: Int?

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--models":
                i += 1
                modelsDir = arguments[safe: i]
            case "--variant":
                i += 1
                variantName = arguments[safe: i] ?? "fast32"
            case "--workers":
                i += 1
                workers = arguments[safe: i].flatMap(Int.init) ?? 2
            case "--compute-units":
                i += 1
                computeUnits = parseComputeUnits(arguments[safe: i])
            case "--dataset":
                i += 1
                dataset = DiarizationBenchmarkUtils.Dataset(rawValue: arguments[safe: i] ?? "ami") ?? .ami
            case "--files":
                i += 1
                files = arguments[safe: i]?.split(separator: ",").map(String.init) ?? []
            case "--max-files":
                i += 1
                maxFiles = arguments[safe: i].flatMap(Int.init)
            case "--threshold":
                i += 1
                threshold = arguments[safe: i].flatMap(Float.init) ?? 0.5
            case "--collar":
                i += 1
                collar = arguments[safe: i].flatMap(Double.init) ?? 0
            case "--help", "-h":
                printUsage()
                return
            default:
                break
            }
            i += 1
        }

        guard let modelsDir else {
            printUsage()
            exit(1)
        }
        if files.isEmpty {
            files = DiarizationBenchmarkUtils.getFiles(for: dataset, maxFiles: maxFiles)
        }
        guard !files.isEmpty else {
            print("No files to process.")
            exit(1)
        }

        print("Batch: \(files.count) files, \(workers) workers, variant \(variantName)")
        let wallStart = Date()

        // Stride-assign files to workers; each worker loads its own model instance so
        // CoreML queues the streams independently.
        let assignments = (0..<workers).map { w in
            stride(from: w, to: files.count, by: workers).map { files[$0] }
        }

        let results = await withTaskGroup(
            of: [DiarizationBenchmarkUtils.BenchmarkResult].self
        ) { group in
            for (w, assigned) in assignments.enumerated() where !assigned.isEmpty {
                let capturedDataset = dataset
                let capturedThreshold = threshold
                let capturedCollar = collar
                let capturedVariant = variantName
                let capturedModelsDir = modelsDir
                let capturedUnits = computeUnits
                group.addTask {
                    guard
                        let (diarizer, _) = try? await loadDiarizer(
                            modelsDir: capturedModelsDir, variantName: capturedVariant,
                            computeUnits: capturedUnits)
                    else {
                        print("worker \(w): model load failed")
                        return []
                    }
                    var workerResults: [DiarizationBenchmarkUtils.BenchmarkResult] = []
                    for meeting in assigned {
                        if let r = await evaluateMeeting(
                            diarizer: diarizer, meeting: meeting, dataset: capturedDataset,
                            threshold: capturedThreshold, collar: capturedCollar)
                        {
                            workerResults.append(r)
                        }
                    }
                    return workerResults
                }
            }
            var all: [DiarizationBenchmarkUtils.BenchmarkResult] = []
            for await r in group { all.append(contentsOf: r) }
            return all
        }

        let wall = Date().timeIntervalSince(wallStart)
        guard !results.isEmpty else {
            print("No results.")
            exit(1)
        }
        let totalAudio = results.map { Double($0.totalFrames) * 0.01 }.reduce(0, +)
        let avgDER = results.map(\.der).reduce(0, +) / Float(results.count)
        print("\n=== Batch (\(variantName), \(workers) workers) ===")
        print("Files: \(results.count)/\(files.count)  audio \(String(format: "%.0f", totalAudio))s")
        print("Wall: \(String(format: "%.1f", wall))s  aggregate RTFx \(String(format: "%.0f", totalAudio / wall))x")
        print("Avg DER: \(String(format: "%.2f", avgDER))%")
    }

    // MARK: - AMI benchmark

    static func runBenchmark(arguments: [String]) async {
        var modelsDir: String?
        var variantName = "low"
        var threshold: Float = 0.5
        var collar: Double = 0
        var singleFile: String?
        var fileList: [String]?
        var maxFiles: Int?
        var outputFile: String?
        var custom = CustomShape()
        var computeUnits: MLComputeUnits = .all
        var dataset: DiarizationBenchmarkUtils.Dataset = .ami
        var useVad = false
        var vadThreshold: Float = 0.85

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--vad":
                useVad = true
            case "--vad-threshold":
                i += 1
                vadThreshold = arguments[safe: i].flatMap(Float.init) ?? 0.85
            case "--models":
                i += 1
                modelsDir = arguments[safe: i]
            case "--variant":
                i += 1
                variantName = arguments[safe: i] ?? "low"
            case "--threshold":
                i += 1
                threshold = arguments[safe: i].flatMap(Float.init) ?? 0.5
            case "--collar":
                i += 1
                collar = arguments[safe: i].flatMap(Double.init) ?? 0
            case "--dataset":
                i += 1
                dataset = DiarizationBenchmarkUtils.Dataset(rawValue: arguments[safe: i] ?? "ami") ?? .ami
            case "--single-file":
                i += 1
                singleFile = arguments[safe: i]
            case "--files":
                i += 1
                fileList = arguments[safe: i]?.split(separator: ",").map(String.init)
            case "--max-files":
                i += 1
                maxFiles = arguments[safe: i].flatMap(Int.init)
            case "--output":
                i += 1
                outputFile = arguments[safe: i]
            case "--chunk-len":
                i += 1
                custom.chunkLen = arguments[safe: i].flatMap(Int.init)
            case "--rc":
                i += 1
                custom.rightContext = arguments[safe: i].flatMap(Int.init)
            case "--fifo":
                i += 1
                custom.fifoLen = arguments[safe: i].flatMap(Int.init)
            case "--spkcache":
                i += 1
                custom.spkcacheLen = arguments[safe: i].flatMap(Int.init)
            case "--update-period":
                i += 1
                custom.updatePeriod = arguments[safe: i].flatMap(Int.init)
            case "--compute-units":
                i += 1
                computeUnits = parseComputeUnits(arguments[safe: i])
            case "--help", "-h":
                printUsage()
                return
            default:
                break
            }
            i += 1
        }

        guard let modelsDir else {
            printUsage()
            exit(1)
        }

        do {
            let (diarizer, loadTime) = try await loadDiarizer(
                modelsDir: modelsDir, variantName: variantName, custom: custom, computeUnits: computeUnits)
            print("Model loaded in \(String(format: "%.2f", loadTime))s (variant: \(variantName))")

            let meetings =
                singleFile.map { [$0] }
                ?? fileList
                ?? DiarizationBenchmarkUtils.getFiles(for: dataset, maxFiles: maxFiles)
            print(
                "Benchmarking \(meetings.count) \(dataset.rawValue) meetings, "
                    + "threshold \(threshold), collar \(collar)")

            let vad: VadManager? =
                useVad ? try await VadManager(config: VadConfig(defaultThreshold: vadThreshold)) : nil

            var results: [DiarizationBenchmarkUtils.BenchmarkResult] = []
            for meeting in meetings {
                if let r = await evaluateMeeting(
                    diarizer: diarizer, meeting: meeting, dataset: dataset,
                    threshold: threshold, collar: collar,
                    vadThreshold: useVad ? vadThreshold : nil, vad: vad)
                {
                    results.append(r)
                    print(
                        "   \(meeting): DER \(String(format: "%.2f", r.der))% "
                            + "(miss \(String(format: "%.1f", r.missRate))% "
                            + "fa \(String(format: "%.1f", r.falseAlarmRate))% "
                            + "conf \(String(format: "%.1f", r.speakerErrorRate))%), "
                            + "RTFx \(String(format: "%.1f", r.rtfx))x, "
                            + "spk \(r.detectedSpeakers)/\(r.groundTruthSpeakers)")
                }
            }

            guard !results.isEmpty else {
                print("No results.")
                exit(1)
            }
            let avgDER = results.map(\.der).reduce(0, +) / Float(results.count)
            let avgMiss = results.map(\.missRate).reduce(0, +) / Float(results.count)
            let avgFA = results.map(\.falseAlarmRate).reduce(0, +) / Float(results.count)
            let avgConf = results.map(\.speakerErrorRate).reduce(0, +) / Float(results.count)
            let avgRTFx = results.map(\.rtfx).reduce(0, +) / Float(results.count)
            print("\n=== Nemotron 3 Diarization (\(variantName)) — AMI SDM test (\(results.count) files) ===")
            print("Avg DER:  \(String(format: "%.2f", avgDER))%")
            print(
                "  miss \(String(format: "%.2f", avgMiss))%  fa \(String(format: "%.2f", avgFA))%  "
                    + "conf \(String(format: "%.2f", avgConf))%")
            print("Avg RTFx: \(String(format: "%.1f", avgRTFx))x")

            if let outputFile {
                DiarizationBenchmarkUtils.saveJSONResults(results: results, to: outputFile)
                print("Saved results to \(outputFile)")
            }
        } catch {
            print("Error: \(error)")
            exit(1)
        }
    }

    private static func evaluateMeeting(
        diarizer: Nemotron3Diarizer, meeting: String,
        dataset: DiarizationBenchmarkUtils.Dataset = .ami,
        threshold: Float, collar: Double,
        vadThreshold: Float? = nil, vad: VadManager? = nil
    ) async -> DiarizationBenchmarkUtils.BenchmarkResult? {
        let audioPath = DiarizationBenchmarkUtils.getAudioPath(for: meeting, dataset: dataset)
        guard FileManager.default.fileExists(atPath: audioPath) else {
            print("   Audio not found: \(audioPath)")
            return nil
        }
        do {
            let audioLoadStart = Date()
            let audio = try AudioConverter().resampleAudioFile(path: audioPath)
            let audioLoadTime = Date().timeIntervalSince(audioLoadStart)
            let duration = Float(audio.count) / 16000.0
            print("   \(meeting): \(String(format: "%.1f", duration))s")

            var mask: [Bool]? = nil
            if let vadThreshold {
                mask = try await speechMask(audio: audio, threshold: vadThreshold, vad: vad)
            }

            let start = Date()
            let (probs, frames) = try diarizer.processComplete(audio, speechMask: mask)
            let processingTime = Date().timeIntervalSince(start)
            let rtfx = duration / Float(processingTime)
            if vadThreshold != nil {
                let p = diarizer.lastProfile
                print(
                    "   VAD skipped \(p.skippedChunks)/\(p.chunkCount + p.skippedChunks) chunks")
            }

            let segments = Nemotron3Diarizer.segments(
                probabilities: probs, frameCount: frames, threshold: threshold, minDurationSeconds: 0)

            var groundTruth: [TimedSpeakerSegment] = []
            if let rttmURL = DiarizationBenchmarkUtils.getRTTMURL(for: meeting, dataset: dataset),
                FileManager.default.fileExists(atPath: rttmURL.path),
                let content = try? String(contentsOf: rttmURL, encoding: .utf8)
            {
                groundTruth = parseRTTM(content)
            }
            if groundTruth.isEmpty, dataset == .ami {
                groundTruth = try AMIParser.loadWordAlignedGroundTruth(for: meeting, duration: duration)
            }
            guard !groundTruth.isEmpty else {
                print("   No ground truth for \(meeting)")
                return nil
            }

            let ref = groundTruth.map {
                DERSpeakerSegment(
                    speaker: $0.speakerId, start: Double($0.startTimeSeconds), end: Double($0.endTimeSeconds))
            }
            let hyp = segments.map {
                DERSpeakerSegment(
                    speaker: "speaker_\($0.speakerIndex)", start: Double($0.startSeconds),
                    end: Double($0.endSeconds))
            }
            let der = DiarizationDER.compute(ref: ref, hyp: hyp, frameStep: 0.01, collar: collar)
            let totalRef = max(der.totalRefSpeech, .leastNonzeroMagnitude)

            return DiarizationBenchmarkUtils.BenchmarkResult(
                meetingName: meeting,
                der: Float(der.der * 100),
                missRate: Float(der.miss / totalRef * 100),
                falseAlarmRate: Float(der.falseAlarm / totalRef * 100),
                speakerErrorRate: Float(der.confusion / totalRef * 100),
                rtfx: rtfx,
                processingTime: processingTime,
                totalFrames: frames,
                detectedSpeakers: Set(segments.map(\.speakerIndex)).count,
                groundTruthSpeakers: Set(groundTruth.map(\.speakerId)).count,
                modelLoadTime: 0,
                audioLoadTime: audioLoadTime
            )
        } catch {
            print("   Error on \(meeting): \(error)")
            return nil
        }
    }

    private static func parseRTTM(_ content: String) -> [TimedSpeakerSegment] {
        var segments: [TimedSpeakerSegment] = []
        for line in content.components(separatedBy: .newlines) {
            let parts = line.trimmingCharacters(in: .whitespaces)
                .components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            guard parts.count >= 8, parts[0] == "SPEAKER",
                let start = Float(parts[3]), let dur = Float(parts[4])
            else { continue }
            segments.append(
                TimedSpeakerSegment(
                    speakerId: parts[7], embedding: [], startTimeSeconds: start,
                    endTimeSeconds: start + dur, qualityScore: 1.0))
        }
        return segments
    }
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
#endif
