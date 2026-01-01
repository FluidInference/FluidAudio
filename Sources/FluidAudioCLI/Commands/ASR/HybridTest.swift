#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Test command for dual-model approach: CTC 110M for keyword spotting + TDT 0.6B for transcription.
public enum HybridTest {
    public static func runCLI(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var audioPath: String?
        var maxFiles: Int = 3

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--audio":
                if i + 1 < arguments.count {
                    audioPath = arguments[i + 1]
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1]) ?? 3
                    i += 1
                }
            default:
                if !arguments[i].hasPrefix("-") && audioPath == nil {
                    audioPath = arguments[i]
                }
            }
            i += 1
        }

        do {
            // Load CTC 110M model for keyword spotting
            print("Loading CTC 110M model for keyword spotting...")
            let hybridModels = try await HybridAsrModels.downloadAndLoad()
            let ctcManager = HybridAsrManager(models: hybridModels)
            let spotter = HybridKeywordSpotter(vocabulary: hybridModels.vocabulary, blankId: hybridModels.blankId)
            print("  CTC vocab size: \(hybridModels.vocabSize)")

            // Load TDT 0.6B model for transcription
            print("Loading TDT 0.6B model for transcription...")
            let tdtModels = try await AsrModels.downloadAndLoad()
            let tdtManager = AsrManager()
            try await tdtManager.initialize(models: tdtModels)
            print("  TDT vocab size: \(tdtModels.vocabulary.count)")
            print()

            if let audioPath = audioPath {
                try await processFile(
                    audioPath: audioPath,
                    ctcManager: ctcManager,
                    tdtManager: tdtManager,
                    spotter: spotter
                )
            } else {
                let dataDir = DatasetDownloader.getEarnings22Directory().appendingPathComponent("test-dataset")
                if FileManager.default.fileExists(atPath: dataDir.path) {
                    try await processEarningsFiles(
                        dataDir: dataDir,
                        ctcManager: ctcManager,
                        tdtManager: tdtManager,
                        spotter: spotter,
                        maxFiles: maxFiles
                    )
                } else {
                    print("No audio file specified and earnings dataset not found.")
                    print("Usage: fluidaudio hybrid-test <audio.wav>")
                }
            }
        } catch {
            print("ERROR: \(error)")
        }
    }

    private static func processFile(
        audioPath: String,
        ctcManager: HybridAsrManager,
        tdtManager: AsrManager,
        spotter: HybridKeywordSpotter
    ) async throws {
        print("Processing: \(audioPath)")

        // Load audio
        let url = URL(fileURLWithPath: audioPath)
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw NSError(
                domain: "HybridTest", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
        }
        try audioFile.read(into: buffer)

        // Convert to 16kHz mono
        let converter = AudioConverter()
        let samples = try converter.resampleBuffer(buffer)

        // Run CTC 110M for keyword detection
        let ctcResult = try await ctcManager.transcribe(audioSamples: samples)

        // Run TDT 0.6B for transcription
        let tdtResult = try await tdtManager.transcribe(samples)

        print()
        print("=== TDT 0.6B TRANSCRIPTION ===")
        print(tdtResult.text)
        print()
        print("=== CTC 110M KEYWORD DETECTION ===")
        print(
            "CTC frames: \(ctcResult.ctcLogProbs.count), frame duration: \(String(format: "%.3f", ctcResult.frameDuration))s"
        )
    }

    private static func processEarningsFiles(
        dataDir: URL,
        ctcManager: HybridAsrManager,
        tdtManager: AsrManager,
        spotter: HybridKeywordSpotter,
        maxFiles: Int
    ) async throws {
        print("Processing earnings test files from: \(dataDir.path)")
        print()

        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: nil)

        var fileIds: [String] = []
        for url in contents.sorted(by: { $0.path < $1.path }) {
            let name = url.lastPathComponent
            if name.hasSuffix(".dictionary.txt") {
                let fileId = String(name.dropLast(".dictionary.txt".count))
                if let data = try? Data(contentsOf: url), !data.isEmpty {
                    fileIds.append(fileId)
                }
            }
        }

        fileIds = Array(fileIds.prefix(maxFiles))
        print("Testing \(fileIds.count) files...")
        print()

        for fileId in fileIds {
            let wavPath = dataDir.appendingPathComponent("\(fileId).wav")
            let dictPath = dataDir.appendingPathComponent("\(fileId).dictionary.txt")

            guard fm.fileExists(atPath: wavPath.path) else { continue }

            // Load dictionary words
            let dictContent = (try? String(contentsOf: dictPath, encoding: .utf8)) ?? ""
            let dictWords = dictContent.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }

            print("[\(fileId)]")
            print("  Keywords: \(dictWords.joined(separator: ", "))")

            // Load audio
            let audioFile = try AVAudioFile(forReading: wavPath)
            let format = audioFile.processingFormat
            let frameCount = AVAudioFrameCount(audioFile.length)

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                continue
            }
            try audioFile.read(into: buffer)
            let converter = AudioConverter()
            let samples = try converter.resampleBuffer(buffer)

            // Build custom vocabulary for CTC spotter
            var terms: [CustomVocabularyTerm] = []
            for word in dictWords {
                terms.append(
                    CustomVocabularyTerm(text: word, weight: nil, aliases: nil, tokenIds: nil, ctcTokenIds: nil))
            }
            let vocab = CustomVocabularyContext(terms: terms)

            // Run CTC 110M for keyword detection
            let ctcResult = try await ctcManager.transcribe(audioSamples: samples, customVocabulary: vocab)

            // Run TDT 0.6B for transcription
            let tdtResult = try await tdtManager.transcribe(samples)

            // Spot keywords using CTC
            let detections = spotter.spotKeywords(
                ctcLogProbs: ctcResult.ctcLogProbs,
                frameDuration: ctcResult.frameDuration,
                customVocabulary: vocab
            )

            print("  Hybrid 110M Transcription (CTC): \(ctcResult.text.prefix(80))...")
            print("  TDT 0.6B Transcription: \(tdtResult.text.prefix(80))...")
            print()
            print("  CTC Keyword Detections:")
            for det in detections {
                let timeRange = "\(String(format: "%.2f", det.startTime))s - \(String(format: "%.2f", det.endTime))s"
                print("    [\(timeRange)] \(det.term.text) (score: \(String(format: "%.2f", det.score)))")
            }
            print()
        }

        print("=== HYBRID MODEL (arxiv:2406.07096) ===")
        print("Hybrid 110M: Shared encoder → CTC head for keyword spotting + transcription")
        print("TDT 0.6B: High-quality transcription baseline for comparison")
        print("Key: CTC timestamps are aligned with transcription (same encoder output)")
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio hybrid-test [options] [audio.wav]

            Dual-model approach: CTC 110M for keyword spotting + TDT 0.6B for transcription.

            Options:
              --audio <path>     Path to audio file
              --max-files <n>    Max files to process from earnings dataset (default: 3)
              -h, --help         Show this help

            Examples:
              fluidaudio hybrid-test audio.wav
              fluidaudio hybrid-test --max-files 5
            """)
    }
}
#endif
