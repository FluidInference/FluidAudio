import AVFoundation
import FluidAudio
import Foundation
import CoreML

struct ParakeetEouCommand {
    static func main(_ arguments: [String]) async {
        let logger = AppLogger(category: "ParakeetEOU")
        
        var input: String?
        var models: String = "Models/ParakeetEOU/Streaming" // Default relative path
        var verbose: Bool = false
        var benchmark: Bool = false
        var download: Bool = false
        var maxFiles: Int = 100

        // Manual Argument Parsing
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--input":
                if i + 1 < arguments.count {
                    input = arguments[i + 1]
                    i += 1
                }
            case "--models":
                if i + 1 < arguments.count {
                    models = arguments[i + 1]
                    i += 1
                }
            case "--benchmark":
                benchmark = true
            case "--download":
                download = true
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1]) ?? 100
                    i += 1
                }
            case "--verbose":
                verbose = true
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }
        
        // Resolve Models Path
        let modelsUrl = URL(fileURLWithPath: models).standardized
        
        // 1. Download Models if requested or missing
        if download || !FileManager.default.fileExists(atPath: modelsUrl.path) {
            logger.info("Downloading models to: \(modelsUrl.path)")
            do {
                try await downloadModels(to: modelsUrl)
            } catch {
                logger.error("Failed to download models: \(error)")
                exit(1)
            }
        }
        
        // 2. Initialize Manager
        let manager = StreamingEouAsrManager()
        do {
            logger.info("Loading models from: \(modelsUrl.path)")
            try await manager.loadModels(modelDir: modelsUrl)
        } catch {
            logger.error("Failed to load models: \(error)")
            exit(1)
        }

        // 3. Run Benchmark or Single File
        if benchmark {
            await runBenchmark(manager: manager, maxFiles: maxFiles, logger: logger)
        } else {
            guard let inputPath = input else {
                logger.error("Missing required argument: --input <path> (or use --benchmark)")
                exit(1)
            }
            let inputUrl = URL(fileURLWithPath: inputPath)
            await runSingleFile(manager: manager, inputUrl: inputUrl, logger: logger)
        }
    }
    
    static func downloadModels(to destination: URL) async throws {
        let downloader = HuggingFaceDownloader()
        let repoId = "alexwengg/parakeet-realtime-eou-120m-coreml"
        
        print("Fetching file list from \(repoId)...")
        // We want to download the contents of the repo to `destination`
        // The repo structure is flat-ish or has subdirs.
        // We'll use the recursive download helper we added.
        
        try await downloader.downloadRepo(repoId: repoId, destinationDir: destination)
        
        // Compile .mlpackage files
        let enumerator = FileManager.default.enumerator(at: destination, includingPropertiesForKeys: nil)
        while let fileUrl = enumerator?.nextObject() as? URL {
            if fileUrl.pathExtension == "mlpackage" {
                // Check if already compiled
                let compiledUrl = fileUrl.deletingPathExtension().appendingPathExtension("mlmodelc")
                if !FileManager.default.fileExists(atPath: compiledUrl.path) {
                    _ = try downloader.compileModel(at: fileUrl)
                }
            }
        }
    }
    
    static func runSingleFile(manager: StreamingEouAsrManager, inputUrl: URL, logger: AppLogger) async {
        logger.info("Loading audio file: \(inputUrl.path)")
        
        // Manual WAV Loading (reusing previous robust logic)
        guard let data = try? Data(contentsOf: inputUrl) else {
            logger.error("Failed to read file data")
            exit(1)
        }
        
        // Find "data" chunk
        var dataOffset = 0
        var dataSize = 0
        let dataTag = "data".data(using: .utf8)!
        var offset = 12
        while offset < data.count - 8 {
            let chunkId = data.subdata(in: offset..<offset+4)
            let chunkSizeData = data.subdata(in: offset+4..<offset+8)
            let chunkSize = chunkSizeData.withUnsafeBytes { $0.load(as: UInt32.self) }
            if chunkId == dataTag {
                dataOffset = offset + 8
                dataSize = Int(chunkSize)
                break
            }
            offset += 8 + Int(chunkSize)
        }
        
        guard dataOffset > 0 else {
            logger.error("Could not find data chunk")
            exit(1)
        }
        
        let sampleCount = dataSize / 2
        guard let floatBuffer = AVAudioPCMBuffer(pcmFormat: AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!, frameCapacity: AVAudioFrameCount(sampleCount)) else {
            logger.error("Failed to create buffer")
            exit(1)
        }
        floatBuffer.frameLength = AVAudioFrameCount(sampleCount)
        if let floatChannelData = floatBuffer.floatChannelData {
            let ptr = floatChannelData[0]
            data.withUnsafeBytes { rawBuffer in
                let int16Ptr = rawBuffer.baseAddress!.advanced(by: dataOffset).assumingMemoryBound(to: Int16.self)
                for i in 0..<sampleCount {
                    ptr[i] = Float(int16Ptr[i]) / 32768.0
                }
            }
        }
        
        let startTime = Date()
        do {
            let transcript = try await manager.process(audioBuffer: floatBuffer)
            let duration = Date().timeIntervalSince(startTime)
            
            logger.info("--- Transcript ---")
            print(transcript)
            logger.info("------------------")
            logger.info("Processing time: \(String(format: "%.3f", duration))s")
        } catch {
            logger.error("Processing failed: \(error)")
            exit(1)
        }
    }
    
    static func runBenchmark(manager: StreamingEouAsrManager, maxFiles: Int, logger: AppLogger) async {
        logger.info("Starting Benchmark (Max Files: \(maxFiles))...")
        
        // 1. Download LibriSpeech
        let benchmark = ASRBenchmark()
        do {
            try await benchmark.downloadLibriSpeech(subset: "test-clean")
        } catch {
            logger.error("Failed to download LibriSpeech: \(error)")
            exit(1)
        }
        
        // 2. List Files
        let datasetPath = benchmark.getLibriSpeechDirectory().appendingPathComponent("test-clean")
        var files: [(url: URL, text: String)] = []
        
        let enumerator = FileManager.default.enumerator(at: datasetPath, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            if url.pathExtension == "txt" {
                // Parse transcript file
                if let content = try? String(contentsOf: url, encoding: .utf8) {
                    let lines = content.components(separatedBy: CharacterSet.newlines)
                    for line in lines {
                        let parts = line.split(separator: " ", maxSplits: 1)
                        if parts.count == 2 {
                            let fileId = String(parts[0])
                            let text = String(parts[1])
                            let audioUrl = url.deletingLastPathComponent().appendingPathComponent("\(fileId).flac")
                            if FileManager.default.fileExists(atPath: audioUrl.path) {
                                files.append((audioUrl, text))
                            }
                        }
                    }
                }
            }
        }
        
        files.shuffle()
        let testFiles = Array(files.prefix(maxFiles))
        logger.info("Found \(files.count) files, running on \(testFiles.count)")
        
        var totalWer = 0.0
        var totalTime = 0.0
        var totalAudioDuration = 0.0
        var results: [BenchmarkFileResult] = []
        
        for (i, file) in testFiles.enumerated() {
            let (audioUrl, reference) = file
            // Convert FLAC to WAV/Float buffer
            // We can use AudioConverter helper if available, or just AVAudioFile
            
            do {
                let audioFile = try AVAudioFile(forReading: audioUrl)
                let format = audioFile.processingFormat
                let frameCount = AVAudioFrameCount(audioFile.length)
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
                try audioFile.read(into: buffer)
                
                // Resample if needed (ASR expects 16k mono)
                // StreamingEouAsrManager handles resampling internally in `process(audioBuffer:)`?
                // Yes, it calls `audioConverter.resampleBuffer(audioBuffer)`
                
                let startTime = Date()
                let transcript = try await manager.process(audioBuffer: buffer)
                let duration = Date().timeIntervalSince(startTime)
                
                let wer = calculateWer(hypothesis: transcript, reference: reference)
                totalWer += wer
                totalTime += duration
                
                // Calculate audio duration
                let audioDuration = Double(frameCount) / format.sampleRate
                totalAudioDuration += audioDuration
                
                logger.info("[\(i+1)/\(testFiles.count)] WER: \(String(format: "%.2f", wer * 100))% | RTFx: \(String(format: "%.2f", audioDuration/duration)) | Ref: \"\(reference.prefix(30))...\" | Hyp: \"\(transcript.prefix(30))...\"")
                
                results.append(BenchmarkFileResult(
                    filename: audioUrl.lastPathComponent,
                    wer: wer,
                    rtfx: audioDuration/duration,
                    reference: reference,
                    hypothesis: transcript,
                    audioDuration: audioDuration,
                    processingTime: duration
                ))
                
            } catch {
                logger.error("Failed to process \(audioUrl.lastPathComponent): \(error)")
            }
        }
        
        let avgWer = totalWer / Double(testFiles.count)
        let avgRtf = totalAudioDuration / totalTime
        
        logger.info("--- Benchmark Results ---")
        logger.info("Average WER: \(String(format: "%.2f", avgWer * 100))%")
        logger.info("Average RTFx: \(String(format: "%.2f", avgRtf))")
        logger.info("Total Audio: \(String(format: "%.2f", totalAudioDuration))s")
        logger.info("Total Time: \(String(format: "%.2f", totalTime))s")
        
        // Save to JSON
        let sortedResults = results.sorted { $0.wer > $1.wer }
        
        let jsonResults = BenchmarkJSONOutput(
            averageWer: avgWer,
            averageRtfx: avgRtf,
            totalAudioDuration: totalAudioDuration,
            totalProcessingTime: totalTime,
            results: sortedResults
        )
        
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(jsonResults)
            let outputPath = URL(fileURLWithPath: "benchmark_results.json")
            try data.write(to: outputPath)
            logger.info("Results saved to \(outputPath.path)")
        } catch {
            logger.error("Failed to save results to JSON: \(error)")
        }
    }
    
    struct BenchmarkJSONOutput: Codable {
        let averageWer: Double
        let averageRtfx: Double
        let totalAudioDuration: Double
        let totalProcessingTime: Double
        let results: [BenchmarkFileResult]
    }
    
    struct BenchmarkFileResult: Codable {
        let filename: String
        let wer: Double
        let rtfx: Double
        let reference: String
        let hypothesis: String
        let audioDuration: Double
        let processingTime: Double
    }
    
    static func calculateWer(hypothesis: String, reference: String) -> Double {
        let hWords = hypothesis.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let rWords = reference.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        
        let d = levenshtein(a: hWords, b: rWords)
        if rWords.isEmpty { return hWords.isEmpty ? 0.0 : 1.0 }
        return Double(d) / Double(rWords.count)
    }
    
    static func levenshtein<T: Equatable>(a: [T], b: [T]) -> Int {
        let m = a.count
        let n = b.count
        
        if m == 0 { return n }
        if n == 0 { return m }
        
        var matrix = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
        
        for i in 1...m { matrix[i][0] = i }
        for j in 1...n { matrix[0][j] = j }
        
        for i in 1...m {
            for j in 1...n {
                if a[i - 1] == b[j - 1] {
                    matrix[i][j] = matrix[i - 1][j - 1]
                } else {
                    matrix[i][j] = min(
                        matrix[i - 1][j] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j - 1] + 1
                    )
                }
            }
        }
        return matrix[m][n]
    }
}
