import AVFoundation
import FluidAudio
import Foundation
import OSLog

struct LibriSpeechBenchmarkCommand {
    private static let logger = AppLogger(category: "LibriSpeechBenchmark")

    static func run(arguments: [String]) async {
        // Parse arguments manually
        var limit: Int = 100
        var cacheDir: String?
        var forceDownload: Bool = false
        
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--limit", "-l":
                if i + 1 < arguments.count {
                    limit = Int(arguments[i + 1]) ?? 100
                    i += 1
                }
            case "--cache-dir":
                if i + 1 < arguments.count {
                    cacheDir = arguments[i + 1]
                    i += 1
                }
            case "--force", "-f":
                forceDownload = true
            default:
                break
            }
            i += 1
        }
        
        do {
            try await runBenchmark(limit: limit, cacheDir: cacheDir, forceDownload: forceDownload)
        } catch {
            logger.error("Benchmark failed: \(error)")
        }
    }

    static func runBenchmark(limit: Int, cacheDir: String?, forceDownload: Bool) async throws {
        logger.info("Starting LibriSpeech Benchmark...")
        
        // 1. Setup paths
        let baseDir = cacheDir.map { URL(fileURLWithPath: $0) } ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!.appendingPathComponent("FluidAudio/LibriSpeech")
        let testCleanDir = baseDir.appendingPathComponent("LibriSpeech/test-clean")
        
        // 2. Download if needed
        if forceDownload || !FileManager.default.fileExists(atPath: testCleanDir.path) {
            try await downloadLibriSpeech(to: baseDir)
        } else {
            logger.info("Dataset found at \(testCleanDir.path)")
        }
        
        // 3. Load samples
        let samples = try loadSamples(from: testCleanDir, limit: limit)
        logger.info("Loaded \(samples.count) samples")
        
        // 4. Initialize Canary
        logger.info("Initializing Canary model...")
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let modelDir = appSupport.appendingPathComponent("FluidAudio").appendingPathComponent("canary")
        
        let models = try await CanaryModels.load(from: modelDir)
        let manager = CanaryManager()
        manager.initialize(models: models)
        
        // 5. Run Benchmark
        var totalWER = 0.0
        var totalDuration = 0.0
        var totalProcessingTime = 0.0
        var processedCount = 0
        
        logger.info("--------------------------------------------------------------------------------")
        logger.info(String(format: "%-40@ | %-10@ | %-10@", "File", "WER", "RTFx"))
        logger.info("--------------------------------------------------------------------------------")
        
        for sample in samples {
            logger.info("Processing \(sample.audioPath.split(separator: "/").last!)...")
            let audioUrl = URL(fileURLWithPath: sample.audioPath)
            
            // Load audio to get duration
            let audioFile = try AVAudioFile(forReading: audioUrl)
            let duration = Double(audioFile.length) / audioFile.fileFormat.sampleRate
            
            let startTime = Date()
            
            // Read audio data
            let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audioFile.length)) else { continue }
            try audioFile.read(into: buffer)
            
            let channelData = buffer.floatChannelData![0]
            let sampleCount = Int(buffer.frameLength)
            var floatSamples = Array(UnsafeBufferPointer(start: channelData, count: sampleCount))
            
            // Pad with silence if needed (Canary streaming window)
            // 5 seconds of silence
            let silence = Array(repeating: Float(0.0), count: 16000 * 5)
            floatSamples.append(contentsOf: silence)
            
            // Reset manager state for new file
            manager.reset()
            
            // Stream processing
            let chunkSize = 32000 // 2s
            var transcription = ""
            
            // Simulating streaming
            var cursor = 0
            while cursor < floatSamples.count {
                let end = min(cursor + chunkSize, floatSamples.count)
                let chunk = Array(floatSamples[cursor..<end])
                let text = try await manager.processStreamingChunk(chunk)
                transcription += text
                cursor += chunkSize
            }
            
            let processingTime = Date().timeIntervalSince(startTime)
            
            // Calculate WER
            let metrics = WERCalculator.calculateWERAndCER(hypothesis: transcription, reference: sample.transcript)
            
            totalWER += metrics.wer
            totalDuration += duration
            totalProcessingTime += processingTime
            processedCount += 1
            
            let rtfx = duration / processingTime
            
            logger.info(String(format: "%-40@ | %6.2f%%    | %6.2fx", audioUrl.lastPathComponent, metrics.wer * 100, rtfx))
        }
        
        print(String(repeating: "-", count: 80))
        let avgWER = processedCount > 0 ? totalWER / Double(processedCount) : 0.0
        let avgRTFx = totalProcessingTime > 0 ? totalDuration / totalProcessingTime : 0.0
        
        logger.info("Benchmark Complete")
        logger.info("Average WER: \(String(format: "%.2f", avgWER * 100))%")
        logger.info("Average RTFx: \(String(format: "%.2f", avgRTFx))x")
    }

    private static func downloadLibriSpeech(to directory: URL) async throws {
        let url = URL(string: "https://www.openslr.org/resources/12/test-clean.tar.gz")!
        let destination = directory.appendingPathComponent("test-clean.tar.gz")
        
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        
        logger.info("Downloading LibriSpeech test-clean...")
        let (data, _) = try await URLSession.shared.data(from: url)
        try data.write(to: destination)
        
        logger.info("Extracting...")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", destination.path, "-C", directory.path]
        try process.run()
        process.waitUntilExit()
        
        try FileManager.default.removeItem(at: destination)
    }

    struct Sample {
        let audioPath: String
        let transcript: String
    }

    private static func loadSamples(from directory: URL, limit: Int) throws -> [Sample] {
        var samples: [Sample] = []
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        
        while let fileUrl = enumerator?.nextObject() as? URL {
            if fileUrl.lastPathComponent.hasSuffix(".trans.txt") {
                let content = try String(contentsOf: fileUrl)
                let lines = content.components(separatedBy: .newlines)
                let dir = fileUrl.deletingLastPathComponent()
                
                for line in lines where !line.isEmpty {
                    let parts = line.split(separator: " ", maxSplits: 1)
                    if parts.count >= 2 {
                        let id = String(parts[0])
                        let text = String(parts[1...].joined(separator: " "))
                        let audioPath = dir.appendingPathComponent("\(id).flac").path
                        
                        if FileManager.default.fileExists(atPath: audioPath) {
                            samples.append(Sample(audioPath: audioPath, transcript: text))
                            if samples.count >= limit { return samples }
                        }
                    }
                }
            }
        }
        return samples
    }
}
