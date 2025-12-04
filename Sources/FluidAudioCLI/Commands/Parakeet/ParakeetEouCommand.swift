import AVFoundation
import FluidAudio
import Foundation

struct ParakeetEouCommand {
    static func main(_ arguments: [String]) async {
        let logger = AppLogger(category: "ParakeetEOU")
        logger.info("Starting Parakeet EOU Streaming ASR...")

        var input: String?
        var models: String = "/Users/kikow/brandon/FluidAudioSwift/Models/ParakeetEOU/Streaming"
        var verbose: Bool = false

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
            case "--verbose":
                verbose = true
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        guard let inputPath = input else {
            logger.error("Missing required argument: --input <path>")
            exit(1)
        }
        let inputUrl = URL(fileURLWithPath: inputPath)

        let modelsUrl = URL(fileURLWithPath: models)
        guard FileManager.default.fileExists(atPath: modelsUrl.path) else {
            logger.error("Models directory not found: \(modelsUrl.path)")
            exit(1)
        }

        // 3. Initialize Manager
        let manager = StreamingEouAsrManager()
        do {
            logger.info("Loading models from: \(modelsUrl.path)")
            try await manager.loadModels(modelDir: modelsUrl)
        } catch {
            logger.error("Failed to load models: \(error)")
            exit(1)
        }

        // 4. Load Audio
        logger.info("Loading audio file: \(inputUrl.path)")
        let audioFile: AVAudioFile
        do {
            audioFile = try AVAudioFile(forReading: inputUrl)
        } catch {
            logger.error("Failed to read audio file: \(error)")
            exit(1)
        }
        
        // Debug: Read raw bytes to verify file content
        do {
            let data = try Data(contentsOf: inputUrl)
            logger.info("File Size: \(data.count)")
            let headerBytes = data.prefix(100).map { $0 }
            logger.info("Raw File Bytes (First 100): \(headerBytes)")
            
            let checkOffset = 2414
            if checkOffset < data.count {
                let checkBytes = data.subdata(in: checkOffset..<min(checkOffset+20, data.count)).map { $0 }
                logger.info("Raw File Bytes (Offset 2414): \(checkBytes)")
            }
            
            // Find first non-zero byte after header (approx 100 bytes)
            var firstNonZeroByteIdx: Int?
            for i in 100..<data.count {
                if data[i] != 0 {
                    firstNonZeroByteIdx = i
                    break
                }
            }
            
            if let idx = firstNonZeroByteIdx {
                logger.info("Swift First Non-Zero Byte Index: \(idx)")
                logger.info("Swift First Non-Zero Byte Value: \(data[idx])")
            } else {
                logger.info("Swift Data is all zeros after header")
            }
        } catch {
            logger.error("Failed to read raw file data: \(error)")
        }

        // Manual WAV Loading
        guard let data = try? Data(contentsOf: inputUrl) else {
            logger.error("Failed to read file data")
            exit(1)
        }
        
        // Find "data" chunk
        var dataOffset = 0
        var dataSize = 0
        
        let dataTag = "data".data(using: .utf8)!
        
        // Simple search for "data" tag (not robust for all WAVs but works for this one)
        // A proper parser would walk chunks.
        // RIFF (4) + Size (4) + WAVE (4) + fmt (4) + Size (4) + ...
        
        var offset = 12 // Skip RIFF header
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
        
        if dataOffset == 0 {
            logger.error("Could not find data chunk")
            exit(1)
        }
        
        logger.info("Found data chunk at offset \(dataOffset), size \(dataSize)")
        
        // Create Float32 buffer
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
        
        let buffer = floatBuffer
        
        // Check sample 10800 (0.675s) to match Python
        let checkIdx = 10800
        if checkIdx < Int(buffer.frameLength) {
            var checkSamples: [Float] = []
            if let floatChannelData = buffer.floatChannelData {
                let ptr = floatChannelData[0]
                for i in checkIdx..<min(checkIdx + 10, Int(buffer.frameLength)) {
                    checkSamples.append(ptr[i])
                }
            }
            logger.info("Raw Buffer Samples (Index 10800): \(checkSamples)")
        }
        
        logger.info("Buffer Frame Length: \(buffer.frameLength)")

        // 5. Process
        let startTime = Date()
        do {
            let transcript = try await manager.process(audioBuffer: buffer)
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
}
