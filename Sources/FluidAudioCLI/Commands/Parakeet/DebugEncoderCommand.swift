import AVFoundation
import FluidAudio
import Foundation
import CoreML

struct DebugEncoderCommand {
    static func main(_ arguments: [String]) async {
        let logger = AppLogger(category: "DebugEncoder")
        
        var input: String?
        var models: String = "temp_swift_models/StreamingModelConvert" // Default to generated models
        var outputDir: String = "ReferenceOutputs/EncoderDebug"
        
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
            case "--output-dir":
                if i + 1 < arguments.count {
                    outputDir = arguments[i + 1]
                    i += 1
                }
            default:
                break
            }
            i += 1
        }
        
        guard let inputPath = input else {
            logger.error("Missing required argument: --input <path>")
            exit(1)
        }
        
        let modelsUrl = URL(fileURLWithPath: models).standardized
        let inputUrl = URL(fileURLWithPath: inputPath)
        let outputUrl = URL(fileURLWithPath: outputDir)
        
        try? FileManager.default.createDirectory(at: outputUrl, withIntermediateDirectories: true)
        
        logger.info("Loading models from: \(modelsUrl.path)")
        
        do {
            let debugger = DebugEncoderManager()
            try await debugger.loadModels(modelDir: modelsUrl)
            try await debugger.process(audioUrl: inputUrl, outputDir: outputUrl)
        } catch {
            logger.error("Failed: \(error)")
            exit(1)
        }
    }
}

class DebugEncoderManager {
    private var preprocessor: MLModel?
    private var encoder: MLModel?
    private var encoderState = StreamingEncoderState()
    private let audioConverter = AudioConverter()
    
    // Configuration (Must match StreamingEouAsrManager)
    private let chunkFrames = 128
    private let shiftFrames = 120
    private let hopLength = 160
    private var chunkSamples: Int { chunkFrames * hopLength } // 21600
    private var shiftSamples: Int { shiftFrames * hopLength } // 19200
    
    func loadModels(modelDir: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        
        self.preprocessor = try MLModel(contentsOf: modelDir.appendingPathComponent("parakeet_eou_preprocessor.mlmodelc"), configuration: config)
        self.encoder = try MLModel(contentsOf: modelDir.appendingPathComponent("streaming_encoder.mlmodelc"), configuration: config)
        self.encoderState.reset()
    }
    
    func process(audioUrl: URL, outputDir: URL) async throws {
        // Load Audio
        let audioFile = try AVAudioFile(forReading: audioUrl)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        try audioFile.read(into: buffer)
        
        // Convert
        let samples = try audioConverter.resampleBuffer(buffer)
        print("Total Samples: \(samples.count)")
        
        var audioBuffer = samples
        var step = 0
        
        while audioBuffer.count >= chunkSamples {
            let chunk = Array(audioBuffer.prefix(chunkSamples))
            
            // Run Pipeline
            let encoded = try processChunk(chunk)
            
            // Save Output
            let filename = outputDir.appendingPathComponent("swift_encoder_step_\(step).bin")
            try save(multiArray: encoded, to: filename)
            print("Step \(step): Saved \(filename) (Shape: \(encoded.shape))")
            
            // Shift
            audioBuffer.removeFirst(shiftSamples)
            step += 1
            
            if step >= 5 { break }
        }
    }
    
    private func processChunk(_ samples: [Float]) throws -> MLMultiArray {
        guard let preprocessor = preprocessor, let encoder = encoder else {
            fatalError("Models not loaded")
        }
        
        // 1. Preprocess
        let audioData = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        for (i, sample) in samples.enumerated() {
            audioData[i] = NSNumber(value: sample)
        }
        let audioLength = try MLMultiArray(shape: [1], dataType: .int32)
        audioLength[0] = NSNumber(value: samples.count)
        
        let prepInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": audioData,
            "audio_length": audioLength
        ])
        let prepOutput = try preprocessor.prediction(from: prepInput)
        let mel = prepOutput.featureValue(for: "mel")!.multiArrayValue!
        let melLength = prepOutput.featureValue(for: "mel_length")!.multiArrayValue!
        
        // Slice Mel to chunkFrames (136) if needed
        // Preprocessor might output 137
        let slicedMel = try sliceMel(mel, length: chunkFrames)
        
        // 2. Encoder
        let encInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": slicedMel,
            "mel_length": melLength, // Length might be 137, but encoder expects 136? 
            // Actually mel_length is just scalar. We should probably cap it too.
            "cache_last_channel": encoderState.cacheLastChannel,
            "cache_last_time": encoderState.cacheLastTime,
            "cache_last_channel_len": encoderState.cacheLastChannelLen
        ])
        
        let encOutput = try encoder.prediction(from: encInput)
        let encoded = encOutput.featureValue(for: "encoded")!.multiArrayValue!
        
        // Update State
        encoderState.cacheLastChannel = encOutput.featureValue(for: "new_cache_last_channel")!.multiArrayValue!
        encoderState.cacheLastTime = encOutput.featureValue(for: "new_cache_last_time")!.multiArrayValue!
        encoderState.cacheLastChannelLen = encOutput.featureValue(for: "new_cache_last_channel_len")!.multiArrayValue!
        
        return encoded
    }
    
    private func sliceMel(_ mel: MLMultiArray, length: Int) throws -> MLMultiArray {
        // mel shape: [1, 128, T]
        let shape = [1, 128, NSNumber(value: length)]
        let newMel = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Copy
        for c in 0..<128 {
            for t in 0..<length {
                let index = [0, NSNumber(value: c), NSNumber(value: t)] as [NSNumber]
                newMel[index] = mel[index]
            }
        }
        return newMel
    }
    
    private func save(multiArray: MLMultiArray, to url: URL) throws {
        let count = multiArray.count
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        let buffer = UnsafeBufferPointer(start: ptr, count: count)
        let data = Data(buffer: buffer)
        try data.write(to: url)
    }
}
