import AVFoundation
import CoreML
import Foundation

/// High-level manager for the Parakeet EOU streaming pipeline.
/// Implements the "Pure CoreML" pipeline verified in Python.
public actor StreamingEouAsrManager {
    private let logger = AppLogger(category: "StreamingEOU")
    
    // Debug
    private var processedSteps = 0
    
    // Models
    private var preprocessor: MLModel?
    private var encoder: MLModel?
    private var decoder: MLModel?
    private var joint: MLModel?
    
    // Components
    private var rnntDecoder: RnntDecoder?
    private var encoderState = StreamingEncoderState()
    private let audioConverter = AudioConverter()
    private var tokenizer: Tokenizer?
    
    // Configuration
    private let chunkFrames = 128
    private let shiftFrames = 120
    private let hopLength = 160
    private var chunkSamples: Int { chunkFrames * hopLength } // 21600
    private var shiftSamples: Int { shiftFrames * hopLength } // 19200
    
    // Audio Buffer
    private var audioBuffer: [Float] = []
    
    public init() {}
    
    public func loadModels(modelDir: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly // Verified in Python
        
        logger.info("Loading CoreML models from \(modelDir.path)...")
        
        self.preprocessor = try MLModel(contentsOf: modelDir.appendingPathComponent("parakeet_eou_preprocessor.mlmodelc"), configuration: config)
        self.encoder = try MLModel(contentsOf: modelDir.appendingPathComponent("streaming_encoder.mlmodelc"), configuration: config)
        self.decoder = try MLModel(contentsOf: modelDir.appendingPathComponent("decoder.mlmodelc"), configuration: config)
        self.joint = try MLModel(contentsOf: modelDir.appendingPathComponent("joint_decision.mlmodelc"), configuration: config)
        
        // Load Tokenizer
        let vocabUrl = modelDir.appendingPathComponent("vocab.json")
        self.tokenizer = try Tokenizer(vocabPath: vocabUrl)
        
        self.rnntDecoder = RnntDecoder(decoderModel: self.decoder!, jointModel: self.joint!)
        self.encoderState.reset()
        self.audioBuffer.removeAll()
        
        logger.info("Models loaded successfully.")
    }
    
    public func process(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        // 1. Convert to 16kHz Mono Float32
        let samples = try audioConverter.resampleBuffer(audioBuffer)
        print("Audio Samples Prefix: \(samples.prefix(10))") // DEBUG
        self.audioBuffer.append(contentsOf: samples)
        
        var transcript = ""
        
        // 2. Process chunks
        while self.audioBuffer.count >= chunkSamples {
            // Extract chunk
            let chunk = Array(self.audioBuffer.prefix(chunkSamples))
            
            // 3. Run Pipeline
            let newText = try await processChunk(chunk)
            transcript += newText
            
            // 4. Shift buffer
            self.audioBuffer.removeFirst(shiftSamples)
        }
        
        processedSteps += 1
        return transcript
    }
    
    public func finish() async throws -> String {
        var transcript = ""
        
        // If there is remaining audio, pad it and process
        if !audioBuffer.isEmpty {
            let remaining = audioBuffer.count
            // Only process if we have a significant amount of audio left, or if it's the only audio
            // But for safety, let's just process whatever is left if it's non-empty
            
            let paddingNeeded = chunkSamples - remaining
            if paddingNeeded > 0 {
                audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
            }
            
            // Process final chunk
            // Note: We don't shift here because it's the end
            let chunk = Array(audioBuffer.prefix(chunkSamples))
            let newText = try await processChunk(chunk)
            transcript += newText
            
            // Clear buffer
            audioBuffer.removeAll()
        }
        
        return transcript
    }
    
    public func reset() {
        self.encoderState.reset()
        self.rnntDecoder?.resetState()
        self.audioBuffer.removeAll()
        self.processedSteps = 0
    }
    
    private func processChunk(_ samples: [Float]) async throws -> String {
        guard let preprocessor = preprocessor, let encoder = encoder, let rnntDecoder = rnntDecoder, let tokenizer = tokenizer else {
            throw ASRError.notInitialized
        }
        
        // A. Preprocess
        if self.processedSteps == 0 {
            print("Swift Audio Start: \(samples.prefix(10))")
            if let firstNonZeroIndex = samples.firstIndex(where: { $0 != 0 }) {
                print("Swift First Non-Zero Index: \(firstNonZeroIndex)")
                print("Swift First Non-Zero Value: \(samples[firstNonZeroIndex])")
            } else {
                print("Swift First Chunk is all zeros")
            }
        }
        // Input: audio_signal [1, N], audio_length [1]
        let audioData = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        let ptr = audioData.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        ptr.assign(from: samples, count: samples.count)
        
        // Verify MLMultiArray Content
        if self.processedSteps == 0 {
            var first10: [Float] = []
            for i in 0..<10 {
                first10.append(ptr[i])
            }
            print("Swift MLMultiArray First 10: \(first10)")
        }
        
        let audioLength = try MLMultiArray(shape: [1], dataType: .int32)
        audioLength[0] = NSNumber(value: samples.count)
        
        let prepInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioData),
            "audio_length": MLFeatureValue(multiArray: audioLength)
        ])
        
        let prepOutput = try await preprocessor.prediction(from: prepInput)
        let mel = prepOutput.featureValue(for: "mel")!.multiArrayValue!
        let melLength = prepOutput.featureValue(for: "mel_length")!.multiArrayValue!
        
        // B. Slice Mel (Critical Fix from Python)
        // Mel shape is [1, 128, T]. T might be 136, we need 135.
        let T = mel.shape[2].intValue
        var slicedMel = mel
        var slicedMelLength = melLength
        if T > chunkFrames {
            slicedMel = try sliceMel(mel, length: chunkFrames)
            slicedMelLength[0] = NSNumber(value: chunkFrames)
        }
        
        // C. Encode
        let encInputDict: [String: Any] = [
            "mel": slicedMel,
            "mel_length": slicedMelLength,
            "cache_last_channel": encoderState.cacheLastChannel,
            "cache_last_time": encoderState.cacheLastTime,
            "cache_last_channel_len": encoderState.cacheLastChannelLen
        ]
        let encInput = try MLDictionaryFeatureProvider(dictionary: encInputDict.mapValues { MLFeatureValue(multiArray: $0 as! MLMultiArray) })
        
        let encOutput = try await encoder.prediction(from: encInput)
        var encoded = encOutput.featureValue(for: "encoded")!.multiArrayValue! // [1, 512, T]
        let encodedLen = encOutput.featureValue(for: "encoded_len")!.multiArrayValue!
        

        
        // Update Cache
        encoderState.update(from: encOutput)
        
        // Get Encoded Output
        var finalEncoded = encoded
        
        // Slice Encoded Output (15 frames valid)
        // Python: if ml_encoded.shape[2] > valid_out_len: ml_encoded = ml_encoded[:, :, :valid_out_len]
        // valid_out_len = 15
        let validOutLen = 15
        if encoded.shape[2].intValue > validOutLen {
             encoded = try sliceEncoded(encoded, length: validOutLen)
        }
        
        // D. Decode
        let tokenIds = try rnntDecoder.decode(encoderOutput: encoded)
        
        // E. Detokenize
        return tokenizer.decode(ids: tokenIds)
    }
    
    private func sliceMel(_ mel: MLMultiArray, length: Int) throws -> MLMultiArray {
        // Shape [1, 128, T] -> [1, 128, length]
        let newShape = [mel.shape[0], mel.shape[1], NSNumber(value: length)]
        let newArray = try MLMultiArray(shape: newShape, dataType: .float32)
        
        let channels = mel.shape[1].intValue
        let srcPtr = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)
        let dstPtr = newArray.dataPointer.bindMemory(to: Float.self, capacity: newArray.count)
        
        let stride1 = mel.strides[1].intValue
        let stride2 = mel.strides[2].intValue
        
        for c in 0..<channels {
            for t in 0..<length {
                let srcIdx = c * stride1 + t * stride2
                let dstIdx = c * length + t // newArray is contiguous
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }
        return newArray
    }
    
    private func sliceEncoded(_ encoded: MLMultiArray, length: Int) throws -> MLMultiArray {
        // Shape [1, 512, T] -> [1, 512, length]
        let newShape = [encoded.shape[0], encoded.shape[1], NSNumber(value: length)]
        let newArray = try MLMultiArray(shape: newShape, dataType: .float32)
        
        let channels = encoded.shape[1].intValue
        let srcPtr = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
        let dstPtr = newArray.dataPointer.bindMemory(to: Float.self, capacity: newArray.count)
        
        let stride1 = encoded.strides[1].intValue
        let stride2 = encoded.strides[2].intValue
        
        for c in 0..<channels {
            for t in 0..<length {
                let srcIdx = c * stride1 + t * stride2
                let dstIdx = c * length + t // newArray is contiguous
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }
        return newArray
    }
}
