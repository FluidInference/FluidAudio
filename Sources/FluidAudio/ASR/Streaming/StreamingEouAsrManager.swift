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
    private var preEncode: MLModel?
    private var conformer: MLModel?
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
    
    public private(set) var configuration: MLModelConfiguration
    public let debugFeatures: Bool
    private var debugFeatureBuffer: [Float] = []

    public init(configuration: MLModelConfiguration = MLModelConfiguration(), debugFeatures: Bool = false) {
        self.configuration = configuration
        self.debugFeatures = debugFeatures
    }
    
    public func loadModels(modelDir: URL) async throws {
        logger.info("Loading CoreML models from \(modelDir.path)...")
        
        self.preprocessor = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("parakeet_eou_preprocessor.mlmodelc"), configuration: self.configuration)
        self.preEncode = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("pre_encode.mlmodelc"), configuration: self.configuration)
        self.conformer = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("conformer_streaming.mlmodelc"), configuration: self.configuration)
        self.decoder = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("decoder.mlmodelc"), configuration: self.configuration)
        self.joint = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("joint_decision.mlmodelc"), configuration: self.configuration)
        
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
    
    public func reset() async {
        audioBuffer.removeAll()
        debugFeatureBuffer.removeAll()
        await encoderState.reset()
        await rnntDecoder?.resetState() // Assuming rnntDecoder has a resetState method
        self.processedSteps = 0
    }
    
    private func processChunk(_ samples: [Float]) async throws -> String {
        guard let preprocessor = preprocessor, let preEncode = preEncode, let conformer = conformer, let rnntDecoder = rnntDecoder, let tokenizer = tokenizer else {
            throw ASRError.notInitialized
        }
        
        // A. Preprocess
        // Input: audio_signal [1, N], audio_length [1]
        let audioData = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        let ptr = audioData.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        ptr.assign(from: samples, count: samples.count)
        
        let audioLength = try MLMultiArray(shape: [1], dataType: .int32)
        audioLength[0] = NSNumber(value: samples.count)
        
        let prepInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioData),
            "audio_length": MLFeatureValue(multiArray: audioLength)
        ])
        
        let prepOutput = try await preprocessor.prediction(from: prepInput)
        let mel = prepOutput.featureValue(for: "mel")!.multiArrayValue!
        
        // Debug: Accumulate features
        if debugFeatures {
            let melMultiArray = mel
            let count = melMultiArray.count
            let melPtr = melMultiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count {
                debugFeatureBuffer.append(melPtr[i])
            }
        }
        
        let melLength = prepOutput.featureValue(for: "mel_length")!.multiArrayValue!
        
        // B. Slice Mel
        // Mel shape is [1, 128, T]. T might be 136, we need 135.
        let T = mel.shape[2].intValue
        var slicedMel = mel
        var slicedMelLength = melLength
        if T > chunkFrames {
            slicedMel = try sliceMel(mel, length: chunkFrames)
            slicedMelLength[0] = NSNumber(value: chunkFrames)
        }
        
        // C. Pre-Encode
        let preEncodeInputDict: [String: Any] = [
            "mel": slicedMel,
            "mel_length": slicedMelLength,
            "pre_cache": encoderState.preCache
        ]
        let preEncodeInput = try MLDictionaryFeatureProvider(dictionary: preEncodeInputDict.mapValues { MLFeatureValue(multiArray: $0 as! MLMultiArray) })
        let preEncodeOutput = try await preEncode.prediction(from: preEncodeInput)
        
        let preEncoded = preEncodeOutput.featureValue(for: "pre_encoded")!.multiArrayValue!
        let preEncodedLen = preEncodeOutput.featureValue(for: "pre_encoded_len")!.multiArrayValue!
        
        // Update Pre-Cache
        encoderState.updatePreCache(from: preEncodeOutput)
        
        // D. Conformer Encode
        let confInputDict: [String: Any] = [
            "pre_encoded": preEncoded,
            "pre_encoded_length": preEncodedLen,
            "cache_last_channel": encoderState.cacheLastChannel,
            "cache_last_time": encoderState.cacheLastTime,
            "cache_last_channel_len": encoderState.cacheLastChannelLen
        ]
        let confInput = try MLDictionaryFeatureProvider(dictionary: confInputDict.mapValues { MLFeatureValue(multiArray: $0 as! MLMultiArray) })
        let confOutput = try await conformer.prediction(from: confInput)
        
        var encoded = confOutput.featureValue(for: "encoder")!.multiArrayValue!
        
        // Update Cache
        encoderState.update(from: confOutput)
        
        // Slice Encoded Output (15 frames valid)
        let validOutLen = 15
        if encoded.shape[2].intValue > validOutLen {
             encoded = try sliceEncoded(encoded, length: validOutLen)
        }
        
        // E. Decode
        let tokenIds = try rnntDecoder.decode(encoderOutput: encoded)
        
        // E. Detokenize
        return tokenizer.decode(ids: tokenIds)
    }
    
    public func saveDebugFeatures(to url: URL) throws {
        let outputData: [String: Any] = [
            "mel_features": debugFeatureBuffer,
            "count": debugFeatureBuffer.count
        ]
        
        let data = try JSONSerialization.data(withJSONObject: outputData, options: .prettyPrinted)
        try data.write(to: url)
        logger.info("Dumped \(debugFeatureBuffer.count) features to \(url.path)")
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
