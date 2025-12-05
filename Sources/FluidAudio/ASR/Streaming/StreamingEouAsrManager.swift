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
    // private var preEncode: MLModel? // Removed
    private var encoder: MLModel? // Renamed from conformer
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
    
    // Mel Cache for Monolithic Encoder
    private var melCache: MLMultiArray?

    public init(configuration: MLModelConfiguration = MLModelConfiguration(), debugFeatures: Bool = false) {
        self.configuration = configuration
        self.debugFeatures = debugFeatures
    }
    
    public func loadModels(modelDir: URL) async throws {
        logger.info("Loading CoreML models from \(modelDir.path)...")
        
        self.preprocessor = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("parakeet_eou_preprocessor.mlmodelc"), configuration: self.configuration)
        self.encoder = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("parakeet_eou_encoder_streaming.mlmodelc"), configuration: self.configuration)
        self.decoder = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("parakeet_eou_decoder.mlmodelc"), configuration: self.configuration)
        self.joint = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("parakeet_eou_joint_decision_single_step.mlmodelc"), configuration: self.configuration)
        
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
        // print("Audio Samples Prefix: \(samples.prefix(10))") // DEBUG
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
        melCache = nil
        await encoderState.reset()
        await rnntDecoder?.resetState() // Assuming rnntDecoder has a resetState method
        self.processedSteps = 0
    }
    
    private func processChunk(_ samples: [Float]) async throws -> String {
        guard let preprocessor = preprocessor, let encoder = encoder, let rnntDecoder = rnntDecoder, let tokenizer = tokenizer else {
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
        let melLength = prepOutput.featureValue(for: "mel_length")!.multiArrayValue!
        
        // Debug: Accumulate features
        if debugFeatures {
            let melMultiArray = mel
            let count = melMultiArray.count
            let melPtr = melMultiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count {
                debugFeatureBuffer.append(melPtr[i])
            }
        }
        
        // B. Prepare Encoder Input (Mel Caching)
        // Monolithic encoder expects [Cache(9) + Current(129)] = 138 frames
        var inputMel: MLMultiArray
        let cacheSize = 9
        
        if let cache = melCache {
            inputMel = try concatenateMel(cache, mel)
        } else {
            // First chunk: Pad with zeros
            let zeroCache = try createZeroMel(channels: mel.shape[1].intValue, length: cacheSize)
            inputMel = try concatenateMel(zeroCache, mel)
        }
        
        // Update Cache (Keep last 9 frames of CURRENT mel)
        // Note: We cache from 'mel', not 'inputMel'
        let currentFrames = mel.shape[2].intValue
        if currentFrames >= cacheSize {
            melCache = try sliceMel(mel, start: currentFrames - cacheSize, length: cacheSize)
        } else {
            // Should not happen if chunk is large enough, but handle safely?
            // Just keep what we have? Or pad?
            // For now assume chunk is large enough (129 frames > 9)
            melCache = mel
        }
        
        let inputMelLength = try MLMultiArray(shape: [1], dataType: .int32)
        inputMelLength[0] = NSNumber(value: inputMel.shape[2].intValue)
        
        // C. Encode
        let encInputDict: [String: Any] = [
            "mel": inputMel,
            "mel_length": inputMelLength,
            "cache_last_channel": encoderState.cacheLastChannel,
            "cache_last_time": encoderState.cacheLastTime,
            "cache_last_channel_len": encoderState.cacheLastChannelLen
        ]
        let encInput = try MLDictionaryFeatureProvider(dictionary: encInputDict.mapValues { MLFeatureValue(multiArray: $0 as! MLMultiArray) })
        let encOutput = try await encoder.prediction(from: encInput)
        
        var encoded = encOutput.featureValue(for: "encoder")!.multiArrayValue!
        
        // Update Cache
        encoderState.update(from: encOutput)
        
        // Slice Encoded Output (15 frames valid)
        let validOutLen = 15
        if encoded.shape[2].intValue > validOutLen {
             encoded = try sliceEncoded(encoded, length: validOutLen)
        }
        
        // D. Decode
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

    private func sliceMel(_ mel: MLMultiArray, start: Int = 0, length: Int) throws -> MLMultiArray {
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
                let srcIdx = c * stride1 + (start + t) * stride2
                let dstIdx = c * length + t // newArray is contiguous
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }
        return newArray
    }
    
    private func concatenateMel(_ mel1: MLMultiArray, _ mel2: MLMultiArray) throws -> MLMultiArray {
        // Shape [1, 128, T1] + [1, 128, T2] -> [1, 128, T1+T2]
        let t1 = mel1.shape[2].intValue
        let t2 = mel2.shape[2].intValue
        let totalLength = t1 + t2
        
        let newShape = [mel1.shape[0], mel1.shape[1], NSNumber(value: totalLength)]
        let newArray = try MLMultiArray(shape: newShape, dataType: .float32)
        
        let channels = mel1.shape[1].intValue
        let srcPtr1 = mel1.dataPointer.bindMemory(to: Float.self, capacity: mel1.count)
        let srcPtr2 = mel2.dataPointer.bindMemory(to: Float.self, capacity: mel2.count)
        let dstPtr = newArray.dataPointer.bindMemory(to: Float.self, capacity: newArray.count)
        
        let stride1_1 = mel1.strides[1].intValue
        let stride2_1 = mel1.strides[2].intValue
        let stride1_2 = mel2.strides[1].intValue
        let stride2_2 = mel2.strides[2].intValue
        
        for c in 0..<channels {
            // Copy mel1
            for t in 0..<t1 {
                let srcIdx = c * stride1_1 + t * stride2_1
                let dstIdx = c * totalLength + t
                dstPtr[dstIdx] = srcPtr1[srcIdx]
            }
            // Copy mel2
            for t in 0..<t2 {
                let srcIdx = c * stride1_2 + t * stride2_2
                let dstIdx = c * totalLength + (t1 + t)
                dstPtr[dstIdx] = srcPtr2[srcIdx]
            }
        }
        return newArray
    }
    
    private func createZeroMel(channels: Int, length: Int) throws -> MLMultiArray {
        let shape = [NSNumber(value: 1), NSNumber(value: channels), NSNumber(value: length)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Zero init
        let count = array.count
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        ptr.update(repeating: 0, count: count)
        
        return array
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
    
    private func padMel(_ mel: MLMultiArray, targetLength: Int) throws -> MLMultiArray {
        // Shape [1, 128, T] -> [1, 128, targetLength]
        let newShape = [mel.shape[0], mel.shape[1], NSNumber(value: targetLength)]
        let newArray = try MLMultiArray(shape: newShape, dataType: .float32)
        
        // Zero init
        let count = newArray.count
        let dstPtr = newArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        dstPtr.update(repeating: 0, count: count)
        
        let channels = mel.shape[1].intValue
        let currentLength = mel.shape[2].intValue
        let srcPtr = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)
        
        let stride1 = mel.strides[1].intValue
        let stride2 = mel.strides[2].intValue
        
        for c in 0..<channels {
            for t in 0..<currentLength {
                let srcIdx = c * stride1 + t * stride2
                let dstIdx = c * targetLength + t // newArray is contiguous
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }
        return newArray
    }
}
