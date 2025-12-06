import AVFoundation
import CoreML
import Foundation

/// High-level manager for the Parakeet EOU streaming pipeline.
/// Implements the "Pure CoreML" pipeline verified in Python.
public actor StreamingEouAsrManager {
    private let logger = AppLogger(category: "StreamingEOU")
    
    // Debug
    private var processedSteps = 0
    private var processedChunks = 0
    
    // Models
    private var preprocessor: MLModel?
    private var streamingEncoder: MLModel? // Single Loopback Model
    private var decoder: MLModel?
    private var joint: MLModel?
    
    // Components
    private var rnntDecoder: RnntDecoder?
    private let audioConverter = AudioConverter()
    private var tokenizer: Tokenizer?
    
    // Configuration
    // 160ms chunk size (matches NeMo reference benchmark)
    // 16 frames * 10ms = 160ms
    private let chunkFrames = 16
    private let hopLength = 160
    private var chunkSamples: Int { chunkFrames * hopLength }
    
    // Audio Buffer
    private var audioBuffer: [Float] = []
    
    public private(set) var configuration: MLModelConfiguration
    public let debugFeatures: Bool
    private var debugFeatureBuffer: [Float] = []
    
    // --- Loopback States ---
    // 1. Pre-Cache (Audio Context) [1, 128, 16]
    private var preCache: MLMultiArray?
    
    // 2. Conformer Caches
    // cache_last_channel: [17, 1, 70, 512]
    // cache_last_time: [17, 1, 512, 8]
    // cache_last_channel_len: [1]
    private var cacheLastChannel: MLMultiArray?
    private var cacheLastTime: MLMultiArray?
    private var cacheLastChannelLen: MLMultiArray?

    public init(configuration: MLModelConfiguration = MLModelConfiguration(), debugFeatures: Bool = false) {
        self.configuration = configuration
        self.debugFeatures = debugFeatures
    }
    
    public func loadModels(modelDir: URL) async throws {
        logger.info("Loading CoreML models from \(modelDir.path)...")
        
        self.preprocessor = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("parakeet_eou_preprocessor.mlmodelc"), configuration: self.configuration)
        self.streamingEncoder = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("streaming_encoder.mlmodelc"), configuration: self.configuration)
        self.decoder = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("decoder.mlmodelc"), configuration: self.configuration)
        self.joint = try await MLModel.load(contentsOf: modelDir.appendingPathComponent("joint_decision.mlmodelc"), configuration: self.configuration)
        
        // Load Tokenizer
        let vocabUrl = modelDir.appendingPathComponent("vocab.json")
        self.tokenizer = try Tokenizer(vocabPath: vocabUrl)
        
        self.rnntDecoder = RnntDecoder(decoderModel: self.decoder!, jointModel: self.joint!)
        
        // Initialize States
        try self.resetStates()
        
        self.audioBuffer.removeAll()
        
        logger.info("Models loaded successfully.")
    }
    
    private func resetStates() throws {
        // Initialize with Zeros
        // pre_cache: [1, 128, 16]
        self.preCache = try MLMultiArray(shape: [1, 128, 16], dataType: .float32)
        self.preCache?.reset(to: 0)
        
        // cache_last_channel: [17, 1, 70, 512]
        self.cacheLastChannel = try MLMultiArray(shape: [17, 1, 70, 512], dataType: .float32)
        self.cacheLastChannel?.reset(to: 0)
        
        // cache_last_time: [17, 1, 512, 8]
        self.cacheLastTime = try MLMultiArray(shape: [17, 1, 512, 8], dataType: .float32)
        self.cacheLastTime?.reset(to: 0)
        
        // cache_last_channel_len: [1]
        self.cacheLastChannelLen = try MLMultiArray(shape: [1], dataType: .int32)
        self.cacheLastChannelLen?.reset(to: 0)
    }
    
    public func process(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        // 1. Convert to 16kHz Mono Float32
        let samples = try audioConverter.resampleBuffer(audioBuffer)
        self.audioBuffer.append(contentsOf: samples)
        
        var transcript = ""
        
        // 2. Process chunks
        while self.audioBuffer.count >= chunkSamples {
            // Extract chunk
            let chunk = Array(self.audioBuffer.prefix(chunkSamples))
            
            // 3. Run Pipeline
            let newText = try await processChunk(chunk)
            transcript += newText
            
            // 4. Shift buffer (No overlap needed, model handles context)
            self.audioBuffer.removeFirst(chunkSamples)
        }
        
        processedSteps += 1
        return transcript
    }
    
    public func finish() async throws -> String {
        var transcript = ""
        
        // 1. Process remaining audio (padded)
        if !audioBuffer.isEmpty {
            let remaining = audioBuffer.count
            let paddingNeeded = chunkSamples - remaining
            
            if paddingNeeded > 0 {
                audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
            }
            
            // Process final chunk
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
        try? resetStates()
        await rnntDecoder?.resetState()
        self.processedSteps = 0
    }
    
    public func injectSilence(_ seconds: Double) {
        let silenceSamples = Int(seconds * 16000)
        audioBuffer.append(contentsOf: Array(repeating: 0.0, count: silenceSamples))
    }
    
    private func processChunk(_ samples: [Float]) async throws -> String {
        guard let preprocessor = preprocessor, 
              let streamingEncoder = streamingEncoder, 
              let rnntDecoder = rnntDecoder, 
              let tokenizer = tokenizer,
              let preCache = preCache,
              let cacheLastChannel = cacheLastChannel,
              let cacheLastTime = cacheLastTime,
              let cacheLastChannelLen = cacheLastChannelLen
        else {
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
        
        // B. Streaming Encoder (Loopback)
        // Inputs: audio_signal (Mel), audio_length, pre_cache, cache_last_channel, cache_last_time, cache_last_channel_len
        guard let mel = prepOutput.featureValue(for: "mel")?.multiArrayValue,
              let melLen = prepOutput.featureValue(for: "mel_length")?.multiArrayValue else {
            let keys = prepOutput.featureNames.joined(separator: ", ")
            throw ASRError.processingFailed("Missing mel output. Available: \(keys)")
        }
        
        if debugFeatures {
            let count = mel.count
            mel.withUnsafeBufferPointer(ofType: Float.self) { ptr in
                if let base = ptr.baseAddress {
                    debugFeatureBuffer.append(contentsOf: UnsafeBufferPointer(start: base, count: count))
                }
            }
        }
        
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: mel),
            "audio_length": MLFeatureValue(multiArray: melLen),
            "pre_cache": MLFeatureValue(multiArray: preCache),
            "cache_last_channel": MLFeatureValue(multiArray: cacheLastChannel),
            "cache_last_time": MLFeatureValue(multiArray: cacheLastTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLastChannelLen)
        ])
        
        let encoderOutput = try await streamingEncoder.prediction(from: encoderInput)
        
        // C. Update States (Loopback)
        if let newPreCache = encoderOutput.featureValue(for: "new_pre_cache")?.multiArrayValue {
            self.preCache = newPreCache
        }
        if let newChannel = encoderOutput.featureValue(for: "new_cache_last_channel")?.multiArrayValue {
            self.cacheLastChannel = newChannel
        }
        if let newTime = encoderOutput.featureValue(for: "new_cache_last_time")?.multiArrayValue {
            self.cacheLastTime = newTime
        }
        if let newChannelLen = encoderOutput.featureValue(for: "new_cache_last_channel_len")?.multiArrayValue {
            self.cacheLastChannelLen = newChannelLen
        }
        
        // D. Decode
        guard let encoded = encoderOutput.featureValue(for: "encoded_output")?.multiArrayValue,
              let encodedLen = encoderOutput.featureValue(for: "encoded_length")?.multiArrayValue else {
             throw ASRError.processingFailed("Missing encoder output")
        }
        
        // Decode tokens
        // Note: encodedLen is [B], we need Int
        // let len = encodedLen[0].intValue // Not used by RnntDecoder
        
        // Calculate time offset (for debug logs)
        // 160ms chunk -> 16 frames -> subsampling 4 -> 4 frames?
        // Actually Parakeet subsampling is 8? (4 from pre-encode, 2 from conformer?)
        // Let's just use processedSteps * 4 for now.
        let timeOffset = processedSteps * 4
        
        let tokenIds = try rnntDecoder.decode(encoderOutput: encoded, timeOffset: timeOffset)
        let decodedText = tokenizer.decode(ids: tokenIds)
        
        return decodedText
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
}

extension MLMultiArray {
    func reset(to value: NSNumber) {
        let count = self.count
        let ptr = self.dataPointer.bindMemory(to: Float.self, capacity: count)
        // Assuming Float32 for simplicity, but should check dataType
        if self.dataType == .float32 {
            ptr.assign(repeating: value.floatValue, count: count)
        } else if self.dataType == .int32 {
            let intPtr = self.dataPointer.bindMemory(to: Int32.self, capacity: count)
            intPtr.assign(repeating: value.int32Value, count: count)
        }
    }
}
