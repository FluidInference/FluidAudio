import CoreML
import FluidAudio
import Foundation

/// Example showing how to bundle ASR models with your app
/// and initialize FluidAudio with predownloaded models.
@available(macOS 13.0, iOS 16.0, *)
class BundledASRExample {
    
    // MARK: - Bundle Models with Your App
    
    /// Step 1: Add model files to your app bundle
    ///
    /// Download the models first using FluidAudio CLI or programmatically:
    /// ```swift
    /// let models = try await AsrModels.downloadAndLoad()
    /// ```
    /// Then copy these files to your app bundle:
    /// - Melspectogram.mlmodelc
    /// - ParakeetEncoder_v2.mlmodelc  
    /// - ParakeetDecoder.mlmodelc
    /// - RNNTJoint.mlmodelc
    /// - parakeet_vocab.json
    
    static func getBundledModelPaths() -> (
        melspectrogram: URL?,
        encoder: URL?, 
        decoder: URL?,
        joint: URL?,
        vocabulary: URL?
    ) {
        let bundle = Bundle.main
        
        let melspectrogramURL = bundle.url(
            forResource: "Melspectogram", 
            withExtension: "mlmodelc"
        )
        
        let encoderURL = bundle.url(
            forResource: "ParakeetEncoder_v2", 
            withExtension: "mlmodelc"
        )
        
        let decoderURL = bundle.url(
            forResource: "ParakeetDecoder", 
            withExtension: "mlmodelc"
        )
        
        let jointURL = bundle.url(
            forResource: "RNNTJoint", 
            withExtension: "mlmodelc"
        )
        
        let vocabularyURL = bundle.url(
            forResource: "parakeet_vocab", 
            withExtension: "json"
        )
        
        return (melspectrogramURL, encoderURL, decoderURL, jointURL, vocabularyURL)
    }
    
    // MARK: - Load Vocabulary from Bundle
    
    static func loadBundledVocabulary() throws -> [Int: String] {
        let (_, _, _, _, vocabularyURL) = getBundledModelPaths()
        
        guard let vocabularyURL = vocabularyURL else {
            throw ExampleError.modelNotFound("parakeet_vocab.json not found in app bundle")
        }
        
        let data = try Data(contentsOf: vocabularyURL)
        let jsonDict = try JSONSerialization.jsonObject(with: data) as? [String: String] ?? [:]
        
        var vocabulary: [Int: String] = [:]
        for (key, value) in jsonDict {
            if let tokenId = Int(key) {
                vocabulary[tokenId] = value
            }
        }
        
        return vocabulary
    }
    
    // MARK: - Initialize with Bundled Models
    
    static func initializeWithBundledModels() async throws -> AsrManager {
        // Get paths to bundled models
        let (melspectrogramURL, encoderURL, decoderURL, jointURL, _) = getBundledModelPaths()
        
        guard let melspectrogramURL = melspectrogramURL,
              let encoderURL = encoderURL,
              let decoderURL = decoderURL,
              let jointURL = jointURL else {
            throw ExampleError.modelNotFound("ASR models not found in app bundle")
        }
        
        // Load vocabulary
        let vocabulary = try loadBundledVocabulary()
        
        // Create default configuration
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine  // Optimal for all ASR models
        configuration.allowLowPrecisionAccumulationOnGPU = true  // Enable FP16
        
        // Load individual models
        let melspectrogramModel = try MLModel(contentsOf: melspectrogramURL, configuration: configuration)
        let encoderModel = try MLModel(contentsOf: encoderURL, configuration: configuration)
        let decoderModel = try MLModel(contentsOf: decoderURL, configuration: configuration)
        let jointModel = try MLModel(contentsOf: jointURL, configuration: configuration)
        
        // Create AsrModels instance
        let models = AsrModels(
            melspectrogram: melspectrogramModel,
            encoder: encoderModel,
            decoder: decoderModel,
            joint: jointModel,
            configuration: configuration,
            vocabulary: vocabulary
        )
        
        // Initialize ASR manager
        let config = ASRConfig.default
        let manager = AsrManager(config: config)
        try await manager.initialize(models: models)
        
        return manager
    }
    
    // MARK: - Performance-Optimized Configuration
    
    static func initializeWithPerformanceOptimization() async throws -> AsrManager {
        let (melspectrogramURL, encoderURL, decoderURL, jointURL, _) = getBundledModelPaths()
        
        guard let melspectrogramURL = melspectrogramURL,
              let encoderURL = encoderURL,
              let decoderURL = decoderURL,
              let jointURL = jointURL else {
            throw ExampleError.modelNotFound("ASR models not found in app bundle")
        }
        
        let vocabulary = try loadBundledVocabulary()
        
        // Use different configurations for different model types (from CLAUDE.md recommendations)
        let melConfiguration = AsrModels.optimizedConfiguration(for: .melSpectrogram)
        let encoderConfiguration = AsrModels.optimizedConfiguration(for: .encoder)
        let decoderConfiguration = AsrModels.optimizedConfiguration(for: .decoder)
        let jointConfiguration = AsrModels.optimizedConfiguration(for: .joint)
        
        // Load models with optimized configurations
        let melspectrogramModel = try MLModel(contentsOf: melspectrogramURL, configuration: melConfiguration)
        let encoderModel = try MLModel(contentsOf: encoderURL, configuration: encoderConfiguration)
        let decoderModel = try MLModel(contentsOf: decoderURL, configuration: decoderConfiguration)
        let jointModel = try MLModel(contentsOf: jointURL, configuration: jointConfiguration)
        
        // Use default configuration for the AsrModels instance
        let models = AsrModels(
            melspectrogram: melspectrogramModel,
            encoder: encoderModel,
            decoder: decoderModel,
            joint: jointModel,
            configuration: AsrModels.defaultConfiguration(),
            vocabulary: vocabulary
        )
        
        // Create performance-optimized ASR config
        let config = ASRConfig(
            enableDebug: false,
            tdtConfig: TdtConfig(
                durations: [0, 1, 2, 3, 4],  // Token duration optimization
                includeTokenDuration: true
            )
        )
        
        let manager = AsrManager(config: config)
        try await manager.initialize(models: models)
        
        return manager
    }
    
    // MARK: - iOS Background Configuration
    
    static func initializeForBackgroundProcessing() async throws -> AsrManager {
        let (melspectrogramURL, encoderURL, decoderURL, jointURL, _) = getBundledModelPaths()
        
        guard let melspectrogramURL = melspectrogramURL,
              let encoderURL = encoderURL,
              let decoderURL = decoderURL,
              let jointURL = jointURL else {
            throw ExampleError.modelNotFound("ASR models not found in app bundle")
        }
        
        let vocabulary = try loadBundledVocabulary()
        
        // Use iOS background-compatible configuration (no GPU)
        let configuration = AsrModels.iOSBackgroundConfiguration()
        
        let melspectrogramModel = try MLModel(contentsOf: melspectrogramURL, configuration: configuration)
        let encoderModel = try MLModel(contentsOf: encoderURL, configuration: configuration)
        let decoderModel = try MLModel(contentsOf: decoderURL, configuration: configuration)
        let jointModel = try MLModel(contentsOf: jointURL, configuration: configuration)
        
        let models = AsrModels(
            melspectrogram: melspectrogramModel,
            encoder: encoderModel,
            decoder: decoderModel,
            joint: jointModel,
            configuration: configuration,
            vocabulary: vocabulary
        )
        
        let config = ASRConfig.default
        let manager = AsrManager(config: config)
        try await manager.initialize(models: models)
        
        return manager
    }
    
    // MARK: - Usage Example
    
    static func performTranscription() async throws {
        // Initialize with bundled models
        let asr = try await initializeWithBundledModels()
        
        // Example audio (replace with your actual audio data)
        let sampleRate = 16000
        let duration = 3.0  // 3 seconds  
        let sampleCount = Int(duration * Double(sampleRate))
        
        // Generate example audio (replace with actual audio loading)
        let audioSamples = (0..<sampleCount).map { i in
            sin(Float(i) * 0.01) * 0.5
        }
        
        // Perform transcription
        let result = try await asr.transcribe(audioSamples)
        
        // Process results
        print("Transcription completed!")
        print("Text: '\(result.text)'")
        print("Confidence: \(result.confidence)")
        
        // Token-level timing information (if available)
        if let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty {
            print("\nToken timings:")
            for timing in tokenTimings {
                print("  '\(timing.token)': \(timing.startTime)s - \(timing.endTime)s (confidence: \(timing.confidence))")
            }
        }
        
        // Performance metrics
        print("\nPerformance:")
        print("Processing time: \(result.processingTime)s")
        print("Real-time factor: \(result.processingTime / duration)")
        
        // Cleanup when done
        asr.cleanup()
    }
    
    // MARK: - Streaming Transcription Example
    
    static func performStreamingTranscription() async throws {
        let asr = try await initializeWithBundledModels()
        
        // Example: Process audio in chunks for streaming
        let chunkSize = 8000  // 0.5 second chunks at 16kHz
        let totalSamples = 48000  // 3 seconds total
        
        var fullTranscription = ""
        
        for chunkStart in stride(from: 0, to: totalSamples, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, totalSamples)
            let chunkSamples = (chunkStart..<chunkEnd).map { i in
                sin(Float(i) * 0.01) * 0.5
            }
            
            // Transcribe chunk
            let result = try await asr.transcribe(chunkSamples)
            
            if !result.text.isEmpty {
                print("Chunk \(chunkStart/chunkSize): '\(result.text)'")
                fullTranscription += result.text + " "
            }
        }
        
        print("\nFull transcription: '\(fullTranscription.trimmingCharacters(in: .whitespaces))'")
        asr.cleanup()
    }
    
    // MARK: - Alternative: Copy Models to Documents Directory
    
    static func copyModelsToDocuments() throws -> URL {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsURL = documentsURL.appendingPathComponent("ASRModels", isDirectory: true)
        
        // Create models directory
        try FileManager.default.createDirectory(at: modelsURL, withIntermediateDirectories: true)
        
        let (melURL, encoderURL, decoderURL, jointURL, vocabURL) = getBundledModelPaths()
        
        guard let melURL = melURL, let encoderURL = encoderURL,
              let decoderURL = decoderURL, let jointURL = jointURL,
              let vocabURL = vocabURL else {
            throw ExampleError.modelNotFound("Models not found in bundle")
        }
        
        // Copy models if they don't exist
        let modelFiles = [
            ("Melspectogram.mlmodelc", melURL),
            ("ParakeetEncoder_v2.mlmodelc", encoderURL),
            ("ParakeetDecoder.mlmodelc", decoderURL), 
            ("RNNTJoint.mlmodelc", jointURL),
            ("parakeet_vocab.json", vocabURL)
        ]
        
        for (fileName, sourceURL) in modelFiles {
            let destinationURL = modelsURL.appendingPathComponent(fileName)
            if !FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            }
        }
        
        return modelsURL
    }
}

// MARK: - Usage Instructions

/*
 Usage Instructions:
 
 1. Download Models:
    - Use FluidAudio CLI: `swift run fluidaudio download-models`
    - Or programmatically: `try await AsrModels.downloadAndLoad()`
 
 2. Add to App Bundle:
    - Copy Melspectogram.mlmodelc to your Xcode project
    - Copy ParakeetEncoder_v2.mlmodelc to your Xcode project
    - Copy ParakeetDecoder.mlmodelc to your Xcode project  
    - Copy RNNTJoint.mlmodelc to your Xcode project
    - Copy parakeet_vocab.json to your Xcode project
    - Make sure "Add to target" is checked for your app target
 
 3. Use in Your App:
    ```swift
    do {
        try await BundledASRExample.performTranscription()
    } catch {
        print("Transcription failed: \(error)")
    }
    ```
 
 4. Different Use Cases:
    - Real-time transcription: `performTranscription()`
    - Streaming processing: `performStreamingTranscription()`  
    - Background processing: `initializeForBackgroundProcessing()`
    - Performance optimization: `initializeWithPerformanceOptimization()`
 
 5. Alternative Approaches:
    - Copy models to Documents directory: `copyModelsToDocuments()`
    - Download on first launch: `AsrModels.downloadAndLoad()`
    - Use different configurations for different model types
 
 Benefits of Bundling:
 - No network dependency
 - Faster app startup
 - Works offline
 - Consistent model versions
 - Better user experience
 - Immediate transcription capability
 
 Considerations:
 - Increases app size significantly (~500MB for ASR models)
 - Models are part of app updates
 - Less flexibility for model updates
 - Consider model size vs. user experience trade-offs
 
 Performance Notes:
 - Use .cpuAndNeuralEngine for optimal performance
 - Enable FP16 for ~2x speedup on compatible devices
 - Different models may benefit from different compute units
 - Consider battery usage for real-time applications
 */