import CoreML
import FluidAudio
import Foundation

/// Example showing how to bundle diarization models with your app
/// and initialize FluidAudio with predownloaded models.
@available(macOS 13.0, iOS 16.0, *)
class BundledDiarizationExample {
    
    // MARK: - Bundle Models with Your App
    
    /// Step 1: Add model files to your app bundle
    /// 
    /// Download the models first using FluidAudio CLI or programmatically:
    /// ```swift
    /// let models = try await DiarizerModels.downloadIfNeeded()
    /// ```
    /// Then copy these files to your app bundle:
    /// - pyannote_segmentation.mlmodelc
    /// - wespeaker_v2.mlmodelc
    
    static func getBundledModelPaths() -> (segmentation: URL?, embedding: URL?) {
        let bundle = Bundle.main
        
        // Look for models in the app bundle
        let segmentationModelURL = bundle.url(
            forResource: "pyannote_segmentation", 
            withExtension: "mlmodelc"
        )
        
        let embeddingModelURL = bundle.url(
            forResource: "wespeaker_v2", 
            withExtension: "mlmodelc"
        )
        
        return (segmentationModelURL, embeddingModelURL)
    }
    
    // MARK: - Initialize with Bundled Models
    
    static func initializeWithBundledModels() async throws -> DiarizerManager {
        // Get paths to bundled models
        let (segmentationURL, embeddingURL) = getBundledModelPaths()
        
        guard let segmentationURL = segmentationURL,
              let embeddingURL = embeddingURL else {
            throw ExampleError.modelNotFound("Models not found in app bundle")
        }
        
        // Load models from bundle
        let models = try await DiarizerModels.load(
            localSegmentationModel: segmentationURL,
            localEmbeddingModel: embeddingURL
        )
        
        // Initialize diarizer with bundled models
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minDurationOn: 1.0,
            minDurationOff: 0.5,
            debugMode: false
        )
        
        let manager = DiarizerManager(config: config)
        manager.initialize(models: models)
        
        return manager
    }
    
    // MARK: - Custom Configuration Example
    
    static func initializeWithCustomConfig() async throws -> DiarizerManager {
        let (segmentationURL, embeddingURL) = getBundledModelPaths()
        
        guard let segmentationURL = segmentationURL,
              let embeddingURL = embeddingURL else {
            throw ExampleError.modelNotFound("Models not found in app bundle")
        }
        
        // Create custom MLModelConfiguration for specific compute units
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine  // iOS background compatible
        configuration.allowLowPrecisionAccumulationOnGPU = true  // Enable FP16 optimization
        
        // Load models with custom configuration
        let models = try await DiarizerModels.load(
            localSegmentationModel: segmentationURL,
            localEmbeddingModel: embeddingURL,
            configuration: configuration
        )
        
        // Use optimal configuration from CLAUDE.md
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,  // Optimal: 17.7% DER
            minDurationOn: 1.0,
            minDurationOff: 0.5,
            minActivityThreshold: 10.0,
            debugMode: false
        )
        
        let manager = DiarizerManager(config: config)
        manager.initialize(models: models)
        
        return manager
    }
    
    // MARK: - Usage Example
    
    static func performDiarization() async throws {
        // Initialize with bundled models
        let diarizer = try await initializeWithBundledModels()
        
        // Example audio (replace with your actual audio data)
        let sampleRate = 16000
        let duration = 3.0  // 3 seconds
        let sampleCount = Int(duration * Double(sampleRate))
        
        // Generate example audio (replace with actual audio loading)
        let audioSamples = (0..<sampleCount).map { i in
            sin(Float(i) * 0.01) * 0.5
        }
        
        // Perform diarization
        let result = try diarizer.performCompleteDiarization(audioSamples, sampleRate: sampleRate)
        
        // Process results
        print("Diarization completed!")
        print("Found \(result.segments.count) speaker segments")
        print("Unique speakers: \(Set(result.segments.map { $0.speakerId }).count)")
        
        for segment in result.segments {
            print("Speaker \(segment.speakerId): \(segment.startTime)s - \(segment.endTime)s")
        }
        
        // Performance metrics
        print("\nPerformance:")
        print("Segmentation: \(result.timings.segmentationSeconds)s")
        print("Embedding: \(result.timings.embeddingExtractionSeconds)s")
        print("Clustering: \(result.timings.speakerClusteringSeconds)s")
        
        // Cleanup when done
        diarizer.cleanup()
    }
    
    // MARK: - Alternative: Copy Models to Documents Directory
    
    /// Alternative approach: Copy models from bundle to Documents directory on first launch
    static func copyModelsToDocuments() throws -> (segmentation: URL, embedding: URL) {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsURL = documentsURL.appendingPathComponent("Models", isDirectory: true)
        
        // Create models directory
        try FileManager.default.createDirectory(at: modelsURL, withIntermediateDirectories: true)
        
        let (bundledSegmentation, bundledEmbedding) = getBundledModelPaths()
        
        guard let bundledSegmentation = bundledSegmentation,
              let bundledEmbedding = bundledEmbedding else {
            throw ExampleError.modelNotFound("Models not found in bundle")
        }
        
        let destinationSegmentation = modelsURL.appendingPathComponent("pyannote_segmentation.mlmodelc")
        let destinationEmbedding = modelsURL.appendingPathComponent("wespeaker_v2.mlmodelc")
        
        // Copy models if they don't exist
        if !FileManager.default.fileExists(atPath: destinationSegmentation.path) {
            try FileManager.default.copyItem(at: bundledSegmentation, to: destinationSegmentation)
        }
        
        if !FileManager.default.fileExists(atPath: destinationEmbedding.path) {
            try FileManager.default.copyItem(at: bundledEmbedding, to: destinationEmbedding)
        }
        
        return (destinationSegmentation, destinationEmbedding)
    }
}

// MARK: - Error Types

enum ExampleError: LocalizedError {
    case modelNotFound(String)
    case audioLoadingFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .modelNotFound(let message):
            return "Model not found: \(message)"
        case .audioLoadingFailed(let message):
            return "Audio loading failed: \(message)"
        }
    }
}

// MARK: - Usage Instructions

/*
 Usage Instructions:
 
 1. Download Models:
    - Use FluidAudio CLI: `swift run fluidaudio download-models`
    - Or programmatically: `try await DiarizerModels.downloadIfNeeded()`
 
 2. Add to App Bundle:
    - Copy pyannote_segmentation.mlmodelc to your Xcode project
    - Copy wespeaker_v2.mlmodelc to your Xcode project  
    - Make sure "Add to target" is checked for your app target
 
 3. Use in Your App:
    ```swift
    do {
        try await BundledDiarizationExample.performDiarization()
    } catch {
        print("Diarization failed: \(error)")
    }
    ```
 
 4. Alternative Approaches:
    - Copy models to Documents directory: `copyModelsToDocuments()`
    - Download on first launch: `DiarizerModels.downloadIfNeeded()`
    - Use custom model configurations for specific compute units
 
 Benefits of Bundling:
 - No network dependency
 - Faster app startup
 - Works offline
 - Consistent model versions
 - Better user experience
 
 Considerations:
 - Increases app size (~100MB for diarization models)
 - Models are part of app updates
 - Less flexibility for model updates
 */