import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
extension DiarizerModels {

    /// Check if INT8 models should be loaded based on environment variable
    static var shouldUseINT8Models: Bool {
        ProcessInfo.processInfo.environment["USE_INT8_MODELS"] != nil
    }

    /// Get the models directory path
    static func getModelsDirectory() throws -> URL {
        guard
            let applicationSupport = FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            ).first
        else {
            throw DiarizerError.modelDownloadFailed
        }

        return
            applicationSupport
            .appendingPathComponent("FluidAudio")
            .appendingPathComponent("Models")
            .appendingPathComponent("speaker-diarization-coreml")
    }

    /// Compare INT8 vs Float32 performance
    static func benchmarkINT8vsFloat32() async throws {
        let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "INT8Benchmark")

        logger.info("🏃 Benchmarking INT8 vs Float32 models...")

        // Load both model variants
        let modelsDir = try getModelsDirectory()
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        // Test inputs
        let waveform = try MLMultiArray(shape: [3, 160000], dataType: .float32)
        let mask = try MLMultiArray(shape: [3, 589], dataType: .float32)

        // Fill with dummy data
        for i in 0..<waveform.count {
            waveform[i] = NSNumber(value: Float.random(in: -1...1))
        }
        for i in 0..<mask.count {
            mask[i] = NSNumber(value: 1.0)
        }

        let inputs = ["waveform": waveform, "mask": mask]

        // Benchmark Float32 model
        if let float32Path = modelsDir.appendingPathComponent("wespeaker_float32.mlmodelc") as URL?,
            FileManager.default.fileExists(atPath: float32Path.path)
        {

            let float32Model = try MLModel(contentsOf: float32Path, configuration: config)

            logger.info("📊 Testing Float32 model...")
            let float32Time = try await benchmarkModel(float32Model, inputs: inputs)
            logger.info("   Float32 time: \(float32Time)ms")
        }

        // Benchmark INT8 model
        config.allowLowPrecisionAccumulationOnGPU = true

        if let int8Path = URL(fileURLWithPath: "wespeaker.mlmodelc") as URL?,
            FileManager.default.fileExists(atPath: int8Path.path)
        {

            let int8Model = try MLModel(contentsOf: int8Path, configuration: config)

            logger.info("📊 Testing INT8 model...")
            let int8Time = try await benchmarkModel(int8Model, inputs: inputs)
            logger.info("   INT8 time: \(int8Time)ms")

            // Compare
            if let float32Time = try? await benchmarkModel(int8Model, inputs: inputs) {
                let speedup = float32Time / int8Time
                logger.info("🚀 INT8 speedup: \(String(format: "%.2fx", speedup))")
            }
        }
    }

    private static func benchmarkModel(_ model: MLModel, inputs: [String: Any]) async throws -> Double {
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)

        // Warmup
        _ = try model.prediction(from: provider)

        // Benchmark
        let iterations = 10
        let start = CFAbsoluteTimeGetCurrent()

        for _ in 0..<iterations {
            _ = try model.prediction(from: provider)
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
        return elapsed
    }
}
