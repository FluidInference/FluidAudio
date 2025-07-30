import Foundation
import FluidAudio
import OSLog

/// Quick performance test for ASR optimizations
@available(macOS 13.0, *)
struct AsrPerformanceTest {
    static func runPerformanceTest() async throws {
        print("🚀 ASR Performance Test")
        print("=" * 50)
        
        // Create test audio (10 seconds of silence as a baseline)
        let sampleRate = 16000
        let duration = 10.0
        let sampleCount = Int(Double(sampleRate) * duration)
        let testAudio = Array(repeating: Float(0), count: sampleCount)
        
        // Initialize ASR with performance profile
        let config = ASRConfig(
            sampleRate: sampleRate,
            enableDebug: false,
            tdtConfig: .default
        )
        
        print("Loading models...")
        let models = try await AsrModels.loadWithAutoRecovery(
            configuration: AsrModels.PerformanceProfile.lowLatency.configuration
        )
        
        let manager = AsrManager(config: config)
        try await manager.initialize(models: models)
        
        // Create performance monitor
        let monitor = PerformanceMonitor()
        
        print("\nRunning performance tests...")
        print("-" * 50)
        
        // Warm-up run
        print("Warm-up run...")
        _ = try await manager.transcribe(testAudio)
        
        // Test runs with metrics
        var results: [ASRPerformanceMetrics] = []
        let testRuns = 5
        
        for i in 1...testRuns {
            print("Test run \(i)/\(testRuns)...")
            
            let (result, metrics) = try await manager.transcribeWithMetrics(
                testAudio,
                monitor: monitor
            )
            
            if let metrics = metrics {
                results.append(metrics)
                print("  RTFx: \(String(format: "%.1f", metrics.rtfx))x real-time")
                print("  Processing time: \(String(format: "%.3f", metrics.totalProcessingTime))s")
            }
        }
        
        // Print aggregated results
        if let aggregated = await monitor.getAggregatedMetrics() {
            print("\n" + "=" * 50)
            print(aggregated.summary)
        }
        
        // Calculate improvement metrics
        let avgRTFx = results.map { $0.rtfx }.reduce(0, +) / Float(results.count)
        print("\n📊 Performance Summary:")
        print("  Average RTFx: \(String(format: "%.1f", avgRTFx))x real-time")
        print("  Expected baseline: ~40-70x")
        print("  Improvement: \(String(format: "%.1f", avgRTFx / 55.0))x") // Using 55x as baseline
        
        // Test batch processing
        print("\n🔄 Testing batch processing...")
        let batchAudio = Array(repeating: testAudio, count: 4)
        let batchStart = Date()
        let batchResults = try await manager.transcribeBatch(batchAudio)
        let batchTime = Date().timeIntervalSince(batchStart)
        let batchRTFx = Float(duration * 4) / Float(batchTime)
        
        print("  Batch size: 4")
        print("  Total time: \(String(format: "%.3f", batchTime))s")
        print("  Batch RTFx: \(String(format: "%.1f", batchRTFx))x real-time")
        print("  Speedup vs sequential: \(String(format: "%.1f", batchRTFx / avgRTFx))x")
        
        print("\n✅ Performance test completed!")
    }
}

// Helper extension
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}