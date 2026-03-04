import AVFoundation
@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Stress tests to reproduce VoiceInk issues #321 (stalling) and #425 (degradation)
final class StressTests: XCTestCase {

    /// Test repeated transcriptions to check for memory leaks and degradation
    /// Reproduces VoiceInk #425 - progressive quality degradation
    func testRepeatedTranscriptions() async throws {
        let manager = AsrManager()

        // Download/load models
        print("📥 Downloading and loading models...")
        let models = try await AsrModels.downloadAndLoad()
        try await manager.initialize(models: models)
        print("✅ Models ready")

        // Download real audio from LibriSpeech
        print("📥 Downloading test audio...")
        let testAudio = try await downloadTestAudio()
        print("✅ Audio ready: \(testAudio.count) samples (\(String(format: "%.1f", Double(testAudio.count) / 16000))s)")

        let iterations = 100
        print("\n🧪 Starting stress test: \(iterations) consecutive transcriptions")
        print("=" * 60)

        var times: [TimeInterval] = []
        var memoryReadings: [UInt64] = []

        for i in 1...iterations {
            let startTime = Date()
            let startMemory = getMemoryUsage()

            do {
                let result = try await manager.transcribe(testAudio, source: .microphone)
                let elapsed = Date().timeIntervalSince(startTime)
                let endMemory = getMemoryUsage()

                times.append(elapsed)
                memoryReadings.append(endMemory)

                let memoryDelta = Int64(endMemory) - Int64(startMemory)
                let memoryDeltaStr = memoryDelta >= 0 ? "+\(memoryDelta / 1024)KB" : "\(memoryDelta / 1024)KB"

                print(
                    "[\(String(format: "%3d", i))/\(iterations)] Time: \(String(format: "%.3f", elapsed))s | Memory: \(endMemory / 1024 / 1024)MB (\(memoryDeltaStr)) | Text: \"\(result.text.prefix(30))...\""
                )
            } catch {
                print("[\(String(format: "%3d", i))/\(iterations)] ❌ ERROR: \(error)")
            }
        }

        print("=" * 60)

        // Analyze results
        let avgTime = times.reduce(0, +) / Double(times.count)
        let minTime = times.min() ?? 0
        let maxTime = times.max() ?? 0

        let firstFiveAvg = times.prefix(5).reduce(0, +) / 5
        let lastFiveAvg = times.suffix(5).reduce(0, +) / 5
        let degradation = ((lastFiveAvg - firstFiveAvg) / firstFiveAvg) * 100

        let startMemoryMB = memoryReadings.first.map { $0 / 1024 / 1024 } ?? 0
        let endMemoryMB = memoryReadings.last.map { $0 / 1024 / 1024 } ?? 0
        let memoryGrowth = Int64(endMemoryMB) - Int64(startMemoryMB)

        print("\n📊 RESULTS:")
        print("  Avg time: \(String(format: "%.3f", avgTime))s")
        print("  Min time: \(String(format: "%.3f", minTime))s")
        print("  Max time: \(String(format: "%.3f", maxTime))s")
        print("  First 5 avg: \(String(format: "%.3f", firstFiveAvg))s")
        print("  Last 5 avg: \(String(format: "%.3f", lastFiveAvg))s")
        print("  Time degradation: \(String(format: "%.1f", degradation))%")
        print("  Memory start: \(startMemoryMB)MB")
        print("  Memory end: \(endMemoryMB)MB")
        print("  Memory growth: \(memoryGrowth)MB")

        // Assertions
        if degradation > 50 {
            print("\n⚠️  WARNING: Significant time degradation detected (>50%)")
        }
        if memoryGrowth > 100 {
            print("⚠️  WARNING: Significant memory growth detected (>100MB)")
        }

        // Test passes if no crashes - we're looking for patterns, not hard failures
        XCTAssertTrue(times.count == iterations, "All \(iterations) transcriptions should complete")
    }

    /// Quick test to check single transcription works
    func testSingleTranscription() async throws {
        let manager = AsrManager()

        print("📥 Loading models...")
        let models = try await AsrModels.downloadAndLoad()
        try await manager.initialize(models: models)
        print("✅ Models loaded")

        // 2 seconds of test audio
        let sampleRate = 16000
        var testAudio = [Float](repeating: 0.0, count: sampleRate * 2)
        for i in 0..<testAudio.count {
            testAudio[i] = 0.05 * sin(2 * .pi * 440 * Float(i) / Float(sampleRate))
        }

        print("🎤 Transcribing...")
        let start = Date()
        let result = try await manager.transcribe(testAudio, source: .microphone)
        let elapsed = Date().timeIntervalSince(start)

        print("✅ Transcription completed in \(String(format: "%.2f", elapsed))s")
        print("   Text: \"\(result.text)\"")

        XCTAssertTrue(elapsed < 30, "Transcription should complete within 30 seconds")
    }

    // MARK: - Helpers

    private func getMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }

    /// Generate realistic speech-like audio for testing
    private func downloadTestAudio() async throws -> [Float] {
        // Generate 5 seconds of speech-like audio with varying frequencies
        // This simulates speech patterns better than a pure tone
        let sampleRate = 16000
        let duration = 5.0
        let sampleCount = Int(Double(sampleRate) * duration)

        var samples = [Float](repeating: 0.0, count: sampleCount)

        // Mix multiple frequencies to simulate speech harmonics
        let frequencies: [(freq: Float, amp: Float)] = [
            (150, 0.3),  // Fundamental (male voice range)
            (300, 0.2),  // First harmonic
            (450, 0.1),  // Second harmonic
            (600, 0.05),  // Third harmonic
            (1200, 0.03),  // Formant region
            (2400, 0.02),  // Higher formant
        ]

        for i in 0..<sampleCount {
            let t = Float(i) / Float(sampleRate)
            var sample: Float = 0.0

            // Add frequency components
            for (freq, amp) in frequencies {
                sample += amp * sin(2 * .pi * freq * t)
            }

            // Add amplitude modulation to simulate syllables (3-4 Hz)
            let syllableRate: Float = 3.5
            let envelope = 0.5 + 0.5 * sin(2 * .pi * syllableRate * t)
            sample *= envelope

            // Add some noise for realism
            sample += Float.random(in: -0.01...0.01)

            samples[i] = sample
        }

        return samples
    }
}

// Helper for string repetition
extension String {
    static func * (lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
