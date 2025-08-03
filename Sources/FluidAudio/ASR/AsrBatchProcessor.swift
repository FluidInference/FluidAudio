import CoreML
import Foundation
import OSLog

/// Result for a single audio file in batch processing
public struct BatchASRResult: Sendable {
    public let fileURL: URL
    public let result: Result<ASRResult, Error>
    public let processingTime: TimeInterval
}

/// Configuration for batch processing
public struct BatchProcessingConfig: Sendable {
    public let maxConcurrency: Int
    public let enableGPUSharing: Bool
    public let performanceProfile: AsrModels.PerformanceProfile

    public init(
        maxConcurrency: Int = ProcessInfo.processInfo.processorCount,
        enableGPUSharing: Bool = true,
        performanceProfile: AsrModels.PerformanceProfile = .balanced
    ) {
        self.maxConcurrency = maxConcurrency
        self.enableGPUSharing = enableGPUSharing
        self.performanceProfile = performanceProfile
    }
}

/// Batch processor for multiple audio files
@available(macOS 13.0, iOS 16.0, *)
public actor AsrBatchProcessor {
    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "BatchProcessor")
    private let models: AsrModels
    private let config: BatchProcessingConfig

    /// Shared array cache for batch processing
    private let arrayCache = MLArrayCache(maxCacheSize: 200)

    public init(models: AsrModels, config: BatchProcessingConfig = BatchProcessingConfig()) {
        self.models = models
        self.config = config
    }

    /// Process multiple audio files concurrently
    public func processBatch(audioFiles: [URL]) async -> [BatchASRResult] {
        logger.info("Starting batch processing of \(audioFiles.count) files")

        // Pre-warm cache with common shapes
        await prewarmCache()

        // Process files with controlled concurrency
        return await withTaskGroup(of: BatchASRResult.self) { group in
            var results: [BatchASRResult] = []

            // Limit concurrent tasks
            let semaphore = AsyncSemaphore(value: config.maxConcurrency)

            for audioFile in audioFiles {
                group.addTask {
                    await semaphore.wait()
                    defer { Task { await semaphore.signal() } }

                    return await self.processFile(audioFile)
                }
            }

            // Collect results
            for await result in group {
                results.append(result)
            }

            return results
        }
    }

    /// Process a single audio file
    private func processFile(_ fileURL: URL) async -> BatchASRResult {
        let startTime = Date()

        do {
            // Load audio samples
            let audioSamples = try await loadAudioSamples(from: fileURL)

            // Create ASR manager with optimized config
            let asrConfig = ASRConfig(
                sampleRate: 16000,
                enableDebug: false,
                tdtConfig: TdtConfig.default
            )

            let manager = AsrManager(config: asrConfig)

            // Initialize with our models
            try await manager.initialize(models: models)

            // Process audio
            let result = try await manager.transcribe(audioSamples)

            let processingTime = Date().timeIntervalSince(startTime)
            logger.info("Processed \(fileURL.lastPathComponent) in \(String(format: "%.2f", processingTime))s")

            return BatchASRResult(
                fileURL: fileURL,
                result: .success(result),
                processingTime: processingTime
            )
        } catch {
            let processingTime = Date().timeIntervalSince(startTime)
            logger.error("Failed to process \(fileURL.lastPathComponent): \(error)")

            return BatchASRResult(
                fileURL: fileURL,
                result: .failure(error),
                processingTime: processingTime
            )
        }
    }

    /// Load audio samples from file
    private func loadAudioSamples(from url: URL) async throws -> [Float] {
        // This is a simplified implementation
        // In production, use proper audio loading with AVFoundation
        throw ASRError.processingFailed("Audio loading not implemented - use AudioUtils")
    }

    /// Pre-warm the array cache with common shapes
    private func prewarmCache() async {
        let commonShapes: [(shape: [NSNumber], dataType: MLMultiArrayDataType)] = [
            // Mel-spectrogram shapes
            ([1, 160000], .float32),
            ([1], .int32),

            // Encoder shapes
            ([1, 500, 80], .float32),
            ([1], .int32),

            // Decoder shapes
            ([1, 1], .int32),
            ([2, 1, 640], .float32),

            // Joint network shapes
            ([1, 1, 1024], .float32),
            ([1, 1, 1280], .float32),
        ]

        await arrayCache.prewarm(shapes: commonShapes)
    }
}

/// Simple async semaphore for concurrency control
@available(macOS 13.0, iOS 16.0, *)
actor AsyncSemaphore {
    private var value: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(value: Int) {
        self.value = value
    }

    func wait() async {
        if value > 0 {
            value -= 1
            return
        }

        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func signal() {
        if let waiter = waiters.first {
            waiters.removeFirst()
            waiter.resume()
        } else {
            value += 1
        }
    }
}

/// Extension to AsrManager for batch processing optimizations
@available(macOS 13.0, iOS 16.0, *)
extension AsrManager {
    /// Process multiple audio chunks in parallel
    public func transcribeBatch(_ audioChunks: [[Float]]) async throws -> [ASRResult] {
        logger.info("Processing batch of \(audioChunks.count) audio chunks")

        return try await withThrowingTaskGroup(of: (Int, ASRResult).self) { group in
            for (index, chunk) in audioChunks.enumerated() {
                group.addTask {
                    let result = try await self.transcribe(chunk)
                    return (index, result)
                }
            }

            // Collect results in order
            var results = [ASRResult?](repeating: nil, count: audioChunks.count)
            for try await (index, result) in group {
                results[index] = result
            }

            return results.compactMap { $0 }
        }
    }
}
