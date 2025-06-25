import Foundation
import CoreML
import OSLog
#if canImport(UIKit)
import UIKit
#endif
import AVFoundation

/// iOS-optimized CoreML Speaker Diarization Manager
/// Designed specifically for iOS deployment with mobile-specific optimizations
@available(iOS 16.0, *)
@MainActor
public final class iOSCoreMLDiarizerManager: ObservableObject {

    // MARK: - Published Properties
    @Published public var isInitialized: Bool = false
    @Published public var isProcessing: Bool = false
    @Published public var downloadProgress: Float = 0.0
    @Published public var modelStatus: ModelStatus = .notLoaded
    @Published public var lastError: DiarizerError?

    // MARK: - Types
    public enum ModelStatus {
        case notLoaded
        case downloading
        case loading
        case ready
        case error(DiarizerError)
    }

    public struct ProcessingResult {
        public let segments: [SpeakerSegment]
        public let processingTime: TimeInterval
        public let confidenceScore: Float
    }

    // MARK: - Private Properties
    private let logger = Logger(subsystem: "com.fluidinfluenceiOS", category: "iOSCoreML")
    private var segmentationModel: MLModel?
    private var embeddingModel: MLModel?
    #if canImport(UIKit)
    private var backgroundTaskID: UIBackgroundTaskIdentifier = .invalid
    #endif

    // iOS-specific configuration
    private let iosConfig: Configuration
    private let maxChunkSize: Int
    private let memoryThreshold: Int64 = 100_000_000 // 100MB
    private let lowMemoryChunkSize: Int

    // MARK: - Configuration
    public struct Configuration: Sendable {
        public let modelCacheDirectory: URL?
        public let enableBackgroundProcessing: Bool
        public let maxProcessingTimeInBackground: TimeInterval
        public let enableMemoryWarning: Bool
        public let optimizeForBattery: Bool

        public init(
            modelCacheDirectory: URL? = nil,
            enableBackgroundProcessing: Bool = true,
            maxProcessingTimeInBackground: TimeInterval = 30.0,
            enableMemoryWarning: Bool = true,
            optimizeForBattery: Bool = true
        ) {
            self.modelCacheDirectory = modelCacheDirectory
            self.enableBackgroundProcessing = enableBackgroundProcessing
            self.maxProcessingTimeInBackground = maxProcessingTimeInBackground
            self.enableMemoryWarning = enableMemoryWarning
            self.optimizeForBattery = optimizeForBattery
        }

        @MainActor
        public static let `default` = Configuration()
    }

    // MARK: - Initialization
    public init(configuration: Configuration = .default) {
        self.iosConfig = configuration

        // Adapt chunk sizes based on device capabilities
        let processorCount = ProcessInfo.processInfo.processorCount

        if processorCount >= 6 && !ProcessInfo.processInfo.isLowPowerModeEnabled {
            // High-end device
            self.maxChunkSize = 16000 * 10 // 10 seconds
            self.lowMemoryChunkSize = 16000 * 5 // 5 seconds
        } else {
            // Lower-end device or low power mode
            self.maxChunkSize = 16000 * 5 // 5 seconds
            self.lowMemoryChunkSize = 16000 * 3 // 3 seconds
        }

        setupMemoryPressureNotifications()
        setupAppStateNotifications()
    }

    deinit {
        #if canImport(UIKit)
        if backgroundTaskID != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTaskID)
            backgroundTaskID = .invalid
        }
        #endif
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Public API

    /// Initialize the diarization system
    public func initialize() async throws {
        guard !isInitialized else { return }

        logger.info("Initializing iOS CoreML diarizer")

        await MainActor.run {
            modelStatus = .loading
            isProcessing = true
        }

        do {
            try await loadModels()
            await MainActor.run {
                isInitialized = true
                modelStatus = .ready
                isProcessing = false
                logger.info("iOS CoreML diarizer initialized successfully")
            }
        } catch {
            logger.error("Initialization failed: \(error.localizedDescription)")
            await cleanup()
            endBackgroundTask()
            throw error
        }
    }

    /// Process audio for speaker diarization
    public func processAudio(_ audioData: Data) async throws -> ProcessingResult {
        guard isInitialized else {
            throw DiarizerError.notInitialized
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        await MainActor.run {
            isProcessing = true
        }

        defer {
            Task { @MainActor in
                isProcessing = false
            }
        }

        do {
            // Start background task for processing
            if iosConfig.enableBackgroundProcessing {
                beginBackgroundTask()
            }

            // Convert audio data to samples
            let samples = try convertAudioDataToSamples(audioData)

            // Process with device-appropriate chunk size
            let chunkSize = getCurrentChunkSize()
            let segments = try await processAudioSamples(samples, chunkSize: chunkSize)

            let processingTime = CFAbsoluteTimeGetCurrent() - startTime
            let confidence = calculateOverallConfidence(segments)

            logger.info("Audio processed in \(processingTime)s with confidence \(confidence)")

            return ProcessingResult(
                segments: segments,
                processingTime: processingTime,
                confidenceScore: confidence
            )

        } catch {
            logger.error("Audio processing failed: \(error)")
            await MainActor.run {
                lastError = error as? DiarizerError ?? .processingFailed("Audio processing failed")
            }
            endBackgroundTask()
            throw error
        }
    }

    /// Process audio file URL
    public func processAudioFile(_ url: URL) async throws -> ProcessingResult {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DiarizerError.invalidAudioData
        }

        let audioData = try Data(contentsOf: url)
        return try await processAudio(audioData)
    }

    /// Download models (with progress tracking)
    public func downloadModels() async throws {
        logger.info("Starting model download")

        await MainActor.run {
            modelStatus = .downloading
            downloadProgress = 0.0
        }

        let modelsDirectory = getModelsDirectory()

        try await downloadModelWithProgress(
            repoPath: "bweng/speaker-diarization-coreml",
            modelName: "pyannote_segmentation.mlmodelc",
            outputPath: modelsDirectory.appendingPathComponent("pyannote_segmentation.mlmodelc"),
            progressCallback: { progress in
                Task { @MainActor in
                    self.downloadProgress = progress * 0.5 // First half
                }
            }
        )

        try await downloadModelWithProgress(
            repoPath: "bweng/speaker-diarization-coreml",
            modelName: "wespeaker.mlmodelc",
            outputPath: modelsDirectory.appendingPathComponent("wespeaker.mlmodelc"),
            progressCallback: { progress in
                Task { @MainActor in
                    self.downloadProgress = 0.5 + (progress * 0.5) // Second half
                }
            }
        )

        await MainActor.run {
            downloadProgress = 1.0
        }

        logger.info("Models downloaded successfully")
    }

    /// Check if models are available locally
    public func areModelsAvailable() -> Bool {
        let modelsDirectory = getModelsDirectory()
        let segmentationPath = modelsDirectory.appendingPathComponent("pyannote_segmentation.mlmodelc")
        let embeddingPath = modelsDirectory.appendingPathComponent("wespeaker.mlmodelc")

        return FileManager.default.fileExists(atPath: segmentationPath.path) &&
               FileManager.default.fileExists(atPath: embeddingPath.path)
    }

    /// Clean up resources
    public func cleanup() {
        logger.info("Cleaning up iOS CoreML diarizer")

        segmentationModel = nil
        embeddingModel = nil
        isInitialized = false
        modelStatus = .notLoaded
        endBackgroundTask()
    }

    // MARK: - Private Methods

    private func loadModels() async throws {
        let modelsDirectory = getModelsDirectory()

        // Check if models exist, download if needed
        if !areModelsAvailable() {
            try await downloadModels()
        }

        // Load segmentation model
        let segmentationURL = modelsDirectory.appendingPathComponent("pyannote_segmentation.mlmodelc")
        let embeddingURL = modelsDirectory.appendingPathComponent("wespeaker.mlmodelc")

        // Load models on background queue to avoid blocking main thread
        let segModel = try MLModel(contentsOf: segmentationURL)
        let embModel = try MLModel(contentsOf: embeddingURL)

        await MainActor.run {
            self.segmentationModel = segModel
            self.embeddingModel = embModel
        }

        logger.info("CoreML models loaded successfully")
    }

    private func processAudioSamples(_ samples: [Float], chunkSize: Int) async throws -> [SpeakerSegment] {
        var allSegments: [SpeakerSegment] = []

        // Process in chunks to manage memory
        for chunkStart in stride(from: 0, to: samples.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, samples.count)
            let chunk = Array(samples[chunkStart..<chunkEnd])

            let chunkOffset = Double(chunkStart) / 16000.0
            let chunkSegments = try await processChunk(chunk, offset: chunkOffset)

            allSegments.append(contentsOf: chunkSegments)

            // Check memory pressure
            if isMemoryPressureHigh() {
                logger.warning("High memory pressure detected, reducing chunk size")
                break
            }
        }

        return mergeSimilarSegments(allSegments)
    }

    private func processChunk(_ samples: [Float], offset: Double) async throws -> [SpeakerSegment] {
        guard let segmentationModel = segmentationModel else {
            throw DiarizerError.notInitialized
        }

        // Ensure proper chunk size
        let targetSize = 160_000 // 10 seconds at 16kHz
        var paddedSamples = samples

        if samples.count < targetSize {
            paddedSamples += Array(repeating: 0.0, count: targetSize - samples.count)
        } else if samples.count > targetSize {
            paddedSamples = Array(samples.prefix(targetSize))
        }

        // Create MLMultiArray for model input
        let audioArray = try MLMultiArray(shape: [1, 1, NSNumber(value: targetSize)], dataType: .float32)
        for i in 0..<paddedSamples.count {
            audioArray[i] = NSNumber(value: paddedSamples[i])
        }

        // Run segmentation
        let input = try MLDictionaryFeatureProvider(dictionary: ["audio": audioArray])
        let output = try segmentationModel.prediction(from: input)

        // Process output and convert to segments
        return try parseSegmentationOutput(output, offset: offset)
    }

    private func parseSegmentationOutput(_ output: MLFeatureProvider, offset: Double) throws -> [SpeakerSegment] {
        guard let segmentOutput = output.featureValue(for: "segments")?.multiArrayValue else {
            throw DiarizerError.processingFailed("Missing segments output")
        }

        let frames = segmentOutput.shape[1].intValue
        let frameStep = 0.016875 // Time per frame

        var segments: [SpeakerSegment] = []
        var currentSpeaker: Int? = nil
        var segmentStart: Double = offset

        for frameIndex in 0..<frames {
            let frameTime = offset + Double(frameIndex) * frameStep

            // Find dominant speaker for this frame
            var maxValue: Float = -1.0
            var dominantSpeaker: Int = 0

            let numSpeakers = segmentOutput.shape[2].intValue
            for speakerIndex in 0..<numSpeakers {
                let index = frameIndex * numSpeakers + speakerIndex
                let value = segmentOutput[index].floatValue

                if value > maxValue {
                    maxValue = value
                    dominantSpeaker = speakerIndex
                }
            }

            // Create segment when speaker changes
            if let current = currentSpeaker, current != dominantSpeaker {
                let segment = SpeakerSegment(
                    speakerClusterId: current,
                    startTimeSeconds: Float(segmentStart),
                    endTimeSeconds: Float(frameTime),
                    confidenceScore: 0.8 // Base confidence
                )
                segments.append(segment)
                segmentStart = frameTime
            }

            currentSpeaker = dominantSpeaker
        }

        // Add final segment
        if let speaker = currentSpeaker {
            let endTime = offset + Double(frames) * frameStep
            let segment = SpeakerSegment(
                speakerClusterId: speaker,
                startTimeSeconds: Float(segmentStart),
                endTimeSeconds: Float(endTime),
                confidenceScore: 0.8
            )
            segments.append(segment)
        }

        return segments
    }

    private func convertAudioDataToSamples(_ data: Data) throws -> [Float] {
        // Convert audio data to Float32 samples at 16kHz
        // This is a simplified implementation - you might want to use AVAudioConverter
        let sampleCount = data.count / MemoryLayout<Int16>.size
        var samples: [Float] = []
        samples.reserveCapacity(sampleCount)

        data.withUnsafeBytes { bytes in
            let int16Samples = bytes.bindMemory(to: Int16.self)
            for sample in int16Samples {
                // Convert Int16 to Float32 and normalize
                samples.append(Float(sample) / Float(Int16.max))
            }
        }

        return samples
    }

    private func getCurrentChunkSize() -> Int {
        if isMemoryPressureHigh() || ProcessInfo.processInfo.isLowPowerModeEnabled {
            return lowMemoryChunkSize
        }
        return maxChunkSize
    }

    private func isMemoryPressureHigh() -> Bool {
        let memoryUsage = getMemoryUsage()
        return memoryUsage > memoryThreshold
    }

    private func getMemoryUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        return result == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }

    private func mergeSimilarSegments(_ segments: [SpeakerSegment]) -> [SpeakerSegment] {
        guard !segments.isEmpty else { return [] }

        var merged: [SpeakerSegment] = []
        var current = segments[0]

        for i in 1..<segments.count {
            let next = segments[i]

            // Merge if same speaker and segments are close
            if current.speakerClusterId == next.speakerClusterId &&
               next.startTimeSeconds - current.endTimeSeconds < 0.5 {
                current = SpeakerSegment(
                    speakerClusterId: current.speakerClusterId,
                    startTimeSeconds: current.startTimeSeconds,
                    endTimeSeconds: next.endTimeSeconds,
                    confidenceScore: max(current.confidenceScore, next.confidenceScore)
                )
            } else {
                merged.append(current)
                current = next
            }
        }

        merged.append(current)
        return merged
    }

    private func calculateOverallConfidence(_ segments: [SpeakerSegment]) -> Float {
        guard !segments.isEmpty else { return 0.0 }

        let totalConfidence = segments.reduce(0.0) { $0 + $1.confidenceScore }
        return totalConfidence / Float(segments.count)
    }

    private func getModelsDirectory() -> URL {
        if let customDirectory = iosConfig.modelCacheDirectory {
            return customDirectory.appendingPathComponent("coreml", isDirectory: true)
        }

        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsDirectory = documentsDirectory.appendingPathComponent("SpeakerDiarizerModels", isDirectory: true)

        try? FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        return modelsDirectory
    }

    private func downloadModelWithProgress(
        repoPath: String,
        modelName: String,
        outputPath: URL,
        progressCallback: @escaping (Float) -> Void
    ) async throws {
        // Create directory
        try FileManager.default.createDirectory(at: outputPath, withIntermediateDirectories: true)

        // Files to download
        let files = ["model.mil", "coremldata.bin", "metadata.json", "weights/weight.bin"]

        for (index, fileName) in files.enumerated() {
            let fileURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/\(fileName)")!

            do {
                let (tempFile, response) = try await URLSession.shared.download(from: fileURL)

                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    let destinationPath = outputPath.appendingPathComponent(fileName)

                    // Create subdirectory if needed
                    if fileName.contains("/") {
                        let parentDir = destinationPath.deletingLastPathComponent()
                        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
                    }

                    try? FileManager.default.removeItem(at: destinationPath)
                    try FileManager.default.moveItem(at: tempFile, to: destinationPath)

                    logger.info("Downloaded \(fileName)")
                }

                let progress = Float(index + 1) / Float(files.count)
                progressCallback(progress)

            } catch {
                logger.error("Failed to download \(fileName): \(error)")
                if fileName == "coremldata.bin" || fileName == "model.mil" {
                    throw DiarizerError.modelDownloadFailed
                }
            }
        }
    }

    // MARK: - Background Task Management

    private func beginBackgroundTask() {
        #if canImport(UIKit)
        guard backgroundTaskID == .invalid else { return }

        backgroundTaskID = UIApplication.shared.beginBackgroundTask(withName: "DiarizerProcessing") {
            self.endBackgroundTask()
        }
        #endif
    }

    private func endBackgroundTask() {
        #if canImport(UIKit)
        guard backgroundTaskID != .invalid else { return }

        UIApplication.shared.endBackgroundTask(backgroundTaskID)
        backgroundTaskID = .invalid
        #endif
    }

    // MARK: - Notification Handling

    private func setupMemoryPressureNotifications() {
        #if canImport(UIKit)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryWarning()
        }
        #endif
    }

    private func setupAppStateNotifications() {
        #if canImport(UIKit)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppDidEnterBackground()
        }

        NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppWillEnterForeground()
        }
        #endif
    }

    private func handleMemoryWarning() {
        logger.warning("Memory warning received, cleaning up")

        // Clear any cached data if needed
        if !isProcessing {
            // Optionally unload models to free memory
            // segmentationModel = nil
            // embeddingModel = nil
        }
    }

    private func handleAppDidEnterBackground() {
        logger.info("App entered background")
        endBackgroundTask()
    }

    private func handleAppWillEnterForeground() {
        logger.info("App will enter foreground")
    }
}

// MARK: - Extensions

@available(iOS 16.0, *)
extension iOSCoreMLDiarizerManager {

    /// Check if models are currently loading
    private func isLoading() -> Bool {
        switch modelStatus {
        case .loading, .downloading:
            return true
        default:
            return false
        }
    }

    /// Convenience method for SwiftUI integration
    public func initializeIfNeeded() {
        guard !isInitialized && !isLoading() else { return }

        Task {
            do {
                try await initialize()
            } catch {
                logger.error("Failed to initialize: \(error)")
            }
        }
    }

    /// Get human-readable status
    public var statusDescription: String {
        switch modelStatus {
        case .notLoaded:
            return "Models not loaded"
        case .downloading:
            return "Downloading models... \(Int(downloadProgress * 100))%"
        case .loading:
            return "Loading models..."
        case .ready:
            return isProcessing ? "Processing audio..." : "Ready"
        case .error(let error):
            return "Error: \(error.localizedDescription)"
        }
    }
}
