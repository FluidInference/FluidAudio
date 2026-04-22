import Foundation
import OSLog

/// A session manager for handling multiple sliding-window ASR instances with shared model loading.
/// This ensures models are loaded only once and shared across all streams.
public actor SlidingWindowAsrSession {
    private let logger = AppLogger(category: "SlidingWindowSession")
    private var loadedModels: AsrModels?
    private var streams: [AudioSource: SlidingWindowAsrManager] = [:]

    /// Initialize a new streaming session
    public init() {
        logger.info("Created new SlidingWindowAsrSession")
    }

    /// Load ASR models for the session (called automatically if needed)
    /// Models are cached and shared across all streams in this session
    public func loadModels() async throws {
        guard loadedModels == nil else {
            logger.info("Models already loaded, skipping initialization")
            return
        }

        logger.info("Loading ASR models for session...")
        loadedModels = try await AsrModels.downloadAndLoad()
        logger.info("ASR models loaded successfully")
    }

    /// Create a new streaming ASR instance for a specific audio source
    /// - Parameters:
    ///   - source: The audio source (microphone or system)
    ///   - config: Configuration for the streaming behavior
    /// - Returns: A configured SlidingWindowAsrManager instance
    public func createStream(
        source: AudioSource,
        config: SlidingWindowAsrConfig = .default
    ) async throws -> SlidingWindowAsrManager {
        // Check if we already have a stream for this source
        if let existingStream = streams[source] {
            logger.warning(
                "Stream already exists for source: \(String(describing: source)). Returning existing stream.")
            return existingStream
        }

        // Ensure models are loaded
        if loadedModels == nil {
            try await loadModels()
        }

        guard let models = loadedModels else {
            throw SlidingWindowAsrError.modelsNotLoaded
        }

        logger.info("Creating new stream for source: \(String(describing: source))")

        // Create new stream with pre-loaded models
        let stream = SlidingWindowAsrManager(config: config)
        try await stream.loadModels(models)
        try await stream.startStreaming(source: source)

        // Store reference
        streams[source] = stream

        return stream
    }

    /// Get an existing stream for a source
    /// - Parameter source: The audio source
    /// - Returns: The stream if it exists, nil otherwise
    public func getStream(for source: AudioSource) -> SlidingWindowAsrManager? {
        return streams[source]
    }

    /// Remove a stream from the session
    /// - Parameter source: The audio source to remove
    public func removeStream(for source: AudioSource) {
        if streams.removeValue(forKey: source) != nil {
            logger.info("Removed stream for source: \(String(describing: source))")
        }
    }

    /// Get all active streams
    public var activeStreams: [AudioSource: SlidingWindowAsrManager] {
        return streams
    }

    // MARK: - Generic Engine Support

    /// Engines created via the factory (stored separately from TDT-specific streams)
    private var engines: [AudioSource: any StreamingAsrManager] = [:]

    /// Create a true streaming ASR engine using the factory.
    ///
    /// Unlike `createStream(source:config:)` which uses TDT (sliding-window), this method
    /// creates engines with native streaming architectures (EOU, Nemotron) via `StreamingModelVariant`.
    ///
    /// - Parameters:
    ///   - variant: The streaming model variant to use.
    ///   - source: The audio source label for tracking (default: `.microphone`).
    /// - Returns: A loaded and ready-to-use streaming ASR engine.
    public func createEngine(
        variant: StreamingModelVariant,
        source: AudioSource = .microphone
    ) async throws -> any StreamingAsrManager {
        if let existing = engines[source] {
            logger.warning(
                "Engine already exists for source: \(String(describing: source)). Returning existing engine.")
            return existing
        }

        logger.info(
            "Creating \(variant.displayName) engine for source: \(String(describing: source))")

        let engine = variant.createManager()
        try await engine.loadModels()
        engines[source] = engine

        return engine
    }

    /// Get an existing engine for a source
    public func getEngine(for source: AudioSource) -> (any StreamingAsrManager)? {
        return engines[source]
    }

    /// Remove an engine for a source
    public func removeEngine(for source: AudioSource) {
        if engines.removeValue(forKey: source) != nil {
            logger.info("Removed engine for source: \(String(describing: source))")
        }
    }

    /// Clean up all streams and release resources
    public func cleanup() async {
        logger.info("Cleaning up SlidingWindowAsrSession...")

        // Cancel TDT streams
        for (source, stream) in streams {
            await stream.cancel()
            logger.info("Cancelled stream for source: \(String(describing: source))")
        }

        // Reset generic engines
        for (source, engine) in engines {
            try? await engine.reset()
            logger.info("Reset engine for source: \(String(describing: source))")
        }

        // Clear references
        streams.removeAll()
        engines.removeAll()
        loadedModels = nil

        logger.info("SlidingWindowAsrSession cleanup complete")
    }
}

/// Errors specific to sliding-window ASR session
public enum SlidingWindowAsrError: LocalizedError {
    case modelsNotLoaded
    case streamAlreadyExists(AudioSource)
    case audioBufferProcessingFailed(Error)
    case audioConversionFailed(Error)
    case modelProcessingFailed(Error)
    case bufferOverflow
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .modelsNotLoaded:
            return "ASR models have not been loaded"
        case .streamAlreadyExists(let source):
            return "A stream already exists for source: \(source)"
        case .audioBufferProcessingFailed(let error):
            return "Audio buffer processing failed: \(error.localizedDescription)"
        case .audioConversionFailed(let error):
            return "Audio conversion failed: \(error.localizedDescription)"
        case .modelProcessingFailed(let error):
            return "Model processing failed: \(error.localizedDescription)"
        case .bufferOverflow:
            return "Audio buffer overflow occurred"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        }
    }
}
