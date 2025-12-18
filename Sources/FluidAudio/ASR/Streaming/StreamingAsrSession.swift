import Foundation
import OSLog

/// A session manager for handling multiple streaming ASR instances with shared model loading
/// This ensures models are loaded only once and shared across all streams
public actor StreamingAsrSession {
    private let logger = AppLogger(category: "StreamingSession")
    private var loadedModels: AsrModels?
    private var streams: [AudioSource: StreamingAsrManager] = [:]

    /// Initialize a new streaming session
    public init() {
        logger.info("Created new StreamingAsrSession")
    }

    /// Load ASR models for the session (called automatically if needed)
    /// Models are cached and shared across all streams in this session
    public func initialize() async throws {
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
    /// - Returns: A configured StreamingAsrManager instance
    public func createStream(
        source: AudioSource,
        config: StreamingAsrConfig = .default
    ) async throws -> StreamingAsrManager {
        // Check if we already have a stream for this source
        if let existingStream = streams[source] {
            logger.warning(
                "Stream already exists for source: \(String(describing: source)). Returning existing stream.")
            return existingStream
        }

        // Ensure models are loaded
        if loadedModels == nil {
            try await initialize()
        }

        guard let models = loadedModels else {
            throw StreamingAsrError.modelsNotLoaded
        }

        logger.info("Creating new stream for source: \(String(describing: source))")

        // Create new stream with pre-loaded models
        let stream = StreamingAsrManager(config: config)
        try await stream.start(models: models, source: source)

        // Store reference
        streams[source] = stream

        return stream
    }

    /// Get an existing stream for a source
    /// - Parameter source: The audio source
    /// - Returns: The stream if it exists, nil otherwise
    public func getStream(for source: AudioSource) -> StreamingAsrManager? {
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
    public var activeStreams: [AudioSource: StreamingAsrManager] {
        return streams
    }

    /// Clean up all streams and release resources
    public func cleanup() async {
        logger.info("Cleaning up StreamingAsrSession...")

        // Cancel all streams
        for (source, stream) in streams {
            await stream.cancel()
            logger.info("Cancelled stream for source: \(String(describing: source))")
        }

        // Clear references
        streams.removeAll()
        loadedModels = nil

        logger.info("StreamingAsrSession cleanup complete")
    }
}

/// Errors specific to streaming ASR session
public enum StreamingAsrError: LocalizedError {
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
