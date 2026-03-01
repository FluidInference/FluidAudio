//
//  EmbeddingManager.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/15/26.
//

import Foundation
import Accelerate
import OrderedCollections

/// Manages speaker embedding extraction asynchronously
/// Handles segment merging, pruning, and embedding lifecycle
public class EmbeddingManager {
    
    // MARK: - Configuration
    
    /// Embedding extraction configuration
    public let config: EmbeddingConfig
    
    // MARK: - State
    
    /// The embedding extractor
    private var extractor: TitaNetEmbeddingExtractor?
    private var preprocessor: NeMoMelSpectrogram
    
    /// Audio buffer for extraction (circular buffer of audio samples)
    private var melFeatures: [Float] = []
    /// First Sortformer frame available in the mel buffer (in Sortformer frame units, not mel features)
    private var firstMelFrame: Int = 0
    
    /// Lock for thread-safe access
    private let queue = DispatchQueue(label: "FluidAudio.EmbeddingManager")
    
    private var availibleEmbeddings: [SpeakerEmbedding] = []
    
    /// Logger
    private static let logger = AppLogger(category: "EmbeddingManager")
    
    // MARK: - Initialization
    
    public init(
        config: EmbeddingConfig = EmbeddingConfig(),
        frameDurationSeconds: Float = 0.08  // Match Sortformer's frame duration
    ) {
        self.config = config
        self.preprocessor = NeMoMelSpectrogram(nMels: config.melFeatures, padTo: config.melPadTo)
        // Initialize extractor lazily on first use
    }
    
    /// Initialize the extractor (call before first use)
    public func initialize() throws {
        try queue.sync(flags: .barrier) {
            if extractor == nil {
                extractor = try TitaNetEmbeddingExtractor(config:  config)
            }
        }
    }
    
    // MARK: - Audio Management
    
    /// Add audio to the mel feature buffer
    public func addAudio(from buffer: ArraySlice<Float>, lastAudioSample: Float = 0) {
        queue.sync(flags: .barrier) {
            let (mels, _, _) = preprocessor.computeFlatTransposed(audio: buffer, lastAudioSample: lastAudioSample)
            melFeatures.append(contentsOf: mels)
        }
    }
    
    /// Number of Sortformer frames currently available in the mel buffer
    private var availableMelFrames: Int {
        melFeatures.count / config.melFeatures / config.subsamplingFactor
    }
    
    /// Last Sortformer frame available in the mel buffer (exclusive)
    private var lastMelFrame: Int {
        firstMelFrame + availableMelFrames
    }
    
    // MARK: - Embedding Extraction
    
    /// Process pending embedding requests from a timeline asynchronously
    public func processRequests(_ requests: [EmbeddingRequest]) throws -> [SpeakerEmbedding] {
        try queue.sync(flags: .barrier) {
            guard let extractor else {
                throw EmbeddingManagerError.extractorNotInitialized
            }
            
            guard !requests.isEmpty else {
                return [] as [SpeakerEmbedding]
            }
            
            var embeddings: [SpeakerEmbedding] = []
            
            for request in requests {
                // Check if request length is valid
                guard validateRequest(request) else {
                    continue
                }
                
                // Calculate mel feature indices relative to buffer start
                let relativeStartFrame = request.startFrame - firstMelFrame
                let melStartIndex = relativeStartFrame * config.subsamplingFactor * config.melFeatures
                
                // Calculate padded mel length for the model
                let melLength = request.length * config.subsamplingFactor
                let paddedMelLength = IndexUtils.nextMultiple(
                    of: config.melPadTo,
                    for: melLength
                )
                let melEndIndex = melStartIndex + melLength * config.melFeatures
                
                // Bounds check on mel features array
                guard melStartIndex >= 0 && melEndIndex <= melFeatures.count else {
                    Self.logger.warning("Mel indices out of bounds: [\(melStartIndex)-\(melEndIndex)] for buffer of \(melFeatures.count) features")
                    continue
                }
                
                let embeddingVector = try extractor.extractEmbedding(
                    mels: melFeatures[melStartIndex..<melEndIndex],
                    melLength: paddedMelLength
                )
                
                let embedding = SpeakerEmbedding(
                    embedding: embeddingVector,
                    startFrame: request.startFrame,
                    endFrame: request.endFrame,
                )
                
                embeddings.append(embedding)
            }
            
            return embeddings
        }
    }
    
    public func takeMatches(for segment: EmbeddingSegment) -> [SpeakerEmbedding] {
        queue.sync(flags: .barrier) {
            guard !availibleEmbeddings.isEmpty else {
                return []
            }
            
            let count = availibleEmbeddings.count
            var embeddings: [SpeakerEmbedding] = []
            for i in (0..<count).reversed() {
                if availibleEmbeddings[i].framesOutside(of: segment) <= config.maxOutsideFrames {
                    embeddings.append(availibleEmbeddings.remove(at: i))
                }
            }
            
            return embeddings
        }
    }
    
    /// Cache all embeddings from a segment
    public func returnEmbeddings(from segment: EmbeddingSegment) {
        queue.sync(flags: .barrier) {
            availibleEmbeddings.append(contentsOf: segment.embeddings)
            segment.clearEmbeddings()
        }
    }
    
    /// Cache all embeddings from a bunch of segments
    public func returnEmbeddings(from segments: [EmbeddingSegment]) {
        queue.sync(flags: .barrier) {
            for segment in segments {
                availibleEmbeddings.append(contentsOf: segment.embeddings)
                segment.clearEmbeddings()
            }
        }
    }
    
    /// Clean all outdated frames and embeddings
    public func dropFrames(before firstTentativeFrame: Int) {
        queue.sync(flags: .barrier) {
            // Don't drop if the new frame is not ahead of current start
            guard firstTentativeFrame > firstMelFrame else {
                return
            }
            
            availibleEmbeddings.removeAll {
                $0.endFrame < firstTentativeFrame
            }
            
            // Calculate how many Sortformer frames to drop
            let framesToDrop = firstTentativeFrame - firstMelFrame
            
            // Calculate how many mel features to drop
            // Each Sortformer frame = subsamplingFactor mel frames
            // Each mel frame = melFeatures features
            let featuresToDrop = framesToDrop * config.subsamplingFactor * config.melFeatures
            
            // Don't drop more than we have
            let actualFeaturesToDrop = min(featuresToDrop, melFeatures.count)
            
            if actualFeaturesToDrop > 0 {
                melFeatures.removeFirst(actualFeaturesToDrop)
                firstMelFrame = firstTentativeFrame
                Self.logger.debug("Dropped \(framesToDrop) frames, mel buffer now starts at frame \(firstMelFrame)")
            }
        }
    }
    
    /// Reset the manager to initial state
    public func reset() {
        queue.sync(flags: .barrier) {
            melFeatures.removeAll(keepingCapacity: true)
            firstMelFrame = 0
            availibleEmbeddings.removeAll(keepingCapacity: true)
            Self.logger.debug("Reset embedding manager state")
        }
    }
    
    /// Validate an embedding extraction request with verbose feedback upon failure.
    /// - Returns: `true` if the request is valid, `false` if not
    private func validateRequest(_ request: EmbeddingRequest) -> Bool {
        guard request.length <= config.maxEmbeddingFrames else {
            Self.logger.debug("Request [\(request.startFrame)-\(request.endFrame)] too long (\(request.length) > \(config.maxEmbeddingFrames)), skipping")
            return false
        }
        
        guard request.length >= config.minEmbeddingFrames else {
            Self.logger.debug("Request [\(request.startFrame)-\(request.endFrame)] too short (\(request.length) < \(config.minEmbeddingFrames)), skipping")
            return false
        }
        
        // Check if request is within available mel buffer range
        guard request.startFrame >= firstMelFrame else {
            Self.logger.debug("Request [\(request.startFrame)-\(request.endFrame)] starts before mel buffer (first frame: \(firstMelFrame)), skipping")
            return false
        }
        
        guard request.endFrame <= lastMelFrame else {
            Self.logger.debug("Request [\(request.startFrame)-\(request.endFrame)] ends after mel buffer (last frame: \(lastMelFrame)), skipping")
            return false
        }
        
        return true
    }
}

// MARK: - Errors

public enum EmbeddingManagerError: Error, LocalizedError {
    case audioNotAvailable
    case extractorNotInitialized
    
    public var errorDescription: String? {
        switch self {
        case .audioNotAvailable:
            return "Audio not available in buffer for requested frame range"
        case .extractorNotInitialized:
            return "Embedding extractor not initialized. Call initialize() first."
        }
    }
}
