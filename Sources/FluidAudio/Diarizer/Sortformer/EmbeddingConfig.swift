//
//  TitaNetConfig.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/13/26.
//

import Foundation

public struct EmbeddingConfig {
    /// Variant of the TitaNet model
    public var modelVariant: TitaNetVariant
    
    /// Minimum number of frames in a segment to receive an embedding
    public let minEmbeddingFrames: Int
    
    /// Minimum number of frames one embedding may belong to
    public let maxEmbeddingFrames: Int
    
    /// Maximum number of frames between embeddings within the same segment
    public let maxFramesSkipped: Int

    /// Minimum number of frames between same-speaker segments to avoid merging
    public let minFramesOff: Int

    /// Maximum number of frames outside a segment an embedding may cover
    public let maxOutsideFrames: Int
    
    // Non-configurable properties
    public let frameDuration: Float = 0.08
    public let subsamplingFactor: Int = 8
    public let melFeatures: Int = 80
    public let melStride: Int = 160
    public let melWindow: Int = 400
    public let melPadTo: Int = 16
    public static let embeddingFeatures: Int = 192

    // Computed properties
    var minMelLength: Int { minEmbeddingFrames * subsamplingFactor }
    var maxMelLength: Int { IndexUtils.nextMultiple(of: melPadTo, for: maxEmbeddingFrames * subsamplingFactor) }
    var melFeaturesPerFrame: Int { subsamplingFactor * melFeatures }
    var minInputSamples: Int { minMelLength * melStride }
    var maxInputSamples: Int { maxMelLength * melStride }
    var minInputDuration: Float { Float(minEmbeddingFrames) * frameDuration }
    var maxInputDuration: Float { Float(maxEmbeddingFrames) * frameDuration }
    
    public static let `default` = EmbeddingConfig()
    public static let `large2_48s` = EmbeddingConfig(modelVariant: .large2_48s)
    public static let `large3_04s` = EmbeddingConfig(modelVariant: .large3_04s)
    public static let `small2_48s` = EmbeddingConfig(modelVariant: .small2_48s)
    public static let `small3_04s` = EmbeddingConfig(modelVariant: .small3_04s)
    
    public init(
        modelVariant: TitaNetVariant = .large2_48s,
        minFramesOn: Int = 12,
        minFramesOff: Int = 3,
        maxFramesSkipped: Int = 6,
        maxOutsideFrames: Int = 2,
    ) {
        self.modelVariant = modelVariant
        self.minEmbeddingFrames = minFramesOn
        self.maxEmbeddingFrames = modelVariant.maxInputFrames
        self.minFramesOff = minFramesOff
        self.maxOutsideFrames = maxOutsideFrames
        self.maxFramesSkipped = maxFramesSkipped
        
        guard minFramesOn <= maxEmbeddingFrames else {
            preconditionFailure(
                "Invalid EmbeddingConfiguration: minFramesOn must be <= \(maxEmbeddingFrames), " +
                "which is the maximum length supported by \(modelVariant.name).")
        }
    }
}
