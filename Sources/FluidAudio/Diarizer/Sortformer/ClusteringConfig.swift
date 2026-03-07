//
//  ClusteringConfig.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/18/26.
//

import Foundation

public struct ClusteringConfig {    
    /// Intra-cluster separation threshold
    var clusteringThreshold: Float
    
    /// Chamfer distance threshold to match with another speaker profile
    var matchThreshold: Float
    
    /// Maximum number of speakers supported by the EEND model
    let numSlots: Int
    
    init(
        clusteringThreshold: Float = 0.25,
        matchThreshold: Float = 0.3,
        numSlots: Int = 4,
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.matchThreshold = matchThreshold
        self.numSlots = numSlots
    }
    
    init(from config: SortformerTimelineConfig) {
        self.clusteringThreshold = config.clusteringThreshold
        self.matchThreshold = config.matchThreshold
        self.numSlots = config.numSpeakers
    }
}
