//
//  DiarizerConfig.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

public struct DiarizerConfig: Sendable {
    public var clusteringThreshold: Float = 0.7  // Similarity threshold for grouping speakers (0.0-1.0, higher = stricter)
    public var minDurationOn: Float = 1.0  // Minimum duration (seconds) for a speaker segment to be considered valid
    public var minDurationOff: Float = 0.5  // Minimum silence duration (seconds) between different speakers
    public var numClusters: Int = -1  // Number of speakers to detect (-1 = auto-detect)
    public var minActivityThreshold: Float = 10.0  // Minimum activity threshold (frames) for speaker to be considered active
    public var debugMode: Bool = false
    public var modelCacheDirectory: URL?

    public static let `default` = DiarizerConfig()

    /// Platform-optimized configuration for iOS devices
    #if os(iOS)
    public static let iosOptimized = DiarizerConfig(
        clusteringThreshold: 0.7,
        minDurationOn: 1.0,
        minDurationOff: 0.5,
        numClusters: -1,
        minActivityThreshold: 10.0,
        debugMode: false,
        modelCacheDirectory: nil
    )
    #endif

    public init(
        clusteringThreshold: Float = 0.7,
        minDurationOn: Float = 1.0,
        minDurationOff: Float = 0.5,
        numClusters: Int = -1,
        minActivityThreshold: Float = 10.0,
        debugMode: Bool = false,
        modelCacheDirectory: URL? = nil
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.minDurationOn = minDurationOn
        self.minDurationOff = minDurationOff
        self.numClusters = numClusters
        self.minActivityThreshold = minActivityThreshold
        self.debugMode = debugMode
        self.modelCacheDirectory = modelCacheDirectory
    }
}