//
//  ASRConfig.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

public struct ASRConfig: Sendable {
    public let sampleRate: Int
    public let maxSymbolsPerFrame: Int
    public let modelCacheDirectory: URL?
    public let enableDebug: Bool

    // iOS Real-Time Optimizations
    public let realtimeMode: Bool
    public let chunkSizeMs: Int           // Chunk size in milliseconds
    public let maxLatencyMs: Int          // Maximum acceptable latency

    // TDT + Post-Processing
    public let enableAdvancedPostProcessing: Bool  // Vocabulary-based post-processing
    public let vocabularyConstraints: Bool // Use vocab for token filtering
    public let tdtConfig: TDTConfig       // TDT-specific configuration

    public static let `default` = ASRConfig()

    // Fast benchmark preset for maximum performance
    public static let fastBenchmark = ASRConfig(
        maxSymbolsPerFrame: 3,        // More aggressive decoding
        realtimeMode: false,          // Batch mode
        chunkSizeMs: 2000,           // Larger chunks
        enableAdvancedPostProcessing: true,
        vocabularyConstraints: false,
        tdtConfig: TDTConfig(
            durations: [0, 1, 2, 3, 4],
            includeTokenDuration: true,
            includeDurationConfidence: false,
            maxSymbolsPerStep: 3     // More aggressive
        )
    )

    // iOS Real-Time preset with TDT + Post-Processing
    public static let realtimeIOS = ASRConfig(
        realtimeMode: true,
        chunkSizeMs: 200,            // 200ms chunks for responsiveness
        maxLatencyMs: 100,           // 100ms max latency
        enableAdvancedPostProcessing: true,  // Vocabulary post-processing
        vocabularyConstraints: true,  // Use vocab constraints during decoding
        tdtConfig: TDTConfig(durations: [0, 1, 2, 3, 4], includeTokenDuration: true, includeDurationConfidence: false, maxSymbolsPerStep: 2)
    )

    public init(
        sampleRate: Int = 16000,
        maxSymbolsPerFrame: Int = 3,      // Faster default
        modelCacheDirectory: URL? = nil,
        enableDebug: Bool = false,
        realtimeMode: Bool = false,
        chunkSizeMs: Int = 1500,          // Larger chunks by default
        maxLatencyMs: Int = 500,          // Default 500ms latency
        enableAdvancedPostProcessing: Bool = true,  // Post-processing enabled by default
        vocabularyConstraints: Bool = false,  // Vocab constraints disabled by default
        tdtConfig: TDTConfig = .default   // TDT configuration
    ) {
        self.sampleRate = sampleRate
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
        self.modelCacheDirectory = modelCacheDirectory
        self.enableDebug = enableDebug
        self.realtimeMode = realtimeMode
        self.chunkSizeMs = chunkSizeMs
        self.maxLatencyMs = maxLatencyMs
        self.enableAdvancedPostProcessing = enableAdvancedPostProcessing
        self.vocabularyConstraints = vocabularyConstraints
        self.tdtConfig = tdtConfig
    }
}