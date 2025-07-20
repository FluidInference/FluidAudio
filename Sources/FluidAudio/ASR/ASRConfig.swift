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
    public let enableDebug: Bool
    public let realtimeMode: Bool
    public let chunkSizeMs: Int
    public let tdtConfig: TDTConfig

    public static let `default` = ASRConfig()

    // Fast benchmark preset for maximum performance
    public static let fastBenchmark = ASRConfig(
        maxSymbolsPerFrame: 3,
        realtimeMode: false,
        chunkSizeMs: 2000,
        tdtConfig: TDTConfig(
            durations: [0, 1, 2, 3, 4],
            includeTokenDuration: true,
            includeDurationConfidence: false,
            maxSymbolsPerStep: 3
        )
    )

    public init(
        sampleRate: Int = 16000,
        maxSymbolsPerFrame: Int = 3,
        enableDebug: Bool = false,
        realtimeMode: Bool = false,
        chunkSizeMs: Int = 1500,
        tdtConfig: TDTConfig = .default
    ) {
        self.sampleRate = sampleRate
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
        self.enableDebug = enableDebug
        self.realtimeMode = realtimeMode
        self.chunkSizeMs = chunkSizeMs
        self.tdtConfig = tdtConfig
    }
}