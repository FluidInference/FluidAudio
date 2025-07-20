//
//  TDTConfiguration.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

/// Token-and-Duration Transducer (TDT) configuration
public struct TDTConfig: Sendable {
    public let durations: [Int]
    public let includeTokenDuration: Bool
    public let includeDurationConfidence: Bool
    public let maxSymbolsPerStep: Int?

    public static let `default` = TDTConfig()

    public init(
        durations: [Int] = [0, 1, 2, 3, 4],  // Fixed: Match notebook training
        includeTokenDuration: Bool = true,
        includeDurationConfidence: Bool = false,
        maxSymbolsPerStep: Int? = nil
    ) {
        self.durations = durations
        self.includeTokenDuration = includeTokenDuration
        self.includeDurationConfidence = includeDurationConfidence
        self.maxSymbolsPerStep = maxSymbolsPerStep
    }
}

/// Hypothesis for TDT beam search decoding
public struct TDTHypothesis: Sendable {
    public var score: Float
    public var ySequence: [Int]
    internal var decState: DecoderState?
    public var timestamps: [Int]
    public var tokenDurations: [Int]
    public var lastToken: Int?

    public init() {
        self.score = 0.0
        self.ySequence = []
        self.decState = nil
        self.timestamps = []
        self.tokenDurations = []
        self.lastToken = nil
    }
}