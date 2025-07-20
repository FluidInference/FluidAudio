//
//  ASRResult.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

public struct ASRResult: Sendable {
    public let text: String
    public let confidence: Float
    public let duration: TimeInterval
    public let processingTime: TimeInterval
    public let tokenTimings: [TokenTiming]?  // TDT support

    public init(text: String, confidence: Float, duration: TimeInterval, processingTime: TimeInterval, tokenTimings: [TokenTiming]? = nil) {
        self.text = text
        self.confidence = confidence
        self.duration = duration
        self.processingTime = processingTime
        self.tokenTimings = tokenTimings
    }
}

/// Token Duration Timing for advanced post-processing
public struct TokenTiming: Sendable {
    public let token: String
    public let tokenId: Int
    public let startTime: TimeInterval
    public let endTime: TimeInterval
    public let confidence: Float

    public init(token: String, tokenId: Int, startTime: TimeInterval, endTime: TimeInterval, confidence: Float) {
        self.token = token
        self.tokenId = tokenId
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
    }
}