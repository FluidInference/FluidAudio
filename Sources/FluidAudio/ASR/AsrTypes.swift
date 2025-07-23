//
//  AsrTypes.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

// MARK: - Configuration

public struct ASRConfig: Sendable {
    public let sampleRate: Int
    public let maxSymbolsPerFrame: Int
    public let enableDebug: Bool
    public let realtimeMode: Bool
    public let chunkSizeMs: Int
    public let tdtConfig: TdtConfig

    public static let `default` = ASRConfig()

    public static let fastBenchmark = ASRConfig(
        maxSymbolsPerFrame: 3,
        realtimeMode: false,
        chunkSizeMs: 2000,
        tdtConfig: TdtConfig(
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
        tdtConfig: TdtConfig = .default
    ) {
        self.sampleRate = sampleRate
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
        self.enableDebug = enableDebug
        self.realtimeMode = realtimeMode
        self.chunkSizeMs = chunkSizeMs
        self.tdtConfig = tdtConfig
    }
}

// MARK: - Results

public struct ASRResult: Sendable {
    public let text: String
    public let confidence: Float
    public let duration: TimeInterval
    public let processingTime: TimeInterval
    public let tokenTimings: [TokenTiming]?

    public init(text: String, confidence: Float, duration: TimeInterval, processingTime: TimeInterval, tokenTimings: [TokenTiming]? = nil) {
        self.text = text
        self.confidence = confidence
        self.duration = duration
        self.processingTime = processingTime
        self.tokenTimings = tokenTimings
    }
}

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

// MARK: - Errors

public enum ASRError: Error, LocalizedError {
    case notInitialized
    case invalidAudioData
    case modelLoadFailed
    case processingFailed(String)
    case modelCompilationFailed
    
    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "AsrManager not initialized. Call initialize() first."
        case .invalidAudioData:
            return "Invalid audio data provided. Must be at least 1 second of 16kHz audio."
        case .modelLoadFailed:
            return "Failed to load Parakeet CoreML models."
        case .processingFailed(let message):
            return "ASR processing failed: \(message)"
        case .modelCompilationFailed:
            return "CoreML model compilation failed after recovery attempts."
        }
    }
}