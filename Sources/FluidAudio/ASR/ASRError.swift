//
//  ASRError.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

public enum ASRError: Error, LocalizedError {
    case notInitialized
    case invalidAudioData
    case modelLoadFailed
    case processingFailed(String)
    case invalidDuration
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
        case .invalidDuration:
            return "Audio must be exactly 10 seconds (160,000 samples at 16kHz)."
        case .modelCompilationFailed:
            return "CoreML model compilation failed after recovery attempts."
        }
    }
}