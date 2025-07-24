//
//  VadTypes.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

/// VAD processing result
public struct VadResult: Sendable {
    public let probability: Float  // Voice activity probability (0.0-1.0)
    public let isVoiceActive: Bool // Whether voice is detected
    public let processingTime: TimeInterval
    public let snrValue: Float?    // Signal-to-Noise Ratio (dB) if calculated
    public let spectralFeatures: SpectralFeatures?  // Spectral analysis results

    public init(probability: Float, isVoiceActive: Bool, processingTime: TimeInterval, snrValue: Float? = nil, spectralFeatures: SpectralFeatures? = nil) {
        self.probability = probability
        self.isVoiceActive = isVoiceActive
        self.processingTime = processingTime
        self.snrValue = snrValue
        self.spectralFeatures = spectralFeatures
    }
}

/// Spectral features for enhanced VAD
public struct SpectralFeatures: Sendable {
    public let spectralCentroid: Float      // Center frequency of the spectrum
    public let spectralRolloff: Float       // Frequency below which 85% of energy is contained
    public let spectralFlux: Float          // Measure of spectral change
    public let mfccFeatures: [Float]        // MFCC coefficients (first 13)
    public let zeroCrossingRate: Float      // Zero crossing rate
    public let spectralEntropy: Float       // Measure of spectral complexity

    public init(spectralCentroid: Float, spectralRolloff: Float, spectralFlux: Float, mfccFeatures: [Float], zeroCrossingRate: Float, spectralEntropy: Float) {
        self.spectralCentroid = spectralCentroid
        self.spectralRolloff = spectralRolloff
        self.spectralFlux = spectralFlux
        self.mfccFeatures = mfccFeatures
        self.zeroCrossingRate = zeroCrossingRate
        self.spectralEntropy = spectralEntropy
    }
}

/// VAD error types
public enum VadError: Error, LocalizedError {
    case notInitialized
    case modelLoadingFailed
    case modelProcessingFailed(String)
    case invalidAudioData
    case invalidModelPath
    case modelDownloadFailed
    case modelCompilationFailed

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "VAD system not initialized. Call initialize() first."
        case .modelLoadingFailed:
            return "Failed to load VAD models."
        case .modelProcessingFailed(let message):
            return "Model processing failed: \(message)"
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .invalidModelPath:
            return "Invalid model path provided."
        case .modelDownloadFailed:
            return "Failed to download VAD models from Hugging Face."
        case .modelCompilationFailed:
            return "Failed to compile VAD models after multiple attempts."
        }
    }
}