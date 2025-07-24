//
//  VadConfig.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation
import CoreML

/// Configuration for VAD processing
public struct VadConfig: Sendable {
    public var threshold: Float = 0.3  // Voice activity threshold (0.0-1.0) - lowered for better sensitivity
    public var chunkSize: Int = 512   // Audio chunk size for processing
    public var sampleRate: Int = 16000 // Sample rate for audio processing
    public var modelCacheDirectory: URL?
    public var debugMode: Bool = true
    public var adaptiveThreshold: Bool = false  // Disable adaptive thresholding temporarily
    public var minThreshold: Float = 0.1       // Minimum threshold for adaptive mode
    public var maxThreshold: Float = 0.7       // Maximum threshold for adaptive mode
    public var computeUnits: MLComputeUnits = .cpuAndNeuralEngine  // Preferred compute units

    // SNR and noise detection parameters
    public var enableSNRFiltering: Bool = true      // Enable SNR-based filtering for better noise rejection
    public var minSNRThreshold: Float = 6.0         // Minimum SNR for speech detection (dB) - more aggressive
    public var noiseFloorWindow: Int = 100          // Window size for noise floor estimation
    public var spectralRolloffThreshold: Float = 0.85  // Threshold for spectral rolloff
    public var spectralCentroidRange: (min: Float, max: Float) = (200.0, 8000.0)  // Expected speech range (Hz)

    public static let `default` = VadConfig()

    /// Platform-optimized configuration for iOS devices
    #if os(iOS)
    public static let iosOptimized = VadConfig(
        threshold: 0.445,  // Optimized threshold for iOS
        chunkSize: 512,
        sampleRate: 16000,
        modelCacheDirectory: nil,
        debugMode: false,  // Disable debug mode on iOS for performance
        adaptiveThreshold: true,
        minThreshold: 0.1,
        maxThreshold: 0.7,
        computeUnits: .cpuAndNeuralEngine,  // Prefer Neural Engine on iOS
        enableSNRFiltering: true,
        minSNRThreshold: 6.0,
        noiseFloorWindow: 100,
        spectralRolloffThreshold: 0.85,
        spectralCentroidRange: (200.0, 8000.0)
    )
    #endif

    public init(
        threshold: Float = 0.3,
        chunkSize: Int = 512,
        sampleRate: Int = 16000,
        modelCacheDirectory: URL? = nil,
        debugMode: Bool = false,
        adaptiveThreshold: Bool = false,
        minThreshold: Float = 0.1,
        maxThreshold: Float = 0.7,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        enableSNRFiltering: Bool = true,
        minSNRThreshold: Float = 6.0,
        noiseFloorWindow: Int = 100,
        spectralRolloffThreshold: Float = 0.85,
        spectralCentroidRange: (min: Float, max: Float) = (200.0, 8000.0)
    ) {
        self.threshold = threshold
        self.chunkSize = chunkSize
        self.sampleRate = sampleRate
        self.modelCacheDirectory = modelCacheDirectory
        self.debugMode = debugMode
        self.adaptiveThreshold = adaptiveThreshold
        self.minThreshold = minThreshold
        self.maxThreshold = maxThreshold
        self.computeUnits = computeUnits
        self.enableSNRFiltering = enableSNRFiltering
        self.minSNRThreshold = minSNRThreshold
        self.noiseFloorWindow = noiseFloorWindow
        self.spectralRolloffThreshold = spectralRolloffThreshold
        self.spectralCentroidRange = spectralCentroidRange
    }
}