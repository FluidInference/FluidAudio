import Foundation
import Accelerate
import OSLog
import CoreML

/// Utility functions for model file management
internal class ModelFileChecker {
    private let logger = Logger(subsystem: "com.fluidinfluence.vad", category: "ModelChecker")
    
    /// Check for missing models and handle corrupted downloads
    func checkModelFiles(in directory: URL, modelNames: [String]) throws -> [String] {
        var missingModels: [String] = []
        
        for modelName in modelNames {
            let modelPath = directory.appendingPathComponent(modelName)
            
            if !FileManager.default.fileExists(atPath: modelPath.path) {
                missingModels.append(modelName)
            } else {
                // Check for corrupted or incomplete downloads
                do {
                    let attributes = try FileManager.default.attributesOfItem(atPath: modelPath.path)
                    if let fileSize = attributes[.size] as? Int64, fileSize < 1000 {
                        // Model folder exists but seems corrupted (too small)
                        logger.warning("Model \(modelName) appears corrupted (size: \(fileSize) bytes)")
                        missingModels.append(modelName)
                        // Remove corrupted model
                        try? FileManager.default.removeItem(at: modelPath)
                    }
                } catch {
                    logger.warning("Could not verify model \(modelName): \(error)")
                    missingModels.append(modelName)
                }
            }
        }
        
        return missingModels
    }
}

/// Comprehensive audio processing for VAD including energy detection, spectral analysis, SNR filtering, and temporal smoothing
internal class VADAudioProcessor {

    private let config: VADConfig
    private let logger = Logger(subsystem: "com.fluidinfluence.vad", category: "AudioProcessor")

    // State for audio processing
    private var probabilityWindow: [Float] = []
    private let windowSize = 5
    private var noiseFloorBuffer: [Float] = []
    private var currentNoiseFloor: Float = -60.0

    init(config: VADConfig) {
        self.config = config
    }

    /// Reset audio processor state
    func reset() {
        probabilityWindow.removeAll()
        noiseFloorBuffer.removeAll()
        currentNoiseFloor = -60.0
    }

    // MARK: - Basic Energy Detection

    /// Basic energy-based VAD (for testing without CoreML)
    func calculateBasicEnergyVAD(_ audioChunk: [Float]) -> Float {
        guard !audioChunk.isEmpty else { return 0.0 }

        // Simple RMS energy calculation
        let rmsEnergy = sqrt(audioChunk.map { $0 * $0 }.reduce(0, +) / Float(audioChunk.count))

        // Convert to dB
        let energyDB = 20.0 * log10(max(rmsEnergy, 1e-10))

        // Simple sigmoid mapping (energy to probability)
        // Typical speech energy: -40 to -10 dB
        // Noise/silence: -60 to -40 dB
        let normalizedEnergy = (energyDB + 50.0) / 40.0  // Map -50dB to 0, -10dB to 1
        let probability = 1.0 / (1.0 + exp(-6.0 * (normalizedEnergy - 0.5)))

        return max(0.0, min(1.0, probability))
    }

    // MARK: - Enhanced VAD Processing

    /// Process raw ML probability with SNR filtering and spectral analysis
    func processRawProbability(
        _ rawProbability: Float,
        audioChunk: [Float]
    ) -> (enhancedProbability: Float, snrValue: Float?, spectralFeatures: SpectralFeatures?) {

        var snrValue: Float?
        var spectralFeatures: SpectralFeatures?
        var enhancedProbability = rawProbability

        if config.enableSNRFiltering {
            // Calculate spectral features
            spectralFeatures = calculateSpectralFeatures(audioChunk)

            // Calculate SNR
            snrValue = calculateSNR(audioChunk)

            // Apply SNR-based filtering
            enhancedProbability = applyAudioQualityFiltering(
                rawProbability: rawProbability,
                snr: snrValue,
                spectralFeatures: spectralFeatures
            )
        }

        return (enhancedProbability, snrValue, spectralFeatures)
    }

    /// Apply temporal smoothing filter to reduce noise
    func applySmoothingFilter(_ probability: Float) -> Float {
        // Add to sliding window
        probabilityWindow.append(probability)
        if probabilityWindow.count > windowSize {
            probabilityWindow.removeFirst()
        }

        // Apply weighted moving average (more weight to recent values)
        guard !probabilityWindow.isEmpty else { return probability }

        let weights: [Float] = [0.1, 0.15, 0.2, 0.25, 0.3] // Most recent gets highest weight
        var weightedSum: Float = 0.0
        var totalWeight: Float = 0.0

        let startIndex = max(0, weights.count - probabilityWindow.count)
        for (i, prob) in probabilityWindow.enumerated() {
            let weightIndex = startIndex + i
            if weightIndex < weights.count {
                let weight = weights[weightIndex]
                weightedSum += prob * weight
                totalWeight += weight
            }
        }

        return totalWeight > 0 ? weightedSum / totalWeight : probability
    }

    // MARK: - SNR and Audio Quality Analysis

    /// Calculate Signal-to-Noise Ratio for audio quality assessment
    private func calculateSNR(_ audioChunk: [Float]) -> Float {
        guard audioChunk.count > 0 else { return -Float.infinity }

        // Calculate signal energy
        let signalEnergy = audioChunk.map { $0 * $0 }.reduce(0, +) / Float(audioChunk.count)
        let signalPower = max(signalEnergy, 1e-10)

        // Update noise floor estimate
        updateNoiseFloor(signalPower)

        // Calculate SNR in dB
        let snrLinear = signalPower / pow(10, currentNoiseFloor / 10.0)
        let snrDB = 10.0 * log10(max(snrLinear, 1e-10))

        return snrDB
    }

    /// Update noise floor estimation using minimum statistics
    private func updateNoiseFloor(_ currentPower: Float) {
        let powerDB = 10.0 * log10(max(currentPower, 1e-10))

        // Add to noise floor buffer
        noiseFloorBuffer.append(powerDB)

        // Keep only recent samples for noise floor estimation
        if noiseFloorBuffer.count > config.noiseFloorWindow {
            noiseFloorBuffer.removeFirst()
        }

        // Update noise floor using minimum statistics (conservative approach)
        if noiseFloorBuffer.count >= 10 {
            let sortedPowers = noiseFloorBuffer.sorted()
            let percentile10 = sortedPowers[sortedPowers.count / 10]  // 10th percentile

            // Smooth the noise floor update
            let alpha: Float = 0.1
            currentNoiseFloor = currentNoiseFloor * (1 - alpha) + percentile10 * alpha
        }
    }

    /// Apply audio quality filtering based on SNR and spectral features
    private func applyAudioQualityFiltering(
        rawProbability: Float,
        snr: Float?,
        spectralFeatures: SpectralFeatures?
    ) -> Float {
        var filteredProbability = rawProbability

        // SNR-based filtering - more aggressive
        if let snr = snr, snr < config.minSNRThreshold {
            let snrPenalty = max(0.0, (config.minSNRThreshold - snr) / config.minSNRThreshold)
            filteredProbability *= (1.0 - snrPenalty * 0.8)  // Reduce probability by up to 80%
        }

        // Spectral feature-based filtering - more aggressive
        if let features = spectralFeatures {
            // Check if spectral centroid is in expected speech range
            let centroidInRange = features.spectralCentroid >= config.spectralCentroidRange.min &&
                                 features.spectralCentroid <= config.spectralCentroidRange.max

            if !centroidInRange {
                filteredProbability *= 0.5  // Reduce probability by 50%
            }

            // Check spectral rolloff (speech should have energy distributed across frequencies)
            if features.spectralRolloff > config.spectralRolloffThreshold {
                filteredProbability *= 0.6  // Reduce probability by 40%
            }

            // Excessive zero crossings indicate noise
            if features.zeroCrossingRate > 0.3 {
                filteredProbability *= 0.4  // Reduce probability by 60%
            }

            // Low spectral entropy indicates tonal/musical content (not speech)
            if features.spectralEntropy < 0.3 {
                filteredProbability *= 0.3  // Reduce probability by 70%
            }
        }

        return max(0.0, min(1.0, filteredProbability))
    }

    // MARK: - Spectral Analysis

    /// Calculate spectral features for enhanced VAD (optimized version)
    private func calculateSpectralFeatures(_ audioChunk: [Float]) -> SpectralFeatures {
        let fftSize = min(256, audioChunk.count)  // Reduced FFT size for better performance
        let fftInput = Array(audioChunk.prefix(fftSize))

        // Compute FFT magnitude spectrum
        let spectrum = computeFFTMagnitude(fftInput)

        // Calculate spectral features (optimized calculations)
        let spectralCentroid = calculateSpectralCentroid(spectrum)
        let spectralRolloff = calculateSpectralRolloff(spectrum)
        let zeroCrossingRate = calculateZeroCrossingRate(fftInput)
        let spectralEntropy = calculateSpectralEntropy(spectrum)

        return SpectralFeatures(
            spectralCentroid: spectralCentroid,
            spectralRolloff: spectralRolloff,
            spectralFlux: 0.0,  // Unused - set to default
            mfccFeatures: [],   // Unused - set to default
            zeroCrossingRate: zeroCrossingRate,
            spectralEntropy: spectralEntropy
        )
    }

    /// Compute FFT magnitude spectrum using Accelerate framework
    private func computeFFTMagnitude(_ input: [Float]) -> [Float] {
        let n = input.count
        guard n > 0 else { return [] }

        // Find next power of 2 for FFT
        let log2n = Int(log2(Float(n)).rounded(.up))
        let fftSize = 1 << log2n

        // Prepare input with zero padding
        var paddedInput = input
        paddedInput.append(contentsOf: Array(repeating: 0.0, count: fftSize - n))

        // Setup FFT
        guard let fftSetup = vDSP_create_fftsetup(vDSP_Length(log2n), FFTRadix(kFFTRadix2)) else {
            return []
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Prepare complex buffer using proper pointer management
        var realInput = paddedInput
        var imagInput = Array(repeating: Float(0.0), count: fftSize)

        // Use withUnsafeMutableBufferPointer for proper pointer management
        return realInput.withUnsafeMutableBufferPointer { realPtr in
            imagInput.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                // Perform FFT
                vDSP_fft_zip(fftSetup, &splitComplex, 1, vDSP_Length(log2n), FFTDirection(FFT_FORWARD))

                // Compute magnitude spectrum
                var magnitudes = Array(repeating: Float(0.0), count: fftSize / 2)
                vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(fftSize / 2))

                // Take square root to get magnitude (not power)
                for i in 0..<magnitudes.count {
                    magnitudes[i] = sqrt(magnitudes[i])
                }

                return magnitudes
            }
        }
    }

    /// Calculate spectral centroid (center of mass of spectrum)
    private func calculateSpectralCentroid(_ spectrum: [Float]) -> Float {
        guard !spectrum.isEmpty else { return 0.0 }

        let totalEnergy = spectrum.reduce(0, +)
        guard totalEnergy > 0 else { return 0.0 }

        var weightedSum: Float = 0.0
        for (i, magnitude) in spectrum.enumerated() {
            let frequency = Float(i) * Float(config.sampleRate) / Float(spectrum.count * 2)
            weightedSum += frequency * magnitude
        }

        return weightedSum / totalEnergy
    }

    /// Calculate spectral rolloff (frequency below which X% of energy is contained)
    private func calculateSpectralRolloff(_ spectrum: [Float]) -> Float {
        guard !spectrum.isEmpty else { return 0.0 }

        let totalEnergy = spectrum.map { $0 * $0 }.reduce(0, +)
        guard totalEnergy > 0 else { return 0.0 }

        let rolloffThreshold = totalEnergy * config.spectralRolloffThreshold
        var cumulativeEnergy: Float = 0.0

        for (i, magnitude) in spectrum.enumerated() {
            cumulativeEnergy += magnitude * magnitude
            if cumulativeEnergy >= rolloffThreshold {
                return Float(i) * Float(config.sampleRate) / Float(spectrum.count * 2)
            }
        }

        return Float(spectrum.count - 1) * Float(config.sampleRate) / Float(spectrum.count * 2)
    }

    /// Calculate spectral entropy (measure of spectral complexity)
    private func calculateSpectralEntropy(_ spectrum: [Float]) -> Float {
        guard !spectrum.isEmpty else { return 0.0 }

        let totalEnergy = spectrum.map { $0 * $0 }.reduce(0, +)
        guard totalEnergy > 0 else { return 0.0 }

        // Normalize to probability distribution
        let probabilities = spectrum.map { ($0 * $0) / totalEnergy }

        // Calculate entropy
        var entropy: Float = 0.0
        for p in probabilities {
            if p > 0 {
                entropy -= p * log(p)
            }
        }

        // Normalize entropy
        return entropy / log(Float(spectrum.count))
    }

    /// Calculate zero crossing rate for VAD
    private func calculateZeroCrossingRate(_ values: [Float]) -> Float {
        guard values.count > 1 else { return 0.0 }

        var crossings = 0
        for i in 1..<values.count {
            if (values[i] >= 0) != (values[i-1] >= 0) {
                crossings += 1
            }
        }
        return Float(crossings) / Float(values.count - 1)
    }

    // MARK: - Fallback VAD Calculations

    /// Calculate VAD probability from RNN features (improved fallback method)
    func calculateVADProbability(from rnnFeatures: MLMultiArray) -> Float {
        let shape = rnnFeatures.shape.map { $0.intValue }
        guard shape.count >= 2 else {
            return 0.0
        }

        let totalElements = rnnFeatures.count
        guard totalElements > 0 else { return 0.0 }

        // Extract values from MLMultiArray
        var values: [Float] = []
        for i in 0..<totalElements {
            values.append(rnnFeatures[i].floatValue)
        }

        // Advanced feature extraction optimized for speech vs non-speech discrimination
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Float(values.count)
        let std = sqrt(variance)

        // Energy-based features (fundamental for VAD)
        let energy = values.map { $0 * $0 }.reduce(0, +) / Float(values.count)
        let logEnergy = log(max(energy, 1e-10))

        // Temporal features
        let zeroCrossingRate = calculateZeroCrossingRate(values)
        let peakCount = calculatePeakCount(values)
        let _ = Float(peakCount) / Float(values.count)  // peakRatio for future use

        // Speech pattern analysis
        let speechIndicator = calculateSpeechIndicator(values)

        // Noise detection features
        let isHighFrequencyNoise = zeroCrossingRate > 0.3  // Too many zero crossings = noise
        let isVeryHighEnergy = logEnergy > -0.5           // Extremely high energy = often noise
        let isTooManyPeaks = peakCount > 200              // Too many peaks = noise

        // Balanced feature weighting for better speech/noise discrimination
        let energyScore = max(0.0, tanh(logEnergy + 5.0) * 0.3)   // Moderate energy threshold
        let varianceScore = tanh(std * 2.0) * 0.15                // Moderate variance importance
        let speechScore = speechIndicator * 0.4                   // Strong weight for speech patterns
        let crossingScore = tanh(zeroCrossingRate * 4.0) * 0.15   // Moderate crossing rate

        // Refined noise detection with moderate penalties
        var penalties: Float = 0.0
        if isHighFrequencyNoise { penalties -= 0.2 }         // Moderate penalty for high-frequency noise
        if isVeryHighEnergy { penalties -= 0.15 }            // Moderate penalty for very high energy
        if isTooManyPeaks { penalties -= 0.15 }              // Moderate penalty for too many peaks
        if speechIndicator < 0.3 { penalties -= 0.2 }       // Moderate penalty for non-speech patterns

        // Moderate additional noise pattern detection
        if std < 0.005 { penalties -= 0.15 }                 // Very little variation = likely noise
        if logEnergy > 3.0 { penalties -= 0.1 }              // Very high energy = potentially noise
        if zeroCrossingRate > 0.5 { penalties -= 0.2 }      // Excessive zero crossings = noise

        let baseScore = energyScore + varianceScore + speechScore + crossingScore
        let combinedScore = max(0.0, baseScore + penalties)  // Ensure non-negative

        // Balanced sigmoid for reasonable speech detection
        let probability = 1.0 / (1.0 + exp(-8.0 * (combinedScore - 0.4)))

        return max(0.0, min(1.0, probability))
    }

    /// Calculate number of peaks (local maxima)
    private func calculatePeakCount(_ values: [Float]) -> Int {
        guard values.count > 2 else { return 0 }

        var peaks = 0
        for i in 1..<(values.count - 1) {
            if values[i] > values[i-1] && values[i] > values[i+1] && abs(values[i]) > 0.1 {
                peaks += 1
            }
        }
        return peaks
    }

    /// Calculate speech-like pattern indicator
    private func calculateSpeechIndicator(_ values: [Float]) -> Float {
        // Look for patterns that are characteristic of speech vs noise/music
        let absValues = values.map { abs($0) }
        let sortedValues = absValues.sorted(by: >)

        // Speech typically has moderate dynamic range (not too flat, not too extreme)
        let topQuartile = Array(sortedValues.prefix(max(1, sortedValues.count / 4)))
        let bottomQuartile = Array(sortedValues.suffix(max(1, sortedValues.count / 4)))
        let middleHalf = Array(sortedValues[sortedValues.count/4..<3*sortedValues.count/4])

        let topMean = topQuartile.reduce(0, +) / Float(topQuartile.count)
        let bottomMean = bottomQuartile.reduce(0, +) / Float(bottomQuartile.count)
        let middleMean = middleHalf.isEmpty ? 0 : middleHalf.reduce(0, +) / Float(middleHalf.count)

        // Speech has balanced distribution (strong middle component)
        let dynamicRange = topMean / max(bottomMean, 1e-10)
        let middleRatio = middleMean / max(topMean, 1e-10)

        // Speech-like patterns: moderate dynamic range + good middle energy
        let rangeScore = 1.0 / (1.0 + exp(-2.0 * (log(max(dynamicRange, 1.0)) - 3.0))) // Peak around 20:1 ratio
        let middleScore = middleRatio  // Higher middle energy is speech-like

        // Simple consistency check - speech has variance
        let mean = absValues.reduce(0, +) / Float(absValues.count)
        let overallVariance = absValues.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(absValues.count)
        let consistency = min(1.0, sqrt(overallVariance) * 5.0)

        // Combine indicators (speech needs all three)
        let speechLikeness = (rangeScore * 0.4 + middleScore * 0.4 + consistency * 0.2)

        return max(0.0, min(1.0, speechLikeness))
    }
}
