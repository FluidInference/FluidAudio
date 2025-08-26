import Accelerate
import CoreML
import Foundation
import OSLog

/// Comprehensive audio processing for VAD including energy detection, spectral analysis, SNR filtering, and temporal smoothing
internal class VadAudioProcessor {

    private let config: VadConfig
    private let logger = Logger(subsystem: "com.fluidinfluence.vad", category: "AudioProcessor")

    // State for audio processing
    private var probabilityWindow: [Float] = []
    private let windowSize = 5
    private var noiseFloorBuffer: [Float] = []
    private var currentNoiseFloor: Float = -60.0

    init(config: VadConfig) {
        self.config = config
    }

    /// Reset audio processor state
    func reset() {
        probabilityWindow.removeAll()
        noiseFloorBuffer.removeAll()
        currentNoiseFloor = -60.0
    }

    // MARK: - Enhanced VAD Processing

    /// Process raw ML probability with SNR filtering, spectral analysis, and temporal smoothing
    func processRawProbability(
        _ rawProbability: Float,
        audioChunk: [Float]
    ) -> (smoothedProbability: Float, snrValue: Float?, spectralFeatures: SpectralFeatures?) {

        var snrValue: Float?
        var spectralFeatures: SpectralFeatures?
        var enhancedProbability = rawProbability

        // Always calculate basic audio features for better accuracy
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

        // Apply temporal smoothing
        let smoothedProbability = applySmoothingFilter(enhancedProbability)

        return (smoothedProbability, snrValue, spectralFeatures)
    }

    /// Apply temporal smoothing filter to reduce noise
    private func applySmoothingFilter(_ probability: Float) -> Float {
        // Add to sliding window
        probabilityWindow.append(probability)
        if probabilityWindow.count > windowSize {
            probabilityWindow.removeFirst()
        }

        // Apply weighted moving average (more weight to recent values)
        guard !probabilityWindow.isEmpty else { return probability }

        let weights: [Float] = [0.1, 0.15, 0.2, 0.25, 0.3]  // Most recent gets highest weight
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
        if noiseFloorBuffer.count > 100 {  // Fixed window size
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
        if let snr = snr, snr < 6.0 {  // Min SNR threshold
            let snrPenalty = max(0.0, (6.0 - snr) / 6.0)
            filteredProbability *= (1.0 - snrPenalty * 0.8)  // Reduce probability by up to 80%
        }

        // Spectral feature-based filtering - more aggressive
        if let features = spectralFeatures {
            // Check if spectral centroid is in expected speech range
            let centroidInRange =
                features.spectralCentroid >= 200.0
                && features.spectralCentroid <= 8000.0

            if !centroidInRange {
                filteredProbability *= 0.5  // Reduce probability by 50%
            }

            // Check spectral rolloff (speech should have energy distributed across frequencies)
            if features.spectralRolloff > 0.85 {
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
            mfccFeatures: [],  // Unused - set to default
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
            let frequency = Float(i) * 16000.0 / Float(spectrum.count * 2)  // 16kHz sample rate
            weightedSum += frequency * magnitude
        }

        return weightedSum / totalEnergy
    }

    /// Calculate spectral rolloff (frequency below which X% of energy is contained)
    private func calculateSpectralRolloff(_ spectrum: [Float]) -> Float {
        guard !spectrum.isEmpty else { return 0.0 }

        let totalEnergy = spectrum.map { $0 * $0 }.reduce(0, +)
        guard totalEnergy > 0 else { return 0.0 }

        let rolloffThreshold = totalEnergy * 0.85
        var cumulativeEnergy: Float = 0.0

        for (i, magnitude) in spectrum.enumerated() {
            cumulativeEnergy += magnitude * magnitude
            if cumulativeEnergy >= rolloffThreshold {
                return Float(i) * 16000.0 / Float(spectrum.count * 2)
            }
        }

        return Float(spectrum.count - 1) * 16000.0 / Float(spectrum.count * 2)
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
            if (values[i] >= 0) != (values[i - 1] >= 0) {
                crossings += 1
            }
        }
        return Float(crossings) / Float(values.count - 1)
    }

    // MARK: - Fallback VAD Calculations

    /// Calculate VAD probability from RNN features (improved fallback method)
    func calculateVadProbability(from rnnFeatures: MLMultiArray) -> Float {
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

        // For RNN output shape [1, 4, 128], we want to focus on the last time step
        // which contains the most recent context
        let timeSteps = shape.count >= 3 ? shape[1] : 1
        let features = shape.count >= 3 ? shape[2] : shape[1]

        // Extract the last time step if we have temporal dimension
        var lastTimeStepValues: [Float] = []
        if timeSteps > 1 && features == 128 {
            // Extract last time step features [last 128 values]
            let startIdx = (timeSteps - 1) * features
            for i in startIdx..<min(startIdx + features, values.count) {
                lastTimeStepValues.append(values[i])
            }
        } else {
            lastTimeStepValues = values
        }

        // Use the RNN output more directly - it should already encode voice activity
        // The model output is designed to indicate voice activity probability
        let mean = lastTimeStepValues.reduce(0, +) / Float(lastTimeStepValues.count)
        let variance = lastTimeStepValues.map { pow($0 - mean, 2) }.reduce(0, +) / Float(lastTimeStepValues.count)
        let std = sqrt(variance)

        // Calculate key statistics from RNN output
        let maxValue = lastTimeStepValues.max() ?? 0.0
        let minValue = lastTimeStepValues.min() ?? 0.0
        let maxActivation = max(abs(maxValue), abs(minValue))
        let meanAbsActivation = lastTimeStepValues.map { abs($0) }.reduce(0, +) / Float(lastTimeStepValues.count)

        // Energy in the RNN features
        let energy = lastTimeStepValues.map { $0 * $0 }.reduce(0, +) / Float(lastTimeStepValues.count)
        let logEnergy = log(max(energy, 1e-10))

        // Look for strong positive activations which often indicate voice
        let positiveActivations = lastTimeStepValues.filter { $0 > 0 }
        let positiveRatio = Float(positiveActivations.count) / Float(lastTimeStepValues.count)
        let positiveMean =
            positiveActivations.isEmpty ? 0 : positiveActivations.reduce(0, +) / Float(positiveActivations.count)

        // RNN models often use specific neurons for voice detection
        // Check for strong activations in any neuron
        let strongActivations = lastTimeStepValues.filter { abs($0) > 0.5 }.count
        let strongRatio = Float(strongActivations) / Float(lastTimeStepValues.count)

        // Primary scoring based on RNN characteristics
        // Energy is key indicator - voice typically has higher energy than silence
        let energyScore = 1.0 / (1.0 + exp(-2.0 * (logEnergy + 10.0)))

        // Max activation indicates strong signals
        let maxScore = 1.0 / (1.0 + exp(-3.0 * (maxActivation - 0.3)))

        // Mean absolute activation
        let activationScore = 1.0 / (1.0 + exp(-5.0 * (meanAbsActivation - 0.1)))

        // Positive activation bias (voice often has more positive values)
        let positiveScore = positiveRatio * positiveMean * 2.0

        // Strong activation indicator
        let strongScore = min(1.0, strongRatio * 5.0)

        // Variance score (moderate variance is good for speech)
        let varianceScore = 1.0 / (1.0 + exp(-4.0 * (std - 0.15)))

        // Weighted combination
        let combinedScore =
            (energyScore * 0.25 + maxScore * 0.20 + activationScore * 0.20 + positiveScore * 0.15 + strongScore * 0.10
                + varianceScore * 0.10)

        // Final probability with smoother curve
        let probability = combinedScore

        // Ensure we don't have extreme values
        let clampedProbability = max(0.001, min(0.999, probability))

        return clampedProbability
    }

    /// Calculate number of peaks (local maxima)
    private func calculatePeakCount(_ values: [Float]) -> Int {
        guard values.count > 2 else { return 0 }

        var peaks = 0
        for i in 1..<(values.count - 1) {
            if values[i] > values[i - 1] && values[i] > values[i + 1] && abs(values[i]) > 0.1 {
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
        let middleHalf = Array(sortedValues[sortedValues.count / 4..<3 * sortedValues.count / 4])

        let topMean = topQuartile.reduce(0, +) / Float(topQuartile.count)
        let bottomMean = bottomQuartile.reduce(0, +) / Float(bottomQuartile.count)
        let middleMean = middleHalf.isEmpty ? 0 : middleHalf.reduce(0, +) / Float(middleHalf.count)

        // Speech has balanced distribution (strong middle component)
        let dynamicRange = topMean / max(bottomMean, 1e-10)
        let middleRatio = middleMean / max(topMean, 1e-10)

        // Speech-like patterns: moderate dynamic range + good middle energy
        let rangeScore = 1.0 / (1.0 + exp(-2.0 * (log(max(dynamicRange, 1.0)) - 3.0)))  // Peak around 20:1 ratio
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
