import Accelerate
import CoreML
import Foundation
import OSLog

/// Comprehensive audio processing for VAD including energy detection, spectral analysis, SNR filtering, and temporal smoothing
internal class VadAudioProcessor {

    private let config: VadConfig
    private let logger = AppLogger(category: "AudioProcessor")

    init(config: VadConfig) {
        self.config = config
    }

    // MARK: - Enhanced VAD Processing

    /// Process raw ML probability with SNR filtering, spectral analysis, and temporal smoothing
    func processRawProbability(
        _ rawProbability: Float,
        audioChunk: [Float]
    ) -> (smoothedProbability: Float, snrValue: Float?, spectralFeatures: SpectralFeatures?) {

        var snrValue: Float?
        var enhancedProbability = rawProbability

        snrValue = calculateSNR(audioChunk)

        // Apply SNR-based filtering
        enhancedProbability = applyAudioQualityFiltering(
            rawProbability: rawProbability,
            snr: snrValue,
            spectralFeatures: nil
        )

        return (enhancedProbability, snrValue, nil)
    }

    // MARK: - SNR and Audio Quality Analysis

    /// Calculate Signal-to-Noise Ratio for audio quality assessment
    private func calculateSNR(_ audioChunk: [Float]) -> Float {
        guard audioChunk.count > 0 else { return -Float.infinity }

        // Calculate signal energy
        let signalEnergy = audioChunk.map { $0 * $0 }.reduce(0, +) / Float(audioChunk.count)
        let signalPower = max(signalEnergy, 1e-10)

        // Simple SNR calculation using fixed noise floor estimate
        // -60 dB represents typical ambient noise in a quiet room (equivalent to ~0.001 linear amplitude)
        // This is a standard reference level used in audio processing for noise floor estimation
        let fixedNoiseFloor: Float = -60.0  // dB
        let snrLinear = signalPower / pow(10, fixedNoiseFloor / 10.0)
        let snrDB = 10.0 * log10(max(snrLinear, 1e-10))

        return snrDB
    }

    /// Apply audio quality filtering based on SNR and spectral features
    private func applyAudioQualityFiltering(
        rawProbability: Float,
        snr: Float?,
        spectralFeatures: SpectralFeatures?
    ) -> Float {
        var filteredProbability = rawProbability

        // 6.0 dB is the minimum SNR for intelligible speech (speech is ~4x louder than noise)
        if let snr = snr, snr < 6.0 {  // Min SNR threshold
            let snrPenalty = max(0.0, (6.0 - snr) / 6.0)
            filteredProbability *= (1.0 - snrPenalty * 0.8)  // Reduce probability by up to 80%
        }

        return max(0.0, min(1.0, filteredProbability))
    }

}
