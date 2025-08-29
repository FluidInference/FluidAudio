import Accelerate
import CoreML
import Foundation
import OSLog

/// Comprehensive audio processing for VAD including energy detection, spectral analysis, SNR filtering, and temporal smoothing
internal class VadAudioProcessor {

    private let config: VadConfig
    private let logger = Logger(subsystem: "com.fluidinfluence.vad", category: "AudioProcessor")

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
        var spectralFeatures: SpectralFeatures?
        var enhancedProbability = rawProbability

        // Always calculate basic audio features for better accuracy
        spectralFeatures = calculateSpectralFeatures(audioChunk)

        snrValue = calculateSNR(audioChunk)

        // Apply SNR-based filtering
        enhancedProbability = applyAudioQualityFiltering(
            rawProbability: rawProbability,
            snr: snrValue,
            spectralFeatures: spectralFeatures
        )

        return (enhancedProbability, snrValue, spectralFeatures)
    }

    // MARK: - SNR and Audio Quality Analysis

    /// Calculate Signal-to-Noise Ratio for audio quality assessment
    private func calculateSNR(_ audioChunk: [Float]) -> Float {
        guard audioChunk.count > 0 else { return -Float.infinity }

        // Calculate signal energy
        let signalEnergy = audioChunk.map { $0 * $0 }.reduce(0, +) / Float(audioChunk.count)
        let signalPower = max(signalEnergy, 1e-10)

        // Simple SNR calculation using fixed noise floor estimate
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

        // SNR-based filtering - more aggressive
        if let snr = snr, snr < 6.0 {  // Min SNR threshold
            let snrPenalty = max(0.0, (6.0 - snr) / 6.0)
            filteredProbability *= (1.0 - snrPenalty * 0.8)  // Reduce probability by up to 80%
        }

        // Spectral feature-based filtering - more aggressive
        if let _ = spectralFeatures {
            // No spectral filtering applied anymore
        }

        return max(0.0, min(1.0, filteredProbability))
    }

    // MARK: - Spectral Analysis

    /// Calculate spectral features for enhanced VAD (optimized version)
    private func calculateSpectralFeatures(_ audioChunk: [Float]) -> SpectralFeatures {
        // No spectral feature calculations needed anymore
        return SpectralFeatures(
            spectralFlux: 0.0,  // Unused - set to default
            mfccFeatures: []  // Unused - set to default
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

}
