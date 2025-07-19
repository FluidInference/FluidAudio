//
//  TranscriptionCore.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import CoreML
import Foundation
import OSLog

/// Core transcription functionality shared across all transcribe methods
extension AsrManager {
    
    /// Core transcription pipeline: Audio → Mel-Spectrogram → Encoder → Decoder → Text
    internal func transcriptionCore(
        _ paddedAudio: [Float],
        enableDebug: Bool = false
    ) async throws -> (tokenIds: [Int], encoderSequenceLength: Int) {
        
        // Step 1: Mel-Spectrogram
        let melSpectrogramInput = try prepareMelSpectrogramInput(paddedAudio)
        
        if enableDebug {
            logger.debug("🔍 DEBUG: Audio processing:")
            logger.debug("   - Audio input length: \(paddedAudio.count) samples")
            logger.debug("   - Audio duration: \(String(format: "%.2f", Float(paddedAudio.count) / 16000.0)) seconds")
        }
        
        guard let melSpectrogramOutput = try melSpectrogramModel?.prediction(
            from: melSpectrogramInput,
            options: predictionOptions ?? MLPredictionOptions()
        ) else {
            throw ASRError.processingFailed("Mel-spectrogram model failed")
        }
        
        if enableDebug {
            debugFeatureProvider("Mel-spectrogram", melSpectrogramOutput)
        }
        
        // Step 2: Encoder
        let encoderInput = try prepareEncoderInput(melSpectrogramOutput)
        guard let encoderOutput = try encoderModel?.prediction(
            from: encoderInput,
            options: predictionOptions ?? MLPredictionOptions()
        ) else {
            throw ASRError.processingFailed("Encoder model failed")
        }
        
        if enableDebug {
            debugFeatureProvider("Encoder", encoderOutput)
        }
        
        // Step 3: Extract encoder outputs
        guard let rawEncoderOutput = encoderOutput.featureValue(for: "encoder_output")?.multiArrayValue,
              let encoderLength = encoderOutput.featureValue(for: "encoder_output_length")?.multiArrayValue else {
            throw ASRError.processingFailed("Invalid encoder output")
        }
        
        let encoderHiddenStates = try transposeEncoderOutput(rawEncoderOutput)
        let encoderSequenceLength = encoderLength[0].intValue
        
        if enableDebug {
            logger.debug("🔍 DEBUG: Encoder processing:")
            logger.debug("   - Raw encoder output shape: \(rawEncoderOutput.shape)")
            logger.debug("   - Encoder sequence length: \(encoderSequenceLength)")
            logger.debug("   - Encoder hidden states shape after transpose: \(encoderHiddenStates.shape)")
        }
        
        // Step 4: Decode (TDT is always enabled)
        let tokenIds = try await tdtDecode(
            encoderOutput: encoderHiddenStates,
            encoderSequenceLength: encoderSequenceLength,
            originalAudioSamples: paddedAudio
        )
        
        return (tokenIds, encoderSequenceLength)
    }
    
    /// Process transcription result into final ASRResult
    internal func processTranscriptionResult(
        tokenIds: [Int],
        encoderSequenceLength: Int,
        audioSamples: [Float],
        processingTime: TimeInterval,
        tokenTimings: [TokenTiming] = []
    ) -> ASRResult {
        
        // Convert tokens to text
        let (text, finalTimings) = convertTokensWithExistingTimings(tokenIds, timings: tokenTimings)
        
        // Apply post-processing if enabled
        let finalText = config.enableAdvancedPostProcessing ?
            applyAdvancedPostProcessing(text, tokenTimings: finalTimings) :
            text
        
        let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)
        
        // Log warning if empty transcription for longer audio
        if finalText.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty && duration > 1.0 {
            logger.warning("⚠️ Empty transcription for \(String(format: "%.1f", duration))s audio (tokens: \(tokenIds.count))")
            
            if ProcessInfo.processInfo.environment["CI"] != nil {
                logger.debug("🔍 CI Debug - Empty transcription:")
                logger.debug("   Audio duration: \(String(format: "%.1f", duration))s")
                logger.debug("   Token IDs count: \(tokenIds.count)")
                logger.debug("   Token IDs: \(tokenIds.prefix(10))...")
                logger.debug("   Encoder sequence length: \(encoderSequenceLength)")
                logger.debug("   Models loaded: mel=\(self.melSpectrogramModel != nil), encoder=\(self.encoderModel != nil), decoder=\(self.decoderModel != nil), joint=\(self.jointModel != nil)")
            }
        }
        
        return ASRResult(
            text: finalText,
            confidence: 1.0,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: finalTimings
        )
    }
    
    /// Debug helper for feature providers
    private func debugFeatureProvider(_ name: String, _ provider: MLFeatureProvider) {
        logger.debug("🔍 DEBUG: \(name) output features:")
        for featureName in provider.featureNames {
            if let value = provider.featureValue(for: featureName) {
                if let array = value.multiArrayValue {
                    logger.debug("   - '\(featureName)': shape=\(array.shape)")
                    if featureName.contains("length") {
                        logger.debug("   - \(featureName) value: \(array[0])")
                    }
                }
            }
        }
    }
    
    /// Pad audio to required length
    internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else {
            return audioSamples
        }
        
        var paddedAudio = audioSamples
        let padding = Array(repeating: Float(0.0), count: targetLength - audioSamples.count)
        paddedAudio.append(contentsOf: padding)
        return paddedAudio
    }
}