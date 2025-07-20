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
        
        let melSpectrogramInput = try prepareMelSpectrogramInput(paddedAudio)
        
        guard let melSpectrogramOutput = try melSpectrogramModel?.prediction(
            from: melSpectrogramInput,
            options: predictionOptions ?? MLPredictionOptions()
        ) else {
            throw ASRError.processingFailed("Mel-spectrogram model failed")
        }
        
        let encoderInput = try prepareEncoderInput(melSpectrogramOutput)
        guard let encoderOutput = try encoderModel?.prediction(
            from: encoderInput,
            options: predictionOptions ?? MLPredictionOptions()
        ) else {
            throw ASRError.processingFailed("Encoder model failed")
        }
        
        guard let rawEncoderOutput = encoderOutput.featureValue(for: "encoder_output")?.multiArrayValue,
              let encoderLength = encoderOutput.featureValue(for: "encoder_output_length")?.multiArrayValue else {
            throw ASRError.processingFailed("Invalid encoder output")
        }
        
        let encoderHiddenStates = try transposeEncoderOutput(rawEncoderOutput)
        let encoderSequenceLength = encoderLength[0].intValue
        
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
        
        let (text, finalTimings) = convertTokensWithExistingTimings(tokenIds, timings: tokenTimings)
        let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)
        
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && duration > 1.0 {
            logger.warning("⚠️ Empty transcription for \(String(format: "%.1f", duration))s audio (tokens: \(tokenIds.count))")
        }
        
        return ASRResult(
            text: text,
            confidence: 1.0,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: finalTimings
        )
    }
    
    /// Pad audio to required length
    internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
    }
}