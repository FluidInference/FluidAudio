//
//  AsrTranscription.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import CoreML
import Foundation
import OSLog


/// ASR transcription functionality including chunking logic and ML inference pipeline
extension AsrManager {
    
    /// Main public transcription entry point
    public func transcribeUnified(_ audioSamples: [Float]) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard audioSamples.count >= 16_000 else { throw ASRError.invalidAudioData }
        
        let startTime = Date()
        
        // Single chunk processing for audio <= 10 seconds
        if audioSamples.count <= 160_000 {
            let paddedAudio = padAudioIfNeeded(audioSamples, targetLength: 160_000)
            let (tokenIds, encoderSequenceLength) = try await executeMLInference(paddedAudio, enableDebug: config.enableDebug)
            
            return processTranscriptionResult(
                tokenIds: tokenIds,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }
        
        // Chunked processing for audio > 10 seconds
        return try await ChunkProcessor(
            audioSamples: audioSamples,
            chunkSize: 160_000,
            enableDebug: config.enableDebug
        ).process(using: self, startTime: startTime)
    }
    
    /// Execute ML inference pipeline: Audio → Mel-Spectrogram → Encoder → Decoder → Tokens
    internal func executeMLInference(
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

/// Handles chunked audio processing
private struct ChunkProcessor {
    let audioSamples: [Float]
    let chunkSize: Int
    let enableDebug: Bool
    
    func process(using manager: AsrManager, startTime: Date) async throws -> ASRResult {
        var allTexts: [String] = []
        let audioLength = Double(audioSamples.count) / 16000.0
        
        var position = 0
        var chunkIndex = 0
        
        while position < audioSamples.count {
            let text = try await processChunk(at: position, chunkIndex: chunkIndex, using: manager)
            allTexts.append(text)
            position += chunkSize
            chunkIndex += 1
        }
        
        return ASRResult(
            text: allTexts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines),
            confidence: 1.0,
            duration: audioLength,
            processingTime: Date().timeIntervalSince(startTime),
            tokenTimings: nil
        )
    }
    
    private func processChunk(at position: Int, chunkIndex: Int, using manager: AsrManager) async throws -> String {
        let endPosition = min(position + chunkSize, audioSamples.count)
        let chunkSamples = Array(audioSamples[position..<endPosition])
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: chunkSize)
        
        if chunkIndex == 0 {
            try await manager.initializeDecoderState()
        }
        
        let (tokenIds, _) = try await manager.executeMLInference(paddedChunk, enableDebug: false)
        let (text, _) = manager.convertTokensWithExistingTimings(tokenIds, timings: [])
        
        return text
    }
}