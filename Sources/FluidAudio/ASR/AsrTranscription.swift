//
//  AsrTranscription.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import CoreML
import Foundation
import OSLog


extension AsrManager {
    
    public func transcribeUnified(_ audioSamples: [Float]) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard audioSamples.count >= 16_000 else { throw ASRError.invalidAudioData }
        
        let startTime = Date()
        
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
        
        return try await ChunkProcessor(
            audioSamples: audioSamples,
            chunkSize: 160_000,
            enableDebug: config.enableDebug
        ).process(using: self, startTime: startTime)
    }
    
    internal func transcribeUnifiedWithState(_ audioSamples: [Float], decoderState: inout DecoderState) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard audioSamples.count >= 16_000 else { throw ASRError.invalidAudioData }
        
        let startTime = Date()
        
        if audioSamples.count <= 160_000 {
            let paddedAudio = padAudioIfNeeded(audioSamples, targetLength: 160_000)
            let (tokenIds, encoderSequenceLength) = try await executeMLInferenceWithState(
                paddedAudio, 
                enableDebug: config.enableDebug,
                decoderState: &decoderState
            )
            
            return processTranscriptionResult(
                tokenIds: tokenIds,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }
        
        let result = try await ChunkProcessor(
            audioSamples: audioSamples,
            chunkSize: 160_000,
            enableDebug: config.enableDebug
        ).process(using: self, startTime: startTime)
        return result
    }
    
    internal func executeMLInference(
        _ paddedAudio: [Float],
        enableDebug: Bool = false
    ) async throws -> (tokenIds: [Int], encoderSequenceLength: Int) {
        
        let melSpectrogramInput = try prepareMelSpectrogramInput(paddedAudio)
        
        guard let melSpectrogramOutput = try melSpectrogramModel?.prediction(
            from: melSpectrogramInput,
            options: MLPredictionOptions()
        ) else {
            throw ASRError.processingFailed("Mel-spectrogram model failed")
        }
        
        let encoderInput = try prepareEncoderInput(melSpectrogramOutput)
        guard let encoderOutput = try encoderModel?.prediction(
            from: encoderInput,
            options: MLPredictionOptions()
        ) else {
            throw ASRError.processingFailed("Encoder model failed")
        }
        
        let rawEncoderOutput = try extractFeatureValue(from: encoderOutput, key: "encoder_output", errorMessage: "Invalid encoder output")
        let encoderLength = try extractFeatureValue(from: encoderOutput, key: "encoder_output_length", errorMessage: "Invalid encoder output length")
        
        let encoderHiddenStates = try transposeEncoderOutput(rawEncoderOutput)
        let encoderSequenceLength = encoderLength[0].intValue
        
        var tempDecoderState = DecoderState()
        let tokenIds = try await tdtDecode(
            encoderOutput: encoderHiddenStates,
            encoderSequenceLength: encoderSequenceLength,
            originalAudioSamples: paddedAudio,
            decoderState: &tempDecoderState
        )
        
        return (tokenIds, encoderSequenceLength)
    }
    
    internal func executeMLInferenceWithState(
        _ paddedAudio: [Float],
        enableDebug: Bool = false,
        decoderState: inout DecoderState
    ) async throws -> (tokenIds: [Int], encoderSequenceLength: Int) {
        
        let melSpectrogramInput = try prepareMelSpectrogramInput(paddedAudio)
        
        guard let melSpectrogramOutput = try melSpectrogramModel?.prediction(
            from: melSpectrogramInput,
            options: MLPredictionOptions()
        ) else {
            throw ASRError.processingFailed("Mel-spectrogram model failed")
        }
        
        let encoderInput = try prepareEncoderInput(melSpectrogramOutput)
        guard let encoderOutput = try encoderModel?.prediction(
            from: encoderInput,
            options: MLPredictionOptions()
        ) else {
            throw ASRError.processingFailed("Encoder model failed")
        }
        
        let rawEncoderOutput = try extractFeatureValue(from: encoderOutput, key: "encoder_output", errorMessage: "Invalid encoder output")
        let encoderLength = try extractFeatureValue(from: encoderOutput, key: "encoder_output_length", errorMessage: "Invalid encoder output length")
        
        let encoderHiddenStates = try transposeEncoderOutput(rawEncoderOutput)
        let encoderSequenceLength = encoderLength[0].intValue
        
        let tokenIds = try await tdtDecode(
            encoderOutput: encoderHiddenStates,
            encoderSequenceLength: encoderSequenceLength,
            originalAudioSamples: paddedAudio,
            decoderState: &decoderState
        )
        
        return (tokenIds, encoderSequenceLength)
    }
    
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
    
    internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
    }
}

private struct ChunkProcessor {
    let audioSamples: [Float]
    let chunkSize: Int
    let enableDebug: Bool
    
    func process(using manager: AsrManager, startTime: Date) async throws -> ASRResult {
        var allTexts: [String] = []
        let audioLength = Double(audioSamples.count) / 16000.0
        
        var position = 0
        var chunkIndex = 0
        var decoderState = DecoderState()
        
        while position < audioSamples.count {
            let text = try await processChunk(at: position, chunkIndex: chunkIndex, using: manager, decoderState: &decoderState)
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
    
    private func processChunk(at position: Int, chunkIndex: Int, using manager: AsrManager, decoderState: inout DecoderState) async throws -> String {
        let endPosition = min(position + chunkSize, audioSamples.count)
        let chunkSamples = Array(audioSamples[position..<endPosition])
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: chunkSize)
        
        
        let (tokenIds, _) = try await manager.executeMLInferenceWithState(paddedChunk, enableDebug: false, decoderState: &decoderState)
        let (text, _) = manager.convertTokensWithExistingTimings(tokenIds, timings: [])
        
        return text
    }
}