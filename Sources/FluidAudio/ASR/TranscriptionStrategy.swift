//
//  TranscriptionStrategy.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

/// Defines the transcription strategy to use
enum TranscriptionStrategy {
    case single
    case chunked
    case streaming
}

/// Transcription request encapsulating all parameters
struct TranscriptionRequest {
    let audioSamples: [Float]
    let strategy: TranscriptionStrategy
    let enableDebug: Bool
    let chunkSize = 160_000
    
    init(audioSamples: [Float], enableDebug: Bool = false) {
        self.audioSamples = audioSamples
        self.enableDebug = enableDebug
        self.strategy = audioSamples.count > 160_000 ? .chunked : .single
    }
}

/// Unified transcription interface
extension AsrManager {
    
    /// Main public transcription entry point
    public func transcribeUnified(_ audioSamples: [Float]) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard audioSamples.count >= 16_000 else { throw ASRError.invalidAudioData }
        
        let request = TranscriptionRequest(audioSamples: audioSamples, enableDebug: config.enableDebug)
        return try await executeTranscription(request)
    }
    
    /// Execute transcription based on strategy
    private func executeTranscription(_ request: TranscriptionRequest) async throws -> ASRResult {
        let startTime = Date()
        
        switch request.strategy {
        case .single:
            return try await executeSingleTranscription(request, startTime: startTime)
            
        case .chunked:
            return try await executeChunkedTranscription(request, startTime: startTime)
            
        case .streaming:
            throw ASRError.processingFailed("Streaming not yet implemented")
        }
    }
    
    /// Execute single chunk transcription
    private func executeSingleTranscription(_ request: TranscriptionRequest, startTime: Date) async throws -> ASRResult {
        let paddedAudio = padAudioIfNeeded(request.audioSamples, targetLength: 160_000)
        let (tokenIds, encoderSequenceLength) = try await transcriptionCore(paddedAudio, enableDebug: request.enableDebug)
        
        return processTranscriptionResult(
            tokenIds: tokenIds,
            encoderSequenceLength: encoderSequenceLength,
            audioSamples: request.audioSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }
    
    /// Execute chunked transcription
    private func executeChunkedTranscription(_ request: TranscriptionRequest, startTime: Date) async throws -> ASRResult {
        return try await ChunkProcessor(
            audioSamples: request.audioSamples,
            chunkSize: request.chunkSize,
            enableDebug: request.enableDebug
        ).process(using: self, startTime: startTime)
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
            let chunk = try await processChunk(at: position, chunkIndex: chunkIndex, using: manager)
            allTexts.append(chunk.text)
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
    
    private func processChunk(at position: Int, chunkIndex: Int, using manager: AsrManager) async throws -> (text: String, processingTime: TimeInterval) {
        let endPosition = min(position + chunkSize, audioSamples.count)
        let chunkSamples = Array(audioSamples[position..<endPosition])
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: chunkSize)
        
        if chunkIndex == 0 {
            try await manager.initializeDecoderState()
        }
        
        let chunkStartTime = Date()
        let (tokenIds, encoderSequenceLength) = try await manager.transcriptionCore(paddedChunk, enableDebug: false)
        
        let chunkResult = manager.processTranscriptionResult(
            tokenIds: tokenIds,
            encoderSequenceLength: encoderSequenceLength,
            audioSamples: paddedChunk,
            processingTime: Date().timeIntervalSince(chunkStartTime)
        )
        
        return (chunkResult.text, chunkResult.processingTime)
    }
}