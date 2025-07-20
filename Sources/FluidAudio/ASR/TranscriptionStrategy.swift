//
//  TranscriptionStrategy.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

/// Defines the transcription strategy to use
enum TranscriptionStrategy {
    case single           // Single chunk transcription
    case chunked         // Multi-chunk with overlap
    case streaming       // Future: real-time streaming
}

/// Transcription request encapsulating all parameters
struct TranscriptionRequest {
    let audioSamples: [Float]
    let strategy: TranscriptionStrategy
    let enableDebug: Bool
    
    var chunkSize: Int = 160_000  // 10 seconds at 16kHz (CoreML constraint)
    
    init(audioSamples: [Float], enableDebug: Bool = false) {
        self.audioSamples = audioSamples
        self.enableDebug = enableDebug
        
        // CoreML model has fixed constraints: max 160,000 samples (10 seconds)
        // and encoder output must be shape (1, 128, 1001)
        if audioSamples.count > 160_000 {
            self.strategy = .chunked
            self.chunkSize = 160_000  // 10 seconds at 16kHz
        } else {
            self.strategy = .single
        }
    }
}

/// Unified transcription interface
extension AsrManager {
    
    /// Main public transcription entry point - simplified
    public func transcribeUnified(_ audioSamples: [Float]) async throws -> ASRResult {
        guard isAvailable else {
            throw ASRError.notInitialized
        }
        
        // Validate minimum audio length
        guard audioSamples.count >= 16_000 else {
            throw ASRError.invalidAudioData
        }
        
        let request = TranscriptionRequest(
            audioSamples: audioSamples,
            enableDebug: config.enableDebug
        )
        
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
        // Pad audio to 10 seconds - CoreML model requires fixed shape
        let paddedAudio = padAudioIfNeeded(request.audioSamples, targetLength: 160_000)
        
        // Use shared transcription core
        let (tokenIds, encoderSequenceLength) = try await transcriptionCore(
            paddedAudio,
            enableDebug: request.enableDebug
        )
        
        let processingTime = Date().timeIntervalSince(startTime)
        
        // Process result
        return processTranscriptionResult(
            tokenIds: tokenIds,
            encoderSequenceLength: encoderSequenceLength,
            audioSamples: request.audioSamples,
            processingTime: processingTime
        )
    }
    
    /// Execute chunked transcription for audio longer than 24 minutes
    private func executeChunkedTranscription(_ request: TranscriptionRequest, startTime: Date) async throws -> ASRResult {
        let chunkProcessor = ChunkProcessor(
            audioSamples: request.audioSamples,
            chunkSize: request.chunkSize,
            enableDebug: request.enableDebug
        )
        
        return try await chunkProcessor.process(using: self, startTime: startTime)
    }
}

/// Handles chunked audio processing
private struct ChunkProcessor {
    let audioSamples: [Float]
    let chunkSize: Int
    let enableDebug: Bool
    
    private var stride: Int { chunkSize } // No overlap - just sequential chunks
    
    func process(using manager: AsrManager, startTime: Date) async throws -> ASRResult {
        var allTexts: [String] = []
        var totalProcessingTime: TimeInterval = 0
        let audioLength = Double(audioSamples.count) / Double(manager.config.sampleRate)
        
        if enableDebug {
            manager.logger.info("🔄 Chunking audio: \(String(format: "%.1f", audioLength))s into \(String(format: "%.1f", Double(chunkSize)/16000.0))s chunks")
        }
        
        var position = 0
        var chunkIndex = 0
        
        // Process chunks sequentially without overlap
        while position < audioSamples.count {
            let chunk = try await processChunk(
                at: position,
                chunkIndex: chunkIndex,
                using: manager
            )
            
            allTexts.append(chunk.text)
            totalProcessingTime += chunk.processingTime
            
            position += stride
            chunkIndex += 1
        }
        
        // Combine results
        let finalText = allTexts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanedText = finalText
        
        let overallProcessingTime = Date().timeIntervalSince(startTime)
        
        if enableDebug {
            let rtfx = audioLength / overallProcessingTime
            manager.logger.info("✅ Chunked transcription complete: \(chunkIndex) chunks, RTFx: \(String(format: "%.1f", rtfx))x")
        }
        
        return ASRResult(
            text: cleanedText,
            confidence: 1.0,
            duration: audioLength,
            processingTime: overallProcessingTime,
            tokenTimings: nil
        )
    }
    
    private func processChunk(
        at position: Int,
        chunkIndex: Int,
        using manager: AsrManager
    ) async throws -> (text: String, processingTime: TimeInterval) {
        
        let endPosition = min(position + chunkSize, audioSamples.count)
        let chunkSamples = Array(audioSamples[position..<endPosition])
        // Pad chunk to 10 seconds - CoreML requires fixed shape
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: chunkSize)
        
        if enableDebug && chunkIndex == 0 {
            let _ = Double(chunkSamples.count) / 16000.0
            manager.logger.info("   📄 Chunk \(chunkIndex): \(String(format: "%.1f", Double(position)/16000.0))s - \(String(format: "%.1f", Double(endPosition)/16000.0))s")
        }
        
        // Reset decoder state only for first chunk
        if chunkIndex == 0 {
            try await manager.initializeDecoderState()
        }
        
        // Transcribe chunk
        let chunkStartTime = Date()
        let (tokenIds, encoderSequenceLength) = try await manager.transcriptionCore(
            paddedChunk,
            enableDebug: false
        )
        
        let chunkResult = manager.processTranscriptionResult(
            tokenIds: tokenIds,
            encoderSequenceLength: encoderSequenceLength,
            audioSamples: paddedChunk,
            processingTime: Date().timeIntervalSince(chunkStartTime)
        )
        
        // No overlap handling needed - just return the text
        return (chunkResult.text, chunkResult.processingTime)
    }
}