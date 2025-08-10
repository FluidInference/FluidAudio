import CoreML
import FluidAudio
import Foundation

/// Example showing how to use both bundled diarization and ASR models together
/// for complete audio analysis (speaker identification + speech recognition).
@available(macOS 13.0, iOS 16.0, *)
class BundledCombinedExample {
    
    private var diarizerManager: DiarizerManager?
    private var asrManager: AsrManager?
    
    // MARK: - Initialize Both Systems
    
    func initialize() async throws {
        // Initialize both diarization and ASR with bundled models
        async let diarizer = BundledDiarizationExample.initializeWithBundledModels()
        async let asr = BundledASRExample.initializeWithBundledModels()
        
        self.diarizerManager = try await diarizer
        self.asrManager = try await asr
        
        print("Both diarization and ASR systems initialized successfully!")
    }
    
    // MARK: - Combined Analysis
    
    func analyzeAudioFile(audioSamples: [Float], sampleRate: Int = 16000) async throws -> CombinedResult {
        guard let diarizer = diarizerManager, let asr = asrManager else {
            throw ExampleError.systemNotInitialized("Systems not initialized")
        }
        
        let startTime = Date()
        
        // Step 1: Perform diarization to identify speakers and segments
        print("Performing speaker diarization...")
        let diarizationResult = try diarizer.performCompleteDiarization(audioSamples, sampleRate: sampleRate)
        
        print("Found \(diarizationResult.segments.count) speaker segments")
        print("Unique speakers: \(Set(diarizationResult.segments.map { $0.speakerId }).count)")
        
        // Step 2: Transcribe each speaker segment
        print("Transcribing speaker segments...")
        var transcribedSegments: [TranscribedSegment] = []
        
        for segment in diarizationResult.segments {
            // Extract audio for this segment
            let startSample = Int(segment.startTime * Double(sampleRate))
            let endSample = Int(segment.endTime * Double(sampleRate))
            let segmentLength = endSample - startSample
            
            // Ensure we don't go out of bounds
            let actualStartSample = max(0, min(startSample, audioSamples.count - 1))
            let actualEndSample = max(actualStartSample, min(endSample, audioSamples.count))
            
            if actualEndSample > actualStartSample {
                let segmentAudio = Array(audioSamples[actualStartSample..<actualEndSample])
                
                // Transcribe this segment
                do {
                    // Reset decoder state for each new segment
                    try await asr.resetDecoderState(for: .microphone)
                    
                    let transcriptionResult = try await asr.transcribe(segmentAudio)
                    
                    let transcribedSegment = TranscribedSegment(
                        speakerId: segment.speakerId,
                        startTime: segment.startTime,
                        endTime: segment.endTime,
                        text: transcriptionResult.text,
                        confidence: transcriptionResult.confidence,
                        tokenTimings: transcriptionResult.tokenTimings ?? []
                    )
                    
                    transcribedSegments.append(transcribedSegment)
                    
                    print("Speaker \(segment.speakerId) (\(segment.startTime)s-\(segment.endTime)s): '\(transcriptionResult.text)'")
                    
                } catch {
                    print("Failed to transcribe segment for speaker \(segment.speakerId): \(error)")
                    
                    // Add segment without transcription
                    let transcribedSegment = TranscribedSegment(
                        speakerId: segment.speakerId,
                        startTime: segment.startTime,
                        endTime: segment.endTime,
                        text: "[Transcription failed]",
                        confidence: 0.0,
                        tokenTimings: []
                    )
                    transcribedSegments.append(transcribedSegment)
                }
            }
        }
        
        let totalProcessingTime = Date().timeIntervalSince(startTime)
        
        return CombinedResult(
            diarizationResult: diarizationResult,
            transcribedSegments: transcribedSegments,
            totalProcessingTime: totalProcessingTime
        )
    }
    
    // MARK: - Generate Conversation Transcript
    
    func generateTranscript(from result: CombinedResult) -> String {
        var transcript = "=== CONVERSATION TRANSCRIPT ===\n\n"
        
        // Sort segments by start time
        let sortedSegments = result.transcribedSegments.sorted { $0.startTime < $1.startTime }
        
        // Group consecutive segments by the same speaker
        var groupedSegments: [(speakerId: String, segments: [TranscribedSegment])] = []
        
        for segment in sortedSegments {
            if let lastGroup = groupedSegments.last, lastGroup.speakerId == segment.speakerId {
                // Add to existing group
                groupedSegments[groupedSegments.count - 1].segments.append(segment)
            } else {
                // Start new group
                groupedSegments.append((segment.speakerId, [segment]))
            }
        }
        
        // Format transcript
        for (speakerId, segments) in groupedSegments {
            let startTime = segments.first?.startTime ?? 0
            let endTime = segments.last?.endTime ?? 0
            let combinedText = segments.map { $0.text }.joined(separator: " ").trimmingCharacters(in: .whitespaces)
            
            if !combinedText.isEmpty && combinedText != "[Transcription failed]" {
                transcript += "[\(String(format: "%.1f", startTime))s - \(String(format: "%.1f", endTime))s] Speaker \(speakerId):\n"
                transcript += "\(combinedText)\n\n"
            }
        }
        
        // Add summary
        transcript += "=== SUMMARY ===\n"
        transcript += "Total duration: \(String(format: "%.1f", result.diarizationResult.segments.last?.endTime ?? 0))s\n"
        transcript += "Unique speakers: \(Set(result.transcribedSegments.map { $0.speakerId }).count)\n"
        transcript += "Total segments: \(result.transcribedSegments.count)\n"
        transcript += "Processing time: \(String(format: "%.2f", result.totalProcessingTime))s\n"
        
        return transcript
    }
    
    // MARK: - Speaker-Specific Analysis
    
    func analyzeBySpeaker(from result: CombinedResult) -> [String: SpeakerAnalysis] {
        var speakerAnalysis: [String: SpeakerAnalysis] = [:]
        
        for segment in result.transcribedSegments {
            let speakerId = segment.speakerId
            
            if speakerAnalysis[speakerId] == nil {
                speakerAnalysis[speakerId] = SpeakerAnalysis(
                    speakerId: speakerId,
                    segments: [],
                    totalSpeakingTime: 0,
                    wordCount: 0,
                    averageConfidence: 0
                )
            }
            
            speakerAnalysis[speakerId]?.segments.append(segment)
            speakerAnalysis[speakerId]?.totalSpeakingTime += (segment.endTime - segment.startTime)
            speakerAnalysis[speakerId]?.wordCount += segment.text.split(separator: " ").count
            
            // Update average confidence
            let currentCount = speakerAnalysis[speakerId]?.segments.count ?? 1
            let currentAvg = speakerAnalysis[speakerId]?.averageConfidence ?? 0
            speakerAnalysis[speakerId]?.averageConfidence = (currentAvg * Double(currentCount - 1) + segment.confidence) / Double(currentCount)
        }
        
        return speakerAnalysis
    }
    
    // MARK: - Cleanup
    
    func cleanup() {
        diarizerManager?.cleanup()
        asrManager?.cleanup()
        diarizerManager = nil
        asrManager = nil
    }
    
    // MARK: - Example Usage
    
    static func demonstrateCompleteWorkflow() async throws {
        let analyzer = BundledCombinedExample()
        
        do {
            // Initialize both systems
            try await analyzer.initialize()
            
            // Generate example audio (replace with actual audio loading)
            let sampleRate = 16000
            let duration = 10.0  // 10 seconds
            let sampleCount = Int(duration * Double(sampleRate))
            
            // Create example audio with multiple "speakers" (different frequencies)
            var audioSamples: [Float] = []
            let segmentDuration = 2.0  // 2 seconds per "speaker"
            let segmentSamples = Int(segmentDuration * Double(sampleRate))
            
            for segment in 0..<5 {  // 5 segments of 2 seconds each
                let frequency = 0.01 + (0.005 * Double(segment % 3))  // Vary frequency for different "speakers"
                let amplitude = 0.5 + (0.2 * Double(segment % 2))      // Vary amplitude
                
                for i in 0..<segmentSamples {
                    let sampleIndex = segment * segmentSamples + i
                    let value = Float(sin(Double(sampleIndex) * frequency) * amplitude)
                    audioSamples.append(value)
                }
            }
            
            // Perform complete analysis
            print("Starting complete audio analysis...")
            let result = try await analyzer.analyzeAudioFile(audioSamples: audioSamples, sampleRate: sampleRate)
            
            // Generate transcript
            let transcript = analyzer.generateTranscript(from: result)
            print("\n" + transcript)
            
            // Analyze by speaker
            let speakerAnalysis = analyzer.analyzeBySpeaker(from: result)
            print("=== SPEAKER ANALYSIS ===")
            for (speakerId, analysis) in speakerAnalysis.sorted(by: { $0.key < $1.key }) {
                print("Speaker \(speakerId):")
                print("  Speaking time: \(String(format: "%.1f", analysis.totalSpeakingTime))s")
                print("  Word count: \(analysis.wordCount)")
                print("  Average confidence: \(String(format: "%.2f", analysis.averageConfidence))")
                print("  Segments: \(analysis.segments.count)")
            }
            
        } catch {
            print("Analysis failed: \(error)")
        } finally {
            analyzer.cleanup()
        }
    }
}

// MARK: - Data Structures

struct CombinedResult {
    let diarizationResult: DiarizationResult
    let transcribedSegments: [TranscribedSegment]
    let totalProcessingTime: TimeInterval
}

struct TranscribedSegment {
    let speakerId: String
    let startTime: Double
    let endTime: Double
    let text: String
    let confidence: Double
    let tokenTimings: [TokenTiming]
}

struct SpeakerAnalysis {
    let speakerId: String
    var segments: [TranscribedSegment]
    var totalSpeakingTime: Double
    var wordCount: Int
    var averageConfidence: Double
}

// MARK: - Extended Error Types

extension ExampleError {
    static func systemNotInitialized(_ message: String) -> ExampleError {
        return .audioLoadingFailed("System not initialized: \(message)")
    }
}

// MARK: - Usage Instructions

/*
 Complete Audio Analysis Usage:
 
 1. Bundle All Models:
    - Diarization models: pyannote_segmentation.mlmodelc, wespeaker_v2.mlmodelc
    - ASR models: Melspectogram.mlmodelc, ParakeetEncoder_v2.mlmodelc, ParakeetDecoder.mlmodelc, RNNTJoint.mlmodelc
    - Vocabulary: parakeet_vocab.json
 
 2. Use Complete Analysis:
    ```swift
    try await BundledCombinedExample.demonstrateCompleteWorkflow()
    ```
 
 3. Features:
    - Automatic speaker identification
    - Per-speaker transcription
    - Conversation transcript generation  
    - Speaker statistics and analysis
    - Performance metrics
 
 4. Real-World Applications:
    - Meeting transcription
    - Interview analysis
    - Podcast processing
    - Call center analysis
    - Educational content processing
 
 Performance Considerations:
 - Sequential processing: Diarization first, then transcription per segment
 - Memory usage: Both systems loaded simultaneously
 - Processing time: ~2-5x real-time depending on hardware
 - Accuracy: Best results with clear, high-quality audio
 
 Optimization Tips:
 - Use .cpuAndNeuralEngine for both systems
 - Reset ASR decoder state between segments
 - Consider parallel processing for independent segments
 - Implement caching for repeated analysis
 - Monitor memory usage for long audio files
 */