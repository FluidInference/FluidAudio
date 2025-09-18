import CoreML
import Foundation
import OSLog

struct VadBasedProcessor {
    let audioSamples: [Float]
    let vadConfig: VadSegmentationConfig
    let minSegmentDuration: Double

    private let logger = AppLogger(category: "VadBasedProcessor")
    private let sampleRate: Int = 16000
    private let maxSegmentDuration: Double = 15.0  // 15s maximum for Parakeet model

    init(
        audioSamples: [Float],
        vadConfig: VadSegmentationConfig,
        minSegmentDuration: Double = 5.0
    ) {
        self.audioSamples = audioSamples
        self.vadConfig = vadConfig
        self.minSegmentDuration = minSegmentDuration
    }

    func process(using manager: AsrManager) async throws -> ASRResult {
        let startTime = Date()

        // 1. Run VAD segmentation
        let vadManager = try await VadManager(config: VadConfig.default)
        let vadSegments = try await vadManager.segmentSpeech(audioSamples, config: vadConfig)

        logger.info("VAD found \(vadSegments.count) speech segments")

        guard !vadSegments.isEmpty else {
            logger.warning("No speech segments found in audio")
            let duration = TimeInterval(audioSamples.count) / TimeInterval(sampleRate)
            return ASRResult(
                text: "",
                confidence: 0.0,
                duration: duration,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        // 2. Merge short segments and handle long ones
        let processableSegments = mergeShortSegments(vadSegments)
        logger.info("After merging: \(processableSegments.count) processable segments")

        // 3. Process each segment with the shared chunk engine
        var combinedTokenData: [(token: Int, timestamp: Int, confidence: Float)] = []

        for (index, segment) in processableSegments.enumerated() {
            // Add a small amount of leading audio for non-initial segments so the decoder sees context.
            let leadingPadSamples = index == 0 ? 0 : Int(vadConfig.speechPadding * Double(sampleRate))
            let extraction = extractAudio(from: segment, leadingPadSamples: leadingPadSamples)
            let segmentAudio = extraction.audio
            let segmentStartSample = extraction.startSample

            guard !segmentAudio.isEmpty else {
                logger.debug("Segment \(index) is empty after extraction, skipping")
                continue
            }

            var workingDecoderState: TdtDecoderState
            do {
                workingDecoderState = try TdtDecoderState()
            } catch {
                workingDecoderState = TdtDecoderState(fallback: true)
            }

            try await manager.initializeDecoderState(decoderState: &workingDecoderState)
            let engine = ChunkProcessingEngine(
                audioSamples: segmentAudio,
                startSampleInOriginalAudio: segmentStartSample
            )

            let chunkResult = try await engine.run(using: manager, decoderState: &workingDecoderState)

            guard chunkResult.tokens.count == chunkResult.timestamps.count,
                chunkResult.tokens.count == chunkResult.confidences.count
            else {
                throw ASRError.processingFailed(
                    "Token, timestamp, and confidence arrays are misaligned for segment \(index)")
            }

            let segmentData = zip(zip(chunkResult.tokens, chunkResult.timestamps), chunkResult.confidences).map {
                (token: $0.0.0, timestamp: $0.0.1, confidence: $0.1)
            }

            if !segmentData.isEmpty {
                let tokensText = chunkResult.tokens.compactMap { manager.vocabulary[$0] }.joined()
                    .replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
                logger.debug("Segment \(index) result: '\(tokensText)'")
            }

            if index > 0 && !combinedTokenData.isEmpty && !segmentData.isEmpty {
                let previousTokens = combinedTokenData.map { $0.token }
                let currentTokens = segmentData.map { $0.token }

                let (_, removedCount) = manager.removeDuplicateTokenSequence(
                    previous: previousTokens,
                    current: currentTokens,
                    maxOverlap: 30
                )
                let adjustedSegmentData = Array(segmentData.dropFirst(removedCount))
                combinedTokenData.append(contentsOf: adjustedSegmentData)
            } else {
                combinedTokenData.append(contentsOf: segmentData)
            }
        }

        combinedTokenData.sort { $0.timestamp < $1.timestamp }

        let allTokens = combinedTokenData.map { $0.token }
        let allTimestamps = combinedTokenData.map { $0.timestamp }
        let allConfidences = combinedTokenData.map { $0.confidence }

        return manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            confidences: allConfidences,
            encoderSequenceLength: 0,  // Not relevant for VAD-based processing
            audioSamples: audioSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    // MARK: - Helper methods

    private func mergeShortSegments(_ vadSegments: [VadSegment]) -> [ProcessableSegment] {
        var processed: [ProcessableSegment] = []
        var pendingSegments: [VadSegment] = []
        var pendingDuration: Double = 0

        for segment in vadSegments {
            let duration = segment.duration

            if duration >= minSegmentDuration && duration <= maxSegmentDuration {
                // Flush any pending segments first
                if !pendingSegments.isEmpty {
                    let mergedSegment = ProcessableSegment(merging: pendingSegments, in: audioSamples)
                    if mergedSegment.duration > maxSegmentDuration {
                        // Split the merged segment if it's too long
                        let splits = splitLongSegment(
                            VadSegment(startTime: mergedSegment.startTime, endTime: mergedSegment.endTime))
                        processed.append(contentsOf: splits)
                    } else {
                        processed.append(mergedSegment)
                    }
                    pendingSegments.removeAll()
                    pendingDuration = 0
                }
                // Add this good segment
                processed.append(ProcessableSegment(single: segment, in: audioSamples))

            } else if duration < minSegmentDuration {
                // Check if adding this segment would exceed the max duration
                if pendingDuration + duration <= maxSegmentDuration {
                    pendingSegments.append(segment)
                    pendingDuration += duration
                } else {
                    // Would exceed max, flush pending and start new
                    if !pendingSegments.isEmpty {
                        let mergedSegment = ProcessableSegment(merging: pendingSegments, in: audioSamples)
                        if mergedSegment.duration > maxSegmentDuration {
                            // Split the merged segment if it's too long
                            let splits = splitLongSegment(
                                VadSegment(startTime: mergedSegment.startTime, endTime: mergedSegment.endTime))
                            processed.append(contentsOf: splits)
                        } else {
                            processed.append(mergedSegment)
                        }
                    }
                    pendingSegments = [segment]
                    pendingDuration = duration
                }

                // Check if we've accumulated enough to process
                if pendingDuration >= minSegmentDuration {
                    let mergedSegment = ProcessableSegment(merging: pendingSegments, in: audioSamples)
                    if mergedSegment.duration > maxSegmentDuration {
                        // Split the merged segment if it's too long
                        let splits = splitLongSegment(
                            VadSegment(startTime: mergedSegment.startTime, endTime: mergedSegment.endTime))
                        processed.append(contentsOf: splits)
                    } else {
                        processed.append(mergedSegment)
                    }
                    pendingSegments.removeAll()
                    pendingDuration = 0
                }

            } else {
                // Segment > 15s - split it
                logger.warning("Segment longer than 15s (\(duration)s) - splitting")
                let splits = splitLongSegment(segment)
                processed.append(contentsOf: splits)
            }
        }

        // Handle remaining pending segments
        if !pendingSegments.isEmpty {
            let mergedSegment = ProcessableSegment(merging: pendingSegments, in: audioSamples)
            if mergedSegment.duration > maxSegmentDuration {
                // Split the merged segment if it's too long
                let splits = splitLongSegment(
                    VadSegment(startTime: mergedSegment.startTime, endTime: mergedSegment.endTime))
                processed.append(contentsOf: splits)
            } else {
                processed.append(mergedSegment)
            }
        }

        return processed
    }

    private func extractAudio(
        from segment: ProcessableSegment, leadingPadSamples: Int = 0
    ) -> (
        audio: [Float], startSample: Int
    ) {
        let originalStart = segment.startSample(sampleRate: sampleRate)
        let startSample = max(0, originalStart - leadingPadSamples)
        let rawEndSample = segment.endSample(sampleRate: sampleRate)
        let trailingTrim = Int(vadConfig.speechPadding * Double(sampleRate) * 0.5)
        let endSample = max(startSample, rawEndSample - trailingTrim)
        let clampedStart = max(0, min(startSample, audioSamples.count))
        let clampedEnd = max(clampedStart, min(endSample, audioSamples.count))

        let audioSlice = clampedStart < clampedEnd ? Array(audioSamples[clampedStart..<clampedEnd]) : []
        return (audioSlice, clampedStart)
    }

    private func splitLongSegment(_ segment: VadSegment) -> [ProcessableSegment] {
        // Split segments > 15s into smaller chunks
        var segments: [ProcessableSegment] = []
        let maxChunkDuration = 14.0  // Conservative: ensure well under 15s limit
        let overlapDuration = 1.0  // 1s overlap between chunks

        var currentStart = segment.startTime
        while currentStart < segment.endTime {
            let remainingDuration = segment.endTime - currentStart

            if remainingDuration <= maxSegmentDuration {
                // Last chunk - take everything remaining if it fits
                let finalSegment = VadSegment(startTime: currentStart, endTime: segment.endTime)
                segments.append(ProcessableSegment(single: finalSegment, in: audioSamples))
                break
            } else {
                // Create a chunk of max duration
                let currentEnd = currentStart + maxChunkDuration
                let splitSegment = VadSegment(startTime: currentStart, endTime: currentEnd)
                segments.append(ProcessableSegment(single: splitSegment, in: audioSamples))

                // Move to next chunk with overlap
                currentStart = currentEnd - overlapDuration

                // Ensure we don't go backwards
                if currentStart >= currentEnd {
                    currentStart = currentEnd
                }
            }
        }

        logger.debug("Split long segment (\(segment.duration)s) into \(segments.count) chunks")
        return segments
    }
}

/// Represents a segment that can be processed by the ASR model
struct ProcessableSegment {
    let startTime: Double
    let endTime: Double
    let originalSegments: [VadSegment]  // Track which VAD segments this combines

    var duration: Double {
        return endTime - startTime
    }

    init(single segment: VadSegment, in audioSamples: [Float]) {
        self.startTime = segment.startTime
        self.endTime = segment.endTime
        self.originalSegments = [segment]
    }

    init(merging segments: [VadSegment], in audioSamples: [Float]) {
        guard !segments.isEmpty else {
            self.startTime = 0
            self.endTime = 0
            self.originalSegments = []
            return
        }

        self.startTime = segments.map(\.startTime).min() ?? 0
        self.endTime = segments.map(\.endTime).max() ?? 0
        self.originalSegments = segments
    }

    func startSample(sampleRate: Int) -> Int {
        return Int(startTime * Double(sampleRate))
    }

    func endSample(sampleRate: Int) -> Int {
        return Int(endTime * Double(sampleRate))
    }
}
