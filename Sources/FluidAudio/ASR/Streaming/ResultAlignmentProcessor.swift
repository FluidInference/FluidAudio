import CoreMedia
import Foundation
import OSLog

/// Processor responsible for aligning ASR transcription results with speaker segments
@available(macOS 13.0, iOS 16.0, *)
internal actor ResultAlignmentProcessor {
    private let logger = Logger(
        subsystem: "com.fluidinfluence.diarized-asr",
        category: "Alignment"
    )
    private let alignmentTolerance: TimeInterval

    // Pending results waiting for alignment
    private var pendingAsrResults: [StreamingTranscriptionResult] = []
    private var pendingSpeakerSegments: [TimedSpeakerSegment] = []

    // Successfully aligned results
    private var alignedResults: [DiarizedTranscriptionResult] = []

    init(alignmentTolerance: TimeInterval = 0.5) {
        self.alignmentTolerance = alignmentTolerance
    }

    /// Add ASR results for alignment
    func addAsrResults(_ results: [StreamingTranscriptionResult]) {
        pendingAsrResults.append(contentsOf: results)
        logger.debug("Added \(results.count) ASR results, total pending: \(self.pendingAsrResults.count)")
    }

    /// Add speaker segments for alignment
    func addSpeakerSegments(_ segments: [TimedSpeakerSegment]) {
        pendingSpeakerSegments.append(contentsOf: segments)
        logger.debug("Added \(segments.count) speaker segments, total pending: \(self.pendingSpeakerSegments.count)")
    }

    /// Process alignment and return newly aligned results
    func processAlignment() -> [DiarizedTranscriptionResult] {
        var newlyAligned: [DiarizedTranscriptionResult] = []
        var remainingAsrResults: [StreamingTranscriptionResult] = []

        // Try to align each ASR result with speaker segments
        for asrResult in pendingAsrResults {
            if let alignment = findBestAlignment(for: asrResult) {
                let diarizedResult = createDiarizedResult(
                    from: alignment.asrResult,
                    speakerSegment: alignment.speakerSegment,
                    alignmentConfidence: alignment.alignmentConfidence
                )
                newlyAligned.append(diarizedResult)
                alignedResults.append(diarizedResult)

                logger.debug(
                    "Aligned ASR result with speaker \(alignment.speakerSegment?.speakerId ?? "unknown") (confidence: \(alignment.alignmentConfidence))"
                )
            } else {
                // Keep unaligned results for future attempts
                remainingAsrResults.append(asrResult)
            }
        }

        // Update pending results
        pendingAsrResults = remainingAsrResults

        // Clean up old speaker segments that are unlikely to be matched
        cleanupOldSegments()

        logger.debug(
            "Processed alignment: \(newlyAligned.count) newly aligned, \(self.pendingAsrResults.count) ASR results pending, \(self.pendingSpeakerSegments.count) speaker segments pending"
        )

        return newlyAligned
    }

    /// Find the best speaker segment alignment for an ASR result
    private func findBestAlignment(for asrResult: StreamingTranscriptionResult) -> AsrSpeakerAlignment? {
        let asrStart = asrResult.audioTimeRange.start.seconds
        let asrEnd = asrResult.audioTimeRange.end.seconds
        let asrMidpoint = (asrStart + asrEnd) / 2.0

        var bestAlignment: AsrSpeakerAlignment?
        var bestScore: Float = 0.0

        for (index, speakerSegment) in pendingSpeakerSegments.enumerated() {
            let segmentStart = Double(speakerSegment.startTimeSeconds)
            let segmentEnd = Double(speakerSegment.endTimeSeconds)
            let segmentMidpoint = (segmentStart + segmentEnd) / 2.0

            // Calculate temporal overlap
            let overlapStart = max(asrStart, segmentStart)
            let overlapEnd = min(asrEnd, segmentEnd)
            let overlapDuration = max(0, overlapEnd - overlapStart)

            let asrDuration = asrEnd - asrStart
            let segmentDuration = segmentEnd - segmentStart

            // Calculate alignment confidence based on:
            // 1. Temporal overlap ratio
            // 2. Distance between midpoints
            // 3. Speaker segment quality

            let overlapRatio = overlapDuration / max(asrDuration, segmentDuration)
            let midpointDistance = abs(asrMidpoint - segmentMidpoint)
            let distanceScore = max(0, 1.0 - Float(midpointDistance) / Float(alignmentTolerance))

            let alignmentScore = Float(overlapRatio) * 0.7 + distanceScore * 0.3
            let qualityWeightedScore = alignmentScore * speakerSegment.qualityScore

            // More flexible alignment criteria: require either overlap OR reasonable proximity
            let hasOverlap = overlapDuration > 0.1
            let hasProximity = midpointDistance <= alignmentTolerance
            let asrWithinSpeakerSegment = asrStart >= segmentStart && asrEnd <= segmentEnd
            let hasReasonableScore = qualityWeightedScore > bestScore && qualityWeightedScore > 0.01

            if (hasOverlap || asrWithinSpeakerSegment || hasProximity) && hasReasonableScore {
                bestScore = qualityWeightedScore
                bestAlignment = AsrSpeakerAlignment(
                    asrResult: asrResult,
                    speakerSegment: speakerSegment,
                    alignmentConfidence: qualityWeightedScore
                )
            }
        }

        // Return alignment if we found any reasonable match
        return bestScore > 0.001 ? bestAlignment : nil
    }

    /// Create a diarized result from aligned ASR and speaker data
    private func createDiarizedResult(
        from asrResult: StreamingTranscriptionResult,
        speakerSegment: TimedSpeakerSegment?,
        alignmentConfidence: Float
    ) -> DiarizedTranscriptionResult {
        let speakerId = speakerSegment?.speakerId ?? "unknown"
        let speakerConfidence = speakerSegment?.qualityScore ?? 0.0

        // Use the original ASR text without adding speaker prefix here
        // The prefix will be added during display/output if needed
        let attributedText = asrResult.attributedText

        return DiarizedTranscriptionResult(
            segmentID: asrResult.segmentID,
            revision: asrResult.revision,
            speakerId: speakerId,
            attributedText: attributedText,
            audioTimeRange: asrResult.audioTimeRange,
            isFinal: asrResult.isFinal,
            transcriptionConfidence: asrResult.confidence,
            speakerConfidence: speakerConfidence,
            timestamp: Date()
        )
    }

    /// Clean up old segments that are unlikely to be matched
    private func cleanupOldSegments() {
        let currentTime = Date().timeIntervalSince1970
        let cleanupThreshold = currentTime - 30.0  // Remove segments older than 30 seconds

        // Remove old ASR results based on their timestamp
        pendingAsrResults.removeAll { result in
            result.timestamp.timeIntervalSince1970 < cleanupThreshold
        }

        // Don't remove speaker segments based on audio time - they use relative time within the audio file
        // Only remove them if there are too many accumulated to prevent unbounded growth
        if pendingSpeakerSegments.count > 100 {
            // Keep only the most recent 50 segments
            pendingSpeakerSegments = Array(pendingSpeakerSegments.suffix(50))
        }

        // Also clean up very old aligned results to prevent unbounded growth
        if alignedResults.count > 1000 {
            alignedResults.removeFirst(alignedResults.count - 500)
        }
    }

    /// Get statistics about the alignment process
    var alignmentStats: (pendingAsrCount: Int, pendingSpeakerCount: Int, alignedCount: Int) {
        (pendingAsrResults.count, pendingSpeakerSegments.count, alignedResults.count)
    }

    /// Reset all alignment state
    func reset() {
        pendingAsrResults.removeAll()
        pendingSpeakerSegments.removeAll()
        alignedResults.removeAll()
        logger.info("Alignment processor reset")
    }
}
