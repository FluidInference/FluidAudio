import Foundation

/// Speech Segmentation functionality for VadManager
@available(macOS 13.0, iOS 16.0, *)
extension VadManager {

    // MARK: - Speech Segmentation API

    /// Segment an audio buffer into speech regions, filtering out silences.
    /// Guarantees segments respect `minSpeechDuration`, `minSilenceDuration`, and `maxSpeechDuration`.
    /// - Parameter samples: 16kHz mono PCM samples.
    /// - Parameter config: Segmentation behavior configuration.
    /// - Returns: Array of `VadSegment` describing speech-only regions (unpadded).
    public func segmentSpeech(
        _ samples: [Float],
        config: VadSegmentationConfig = .default
    ) async throws -> [VadSegment] {
        // Single pass VAD inference for entire input
        let vadResults = try await processAudioSamples(samples)

        // Build initial speech runs from voice activity
        let initial = buildSpeechRuns(from: vadResults)

        // Merge runs separated by short silences
        let merged = mergeNearbyRuns(initial, config: config)

        // Enforce min speech duration
        let minSpeechSamples = Int(config.minSpeechDuration * Double(Self.sampleRate))
        let filtered = merged.filter { ($0.endSample - $0.startSample) >= minSpeechSamples }

        // Enforce max speech duration with probability-aware splits (no re-inference)
        let split = splitOverlongRuns(
            filtered,
            vadResults: vadResults,
            totalSamples: samples.count,
            config: config
        )

        // Clamp to total sample bounds and compute times
        let clamped: [VadSegment] = split.map { seg in
            let start = max(0, min(seg.startSample, samples.count))
            let end = max(start, min(seg.endSample, samples.count))
            return VadSegment(
                startTime: Double(start) / Double(Self.sampleRate),
                endTime: Double(end) / Double(Self.sampleRate),
                startSample: start,
                endSample: end
            )
        }

        return clamped
    }

    /// Internal helper to segment precomputed VAD results without running the model.
    /// Useful for unit testing the splitting/merging logic deterministically.
    internal func segmentSpeech(
        from vadResults: [VadResult],
        totalSamples: Int,
        config: VadSegmentationConfig = .default
    ) async -> [VadSegment] {
        // Build initial speech runs from voice activity
        let initial = buildSpeechRuns(from: vadResults)

        // Merge runs separated by short silences
        let merged = mergeNearbyRuns(initial, config: config)

        // Enforce min speech duration
        let minSpeechSamples = Int(config.minSpeechDuration * Double(Self.sampleRate))
        let filtered = merged.filter { ($0.endSample - $0.startSample) >= minSpeechSamples }

        // Split overlong runs using provided probabilities
        let split = splitOverlongRuns(
            filtered,
            vadResults: vadResults,
            totalSamples: totalSamples,
            config: config
        )

        // Clamp and compute times
        let clamped: [VadSegment] = split.map { seg in
            let start = max(0, min(seg.startSample, totalSamples))
            let end = max(start, min(seg.endSample, totalSamples))
            return VadSegment(
                startTime: Double(start) / Double(Self.sampleRate),
                endTime: Double(end) / Double(Self.sampleRate),
                startSample: start,
                endSample: end
            )
        }

        return clamped
    }

    /// Convenience: return the actual audio for each speech segment, with padding.
    /// Uses the same single-pass VAD and segmentation logic as `segmentSpeech`.
    public func segmentSpeechAudio(
        _ samples: [Float],
        config: VadSegmentationConfig = .default
    ) async throws -> [[Float]] {
        let segments = try await segmentSpeech(samples, config: config)
        var out: [[Float]] = []
        out.reserveCapacity(segments.count)
        for seg in segments {
            let padded = addPaddingToSegment(seg, totalSamples: samples.count, config: config)
            out.append(Array(samples[padded.startSample..<padded.endSample]))
        }
        return out
    }

    // extractSpeechSegments removed in favor of `segmentSpeech`/`segmentSpeechAudio`.

    // MARK: - Internal Segmentation Methods

    /// Build initial speech runs from per-chunk VAD results.
    private func buildSpeechRuns(from vadResults: [VadResult]) -> [VadSegment] {
        var segments: [VadSegment] = []
        var currentStartChunk: Int?

        for (chunkIndex, result) in vadResults.enumerated() {
            if result.isVoiceActive {
                if currentStartChunk == nil { currentStartChunk = chunkIndex }
            } else if let start = currentStartChunk {
                let endExclusive = chunkIndex
                let startSample = start * Self.chunkSize
                let endSample = endExclusive * Self.chunkSize
                segments.append(
                    VadSegment(
                        startTime: Double(startSample) / Double(Self.sampleRate),
                        endTime: Double(endSample) / Double(Self.sampleRate),
                        startSample: startSample,
                        endSample: endSample
                    )
                )
                currentStartChunk = nil
            }
        }

        if let start = currentStartChunk {
            let endExclusive = vadResults.count
            let startSample = start * Self.chunkSize
            let endSample = endExclusive * Self.chunkSize
            segments.append(
                VadSegment(
                    startTime: Double(startSample) / Double(Self.sampleRate),
                    endTime: Double(endSample) / Double(Self.sampleRate),
                    startSample: startSample,
                    endSample: endSample
                )
            )
        }

        return segments
    }

    /// Merge segments separated by less than minSilenceDuration
    private func mergeNearbyRuns(_ segments: [VadSegment], config: VadSegmentationConfig) -> [VadSegment] {
        guard !segments.isEmpty else { return [] }

        var mergedSegments: [VadSegment] = []
        var currentSegment = segments[0]

        for i in 1..<segments.count {
            let nextSegment = segments[i]
            let silenceDuration = nextSegment.startTime - currentSegment.endTime

            if silenceDuration < config.minSilenceDuration {
                // Merge with current segment
                currentSegment = VadSegment(
                    startTime: currentSegment.startTime,
                    endTime: nextSegment.endTime,
                    startSample: currentSegment.startSample,
                    endSample: nextSegment.endSample
                )
            } else {
                // Add current segment and start new one
                mergedSegments.append(currentSegment)
                currentSegment = nextSegment
            }
        }

        // Add final segment
        mergedSegments.append(currentSegment)

        return mergedSegments
    }

    /// Split segments longer than `maxSpeechDuration` using the existing VAD probabilities.
    private func splitOverlongRuns(
        _ segments: [VadSegment],
        vadResults: [VadResult],
        totalSamples: Int,
        config: VadSegmentationConfig
    ) -> [VadSegment] {
        guard !segments.isEmpty else { return [] }

        let maxSpeechSamples = Int(config.maxSpeechDuration * Double(Self.sampleRate))
        let searchBackSamples = Int(2.0 * Double(Self.sampleRate))  // prefer a silence within the last 2s window

        var out: [VadSegment] = []
        out.reserveCapacity(segments.count)

        for seg in segments {
            var currentStart = seg.startSample
            let segmentEnd = seg.endSample

            while (segmentEnd - currentStart) > maxSpeechSamples {
                let targetEnd = min(currentStart + maxSpeechSamples, segmentEnd)

                // Convert to chunk indices
                let startChunk = max(0, currentStart / Self.chunkSize)
                let targetChunk = min(vadResults.count, (targetEnd + Self.chunkSize - 1) / Self.chunkSize)

                let searchBackChunks = max(1, searchBackSamples / Self.chunkSize)
                let searchStartChunk = max(startChunk, targetChunk - searchBackChunks)

                // Choose the chunk with the lowest probability in the window,
                // preferring any below the silence threshold
                var bestChunk = targetChunk
                var lowestProb: Float = 1.0
                for idx in searchStartChunk..<targetChunk {
                    let p = vadResults[idx].probability
                    if p < lowestProb {
                        lowestProb = p
                        bestChunk = idx
                        if p < config.silenceThresholdForSplit { break }
                    }
                }

                // Fallback if we didn't move
                if bestChunk <= startChunk { bestChunk = targetChunk }

                let splitSample = min(totalSamples, bestChunk * Self.chunkSize)
                out.append(
                    VadSegment(
                        startTime: Double(currentStart) / Double(Self.sampleRate),
                        endTime: Double(splitSample) / Double(Self.sampleRate),
                        startSample: currentStart,
                        endSample: splitSample
                    )
                )
                currentStart = splitSample
            }

            // Remainder
            out.append(
                VadSegment(
                    startTime: Double(currentStart) / Double(Self.sampleRate),
                    endTime: Double(segmentEnd) / Double(Self.sampleRate),
                    startSample: currentStart,
                    endSample: segmentEnd
                )
            )
        }

        return out
    }

    /// Add padding around a segment, clamped to audio bounds
    private func addPaddingToSegment(
        _ segment: VadSegment, totalSamples: Int, config: VadSegmentationConfig
    ) -> VadSegment {
        let paddingSamples = Int(config.speechPadding * Double(Self.sampleRate))

        let paddedStartSample = max(0, segment.startSample - paddingSamples)
        let paddedEndSample = min(totalSamples, segment.endSample + paddingSamples)

        return VadSegment(
            startTime: Double(paddedStartSample) / Double(Self.sampleRate),
            endTime: Double(paddedEndSample) / Double(Self.sampleRate),
            startSample: paddedStartSample,
            endSample: paddedEndSample
        )
    }
}
