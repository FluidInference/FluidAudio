import Foundation

/// Speech Segmentation functionality for VadManager
extension VadManager {

    // MARK: - Speech Segmentation API

    /// Segment an audio buffer into speech regions using Silero-style timestamp extraction.
    /// - Parameter samples: 16kHz mono PCM samples.
    /// - Parameter config: Segmentation behavior configuration.
    /// - Returns: Array of `VadSegment` describing speech regions padded according to `speechPadding`.
    public func segmentSpeech(
        _ samples: [Float],
        config: VadSegmentationConfig = .default
    ) async throws -> [VadSegment] {
        let vadResults = try await processAudioSamples(samples)
        return await segmentSpeech(from: vadResults, totalSamples: samples.count, config: config)
    }

    /// Segment precomputed VAD results without re-running the model.
    /// Can be used with `process(_:)` to compute metrics (e.g., RTF) alongside segmentation.
    public func segmentSpeech(
        from vadResults: [VadResult],
        totalSamples: Int,
        config: VadSegmentationConfig = .default
    ) async -> [VadSegment] {
        guard !vadResults.isEmpty, totalSamples > 0 else { return [] }

        let probabilities = vadResults.map { $0.probability }
        // If the caller pins the negative threshold, derive a matching entry threshold via the offset.
        let thresholdOverride: Float? = {
            guard let negative = config.negativeThreshold else { return nil }
            return min(1.0, negative + config.negativeThresholdOffset)
        }()
        let threshold = thresholdOverride ?? self.config.defaultThreshold
        let rawSegments = detectSpeechSampleRanges(
            probabilities: probabilities,
            audioLengthSamples: totalSamples,
            threshold: threshold,
            config: config
        )

        return rawSegments.map { range in
            let start = max(0, min(range.start, totalSamples))
            let end = max(start, min(range.end, totalSamples))
            return VadSegment(
                startTime: Double(start) / Double(Self.sampleRate),
                endTime: Double(end) / Double(Self.sampleRate)
            )
        }
    }

    /// Convenience: return the actual audio for each speech segment.
    /// Returned segments already apply padding from `VadSegmentationConfig`.
    public func segmentSpeechAudio(
        _ samples: [Float],
        config: VadSegmentationConfig = .default
    ) async throws -> [[Float]] {
        let segments = try await segmentSpeech(samples, config: config)
        return segments.map { segment in
            let startSample = segment.startSample(sampleRate: Self.sampleRate)
            let endSample = segment.endSample(sampleRate: Self.sampleRate)
            let clampedStart = max(0, min(startSample, samples.count))
            let clampedEnd = max(clampedStart, min(endSample, samples.count))
            return Array(samples[clampedStart..<clampedEnd])
        }
    }

    // MARK: - Silero-inspired state machine

    private func detectSpeechSampleRanges(
        probabilities: [Float],
        audioLengthSamples: Int,
        threshold: Float,
        config: VadSegmentationConfig
    ) -> [(start: Int, end: Int)] {
        struct CandidateSilence {
            let start: Int
            let duration: Int
            let minProbability: Float
        }

        guard !probabilities.isEmpty else { return [] }

        let hopSizeSamples = Self.chunkSize
        let windowSizeSamples = Self.chunkSize
        let minSpeechSamples = Int(config.minSpeechDuration * Double(Self.sampleRate))
        let speechPadSamples = Int(config.speechPadding * Double(Self.sampleRate))
        let maxSpeechSamplesBase: Int
        if config.maxSpeechDuration.isInfinite {
            maxSpeechSamplesBase = Int.max
        } else {
            let raw =
                Int(config.maxSpeechDuration * Double(Self.sampleRate)) - windowSizeSamples - (2 * speechPadSamples)
            maxSpeechSamplesBase = max(0, raw)
        }
        let maxSpeechSamples = maxSpeechSamplesBase
        let minSilenceSamples = Int(config.minSilenceDuration * Double(Self.sampleRate))
        let minSilenceAtMaxSpeech = Int(config.minSilenceAtMaxSpeech * Double(Self.sampleRate))
        let negativeThreshold = config.effectiveNegativeThreshold(baseThreshold: threshold)

        var triggered = false
        var currentSpeechStart = 0
        var tempEnd: Int?
        var tempSilenceMinProb: Float?
        var possibleEnds: [CandidateSilence] = []
        var speeches: [(start: Int, end: Int)] = []

        func flushCurrentSpeechEnd(_ endSample: Int) {
            guard endSample > currentSpeechStart else { return }
            if (endSample - currentSpeechStart) >= minSpeechSamples {
                let clampedEnd = min(endSample, audioLengthSamples)
                speeches.append((start: currentSpeechStart, end: clampedEnd))
            }
        }

        for (index, prob) in probabilities.enumerated() {
            let frameStart = index * hopSizeSamples

            if prob >= threshold {
                if let tempEndSample = tempEnd {
                    let silenceDuration = frameStart - tempEndSample
                    if silenceDuration > minSilenceAtMaxSpeech {
                        let candidate = CandidateSilence(
                            start: tempEndSample,
                            duration: silenceDuration,
                            minProbability: tempSilenceMinProb ?? 1.0
                        )
                        possibleEnds.append(candidate)
                    }
                }
                tempEnd = nil
                tempSilenceMinProb = nil

                if !triggered {
                    triggered = true
                    currentSpeechStart = frameStart
                    continue
                }
            }

            if triggered && maxSpeechSamples < Int.max {
                let currentDuration = frameStart - currentSpeechStart
                if currentDuration > maxSpeechSamples {
                    var chosenSplit: CandidateSilence?
                    if !possibleEnds.isEmpty {
                        if let candidateBelowThreshold =
                            possibleEnds
                            .filter({ $0.minProbability <= config.silenceThresholdForSplit })
                            .max(by: { $0.duration < $1.duration })
                        {
                            chosenSplit = candidateBelowThreshold
                        } else if config.useMaxPossibleSilenceAtMaxSpeech {
                            chosenSplit = possibleEnds.max(by: { $0.duration < $1.duration })
                        } else {
                            chosenSplit = possibleEnds.last
                        }
                    }

                    let splitEnd = chosenSplit?.start ?? frameStart
                    flushCurrentSpeechEnd(splitEnd)

                    if let split = chosenSplit {
                        let newStart = split.start + split.duration
                        if newStart < frameStart {
                            currentSpeechStart = newStart
                            triggered = true
                        } else {
                            triggered = false
                        }
                    } else {
                        triggered = false
                    }

                    possibleEnds.removeAll()
                    tempEnd = nil
                    tempSilenceMinProb = nil

                    if !triggered {
                        continue
                    }
                }
            }

            if prob < negativeThreshold && triggered {
                if tempEnd == nil { tempEnd = frameStart }
                tempSilenceMinProb = min(tempSilenceMinProb ?? prob, prob)
                if let startSilence = tempEnd, frameStart - startSilence >= minSilenceSamples {
                    flushCurrentSpeechEnd(startSilence)
                    triggered = false
                    tempEnd = nil
                    tempSilenceMinProb = nil
                    possibleEnds.removeAll()
                    continue
                }
            }
        }

        if triggered {
            flushCurrentSpeechEnd(audioLengthSamples)
        }

        guard !speeches.isEmpty else { return [] }

        var adjusted = speeches
        for index in 0..<adjusted.count {
            if index == 0 {
                adjusted[index].start = max(0, adjusted[index].start - speechPadSamples)
            }

            if index < adjusted.count - 1 {
                let silence = adjusted[index + 1].start - adjusted[index].end
                if silence < 2 * speechPadSamples {
                    let half = silence / 2
                    adjusted[index].end = min(audioLengthSamples, adjusted[index].end + half)
                    adjusted[index + 1].start = max(0, adjusted[index + 1].start - half)
                } else {
                    adjusted[index].end = min(audioLengthSamples, adjusted[index].end + speechPadSamples)
                    adjusted[index + 1].start = max(0, adjusted[index + 1].start - speechPadSamples)
                }
            } else {
                adjusted[index].end = min(audioLengthSamples, adjusted[index].end + speechPadSamples)
            }
        }

        return
            adjusted
            .map { range -> (start: Int, end: Int) in
                let start = max(0, min(range.start, audioLengthSamples))
                let end = max(start, min(range.end, audioLengthSamples))
                return (start: start, end: end)
            }
            .filter { $0.end > $0.start }
    }
}
