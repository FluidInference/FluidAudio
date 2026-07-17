import Foundation

struct ChunkProcessor {
    let sampleSource: AudioSampleSource
    let totalSamples: Int

    private let logger = AppLogger(category: "ChunkProcessor")
    typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float, duration: Int)
    private struct TaskResult: Sendable {
        let index: Int
        let tokens: [TokenWindow]
        let workerIndex: Int
    }
    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let start: Double
        let end: Double
    }
    struct ChunkStartDecision {
        let start: Int
        let useWarmupPrefix: Bool
    }

    // Stateless chunking aligned with CoreML reference:
    // - process ~14.96s of audio per window (frame-aligned) to stay under encoder limit
    // - 2.0s overlap (frame-aligned) to give the decoder slack when merging windows
    let overlapSeconds: Double = 2.0

    /// 80ms prepend from the previous chunk so the encoder's convolutions
    /// have left context (blank-first-frames fix, PR #264). Opt out via
    /// `ASRConfig.melChunkContext` for v3 multilingual drift (issue #594) —
    /// see "Current Paths" in Documentation/ASR/LongTranscription.md.
    private let melContextSamples: Int = ASRConstants.samplesPerEncoderFrame  // 1280 samples = 80ms

    /// Default v3/no-mel path warmup size. v42 intentionally keeps the
    /// non-arbitrated path warmup-free; the opt-in arbitration path's path B
    /// owns the explicit 7-frame warmup probe.
    private let noMelWarmupPrefixFrames: Int = 0

    private var maxModelSamples: Int { ASRConstants.maxModelSamples }

    private var noMelWarmupPrefixSamples: Int {
        noMelWarmupPrefixFrames * ASRConstants.samplesPerEncoderFrame
    }

    /// Effective per-chunk mel-context size based on the runtime flag.
    private func effectiveMelContextSamples(melChunkContext: Bool) -> Int {
        melChunkContext ? melContextSamples : 0
    }

    private func effectiveWarmupPrefixSamples(melChunkContext: Bool, modelVersion: AsrModelVersion?) -> Int {
        guard !melChunkContext, case .v3? = modelVersion else { return 0 }
        return noMelWarmupPrefixSamples
    }

    /// Frame-aligned chunk size that reserves space for the context prepend
    /// (or fills the encoder window when context is disabled).
    private func chunkSamples(melChunkContext: Bool, modelVersion: AsrModelVersion?) -> Int {
        let reserved = effectiveMelContextSamples(melChunkContext: melChunkContext)
        let maxActualChunk = maxModelSamples - reserved
        let raw = max(maxActualChunk - ASRConstants.melHopSize, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    private func overlapSamples(forChunkSamples chunkSamples: Int) -> Int {
        let requested = Int(overlapSeconds * Double(ASRConstants.sampleRate))
        let capped = min(requested, chunkSamples / 2)
        return capped / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    private func strideSamples(forChunkSamples chunkSamples: Int) -> Int {
        let raw = max(chunkSamples - overlapSamples(forChunkSamples: chunkSamples), ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    /// End-align the final window (issue #747): fill a short last chunk
    /// backwards with real audio (decoded as a suppressed warmup prefix)
    /// instead of zero-padding, ending at `speechEndSamples` — the last
    /// speech-bearing frame, not EOF, because a window that ends in an
    /// extended dead-silence run decodes degenerately. See "End-Aligned
    /// Final Window" in Documentation/ASR/LongTranscription.md.
    /// Non-final chunks, already-full windows, and single-chunk files pass
    /// `defaultWarmupSamples` through unchanged.
    static func lastChunkWarmupSamples(
        chunkStart: Int,
        defaultWarmupSamples: Int,
        chunkSamples: Int,
        totalSamples: Int,
        speechEndSamples: Int
    ) -> Int {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let defaultVisible = max(frameSamples, chunkSamples - defaultWarmupSamples)
        let isLastChunk = chunkStart + defaultVisible >= totalSamples
        let remaining = min(speechEndSamples, totalSamples) - chunkStart
        guard isLastChunk, remaining > 0, chunkStart > 0 else { return defaultWarmupSamples }
        // Frame-aligned so the suppression boundary maps to an exact frame.
        let fill = (chunkSamples - remaining) / frameSamples * frameSamples
        guard fill > 0 else { return defaultWarmupSamples }
        let available = chunkStart / frameSamples * frameSamples
        return max(defaultWarmupSamples, min(available, fill))
    }

    /// Last speech-bearing sample of the recording: `totalSamples` minus
    /// the trailing run of frames whose RMS sits below `speechRmsFloor` —
    /// below the quietest real speech, so nothing transcribable is
    /// excluded. A window that ends inside a dead-silence run *within its
    /// declared audio length* decodes degenerately (zero padding beyond
    /// the length is masked and safe) — see "End-Aligned Final Window" in
    /// LongTranscription.md. Returns `totalSamples` for an all-sub-floor
    /// file (leave the layout untouched).
    func speechEndSamples() throws -> Int {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        var end = totalSamples
        while end > 0 {
            let frameStart = max(0, end - frameSamples)
            let samples = try readSamples(offset: frameStart, count: end - frameStart)
            var sum: Float = 0
            for sample in samples {
                sum += sample * sample
            }
            if sqrt(sum / Float(samples.count)) >= Self.speechRmsFloor {
                return end
            }
            end = frameStart
        }
        return totalSamples
    }

    /// Prefix suppression (`emitTokensAfterGlobalFrame`) is a V3-decoder
    /// feature; `TdtDecoderV2` ignores it, which would emit the backfilled
    /// prefix twice. V2-family models keep the zero-padded final window.
    static func supportsSuppressedPrefix(_ version: AsrModelVersion?) -> Bool {
        switch version {
        case .v3, .tdtJa: return true
        default: return false
        }
    }

    func chunkLayout(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) -> (
        chunkSamples: Int,
        strideSamples: Int,
        melContextSamples: Int,
        warmupPrefixSamples: Int
    ) {
        let chunkSamples = self.chunkSamples(melChunkContext: melChunkContext, modelVersion: modelVersion)
        let warmupPrefixSamples = effectiveWarmupPrefixSamples(
            melChunkContext: melChunkContext,
            modelVersion: modelVersion
        )
        let stride = strideSamples(forChunkSamples: chunkSamples)
        return (
            chunkSamples: chunkSamples,
            strideSamples: stride,
            melContextSamples: effectiveMelContextSamples(melChunkContext: melChunkContext),
            warmupPrefixSamples: warmupPrefixSamples
        )
    }

    private func chunkStarts(
        warmupPrefixSamples: Int,
        chunkSamples: Int,
        strideSamples: Int,
        preferSilenceAlignment: Bool
    ) throws -> [ChunkStartDecision] {
        guard preferSilenceAlignment || warmupPrefixSamples > 0 else {
            return regularChunkStarts(strideSamples: strideSamples)
        }
        return try silenceAlignedChunkStarts(
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            canUseWarmupPrefix: warmupPrefixSamples > 0
        )
    }

    func regularChunkStarts(strideSamples: Int) -> [ChunkStartDecision] {
        var starts = [ChunkStartDecision(start: 0, useWarmupPrefix: false)]
        var start = strideSamples
        while start < totalSamples {
            starts.append(ChunkStartDecision(start: start, useWarmupPrefix: false))
            start += strideSamples
        }
        return starts
    }

    func silenceAlignedChunkStarts(
        chunkSamples: Int,
        strideSamples: Int,
        canUseWarmupPrefix: Bool
    ) throws -> [ChunkStartDecision] {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let silenceSearchRadiusFrames = max(1, Int((4.0 * Double(ASRConstants.sampleRate)) / Double(frameSamples)))
        let valleySearchRadiusFrames = max(1, Int((0.5 * Double(ASRConstants.sampleRate)) / Double(frameSamples)))
        let halfEnergyWindowSamples = frameSamples
        let minimumOverlapSamples = frameSamples * 6

        var starts = [ChunkStartDecision(start: 0, useWarmupPrefix: false)]
        var previousStart = 0
        var target = strideSamples

        while target < totalSamples {
            let targetFrame = target / frameSamples
            let latestCoveredStart = previousStart + chunkSamples - minimumOverlapSamples
            let targetStart = min(max(targetFrame * frameSamples, previousStart + frameSamples), latestCoveredStart)

            let silenceCandidate = try bestBoundaryCandidate(
                targetFrame: targetFrame,
                searchRadiusFrames: silenceSearchRadiusFrames,
                previousStart: previousStart,
                latestCoveredStart: latestCoveredStart,
                halfEnergyWindowSamples: halfEnergyWindowSamples
            )
            let foundNearSilence = isNearSilenceBoundary(silenceCandidate)

            var bestStart: Int
            var useWarmupPrefix = false
            if foundNearSilence {
                let shouldWarmup =
                    canUseWarmupPrefix ? (try shouldUseWarmupPrefix(at: silenceCandidate.start)) : false
                let compressesSpeechTail: Bool
                if shouldWarmup && silenceCandidate.start < targetStart {
                    compressesSpeechTail = try wouldCompressSpeechTail(
                        candidateStart: silenceCandidate.start,
                        targetStart: targetStart,
                        chunkSamples: chunkSamples,
                        minimumOverlapSamples: minimumOverlapSamples,
                        medianScore: silenceCandidate.medianScore,
                        halfEnergyWindowSamples: halfEnergyWindowSamples
                    )
                } else {
                    compressesSpeechTail = false
                }
                if compressesSpeechTail {
                    bestStart = targetStart
                } else {
                    bestStart = silenceCandidate.start
                    useWarmupPrefix = shouldWarmup
                }
            } else {
                let valleyCandidate = try bestBoundaryCandidate(
                    targetFrame: targetFrame,
                    searchRadiusFrames: valleySearchRadiusFrames,
                    previousStart: previousStart,
                    latestCoveredStart: latestCoveredStart,
                    halfEnergyWindowSamples: halfEnergyWindowSamples
                )
                bestStart = isUsableValleyBoundary(valleyCandidate) ? valleyCandidate.start : targetStart
            }

            if bestStart <= previousStart {
                bestStart = min(previousStart + strideSamples, totalSamples)
            }

            starts.append(
                ChunkStartDecision(
                    start: bestStart,
                    useWarmupPrefix: useWarmupPrefix
                )
            )
            previousStart = bestStart
            target += strideSamples
        }

        return starts
    }

    private func bestBoundaryCandidate(
        targetFrame: Int,
        searchRadiusFrames: Int,
        previousStart: Int,
        latestCoveredStart: Int,
        halfEnergyWindowSamples: Int
    ) throws -> (start: Int, score: Float, medianScore: Float) {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let lowerFrame = max(1, targetFrame - searchRadiusFrames)
        let upperFrame = min((totalSamples - 1) / frameSamples, targetFrame + searchRadiusFrames)
        let targetStart = min(max(targetFrame * frameSamples, previousStart + frameSamples), latestCoveredStart)

        var bestStart = targetStart
        var bestScore = Float.greatestFiniteMagnitude
        var scores: [Float] = []

        if lowerFrame <= upperFrame {
            for frameIndex in lowerFrame...upperFrame {
                let candidate = frameIndex * frameSamples
                if candidate <= previousStart { continue }
                if candidate > latestCoveredStart { continue }
                let score = try boundaryEnergyScore(
                    centeredAt: candidate,
                    halfWindowSamples: halfEnergyWindowSamples
                )
                scores.append(score)
                if score < bestScore {
                    bestScore = score
                    bestStart = candidate
                }
            }
        }

        guard !scores.isEmpty else {
            return (targetStart, Float.greatestFiniteMagnitude, 0)
        }

        let sortedScores = scores.sorted()
        let medianScore = sortedScores[sortedScores.count / 2]
        return (bestStart, bestScore, medianScore)
    }

    private func isNearSilenceBoundary(_ candidate: (start: Int, score: Float, medianScore: Float)) -> Bool {
        candidate.score <= adaptiveBoundaryThreshold(medianScore: candidate.medianScore, ratio: 0.05)
    }

    private func isUsableValleyBoundary(_ candidate: (start: Int, score: Float, medianScore: Float)) -> Bool {
        candidate.score <= adaptiveBoundaryThreshold(medianScore: candidate.medianScore, ratio: 0.35)
    }

    private func adaptiveBoundaryThreshold(medianScore: Float, ratio: Float) -> Float {
        guard medianScore > 0 else { return 0 }
        return medianScore * ratio
    }

    private func wouldCompressSpeechTail(
        candidateStart: Int,
        targetStart: Int,
        chunkSamples: Int,
        minimumOverlapSamples: Int,
        medianScore: Float,
        halfEnergyWindowSamples: Int
    ) throws -> Bool {
        guard medianScore > 0 else { return false }

        let forcedNextBoundary = candidateStart + chunkSamples - minimumOverlapSamples
        guard forcedNextBoundary < totalSamples else { return false }

        let speechLikeThreshold = medianScore * 0.8
        let targetScore = try boundaryEnergyScore(
            centeredAt: targetStart,
            halfWindowSamples: halfEnergyWindowSamples
        )
        let forcedScore = try boundaryEnergyScore(
            centeredAt: forcedNextBoundary,
            halfWindowSamples: halfEnergyWindowSamples
        )
        return targetScore > speechLikeThreshold && forcedScore > speechLikeThreshold
    }

    private func shouldUseWarmupPrefix(at centerSample: Int) throws -> Bool {
        let lookaheadSamples = Int(0.5 * Double(ASRConstants.sampleRate))
        let minimumStableQuietSamples = Int(0.2 * Double(ASRConstants.sampleRate))
        let windowSamples = max(1, ASRConstants.sampleRate / 50)  // 20ms
        let quietRmsThreshold: Float = 0.003

        var offset = 0
        var quietSamples = 0

        while offset < lookaheadSamples {
            let start = centerSample + offset
            guard start < totalSamples else { break }

            let count = min(windowSamples, totalSamples - start, lookaheadSamples - offset)
            guard count > 0 else { break }

            let samples = try readSamples(offset: start, count: count)
            var sum: Float = 0
            for sample in samples {
                sum += sample * sample
            }
            let rms = sqrt(sum / Float(samples.count))
            guard rms < quietRmsThreshold else { break }

            quietSamples += samples.count
            if quietSamples >= minimumStableQuietSamples {
                return false
            }
            offset += samples.count
        }

        return true
    }

    private func boundaryEnergyScore(centeredAt centerSample: Int, halfWindowSamples: Int) throws -> Float {
        let start = max(0, centerSample - halfWindowSamples)
        let end = min(totalSamples, centerSample + halfWindowSamples)
        let count = end - start
        guard count > 0 else { return 0 }

        let samples = try readSamples(offset: start, count: count)
        var sum: Float = 0
        for sample in samples {
            sum += sample * sample
        }
        return sum / Float(count)
    }

    #if DEBUG
    internal func chunkLayoutForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) -> (
        chunkSamples: Int,
        strideSamples: Int,
        melContextSamples: Int,
        warmupPrefixSamples: Int
    ) {
        chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
    }

    internal func chunkStartsForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) throws -> [Int] {
        try chunkStartDecisionsForTesting(
            melChunkContext: melChunkContext,
            modelVersion: modelVersion
        ).map(\.start)
    }

    internal func chunkStartDecisionsForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) throws -> [(start: Int, useWarmupPrefix: Bool)] {
        let layout = chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
        return try chunkStarts(
            warmupPrefixSamples: layout.warmupPrefixSamples,
            chunkSamples: layout.chunkSamples,
            strideSamples: layout.strideSamples,
            preferSilenceAlignment: !melChunkContext && modelVersion == .v3
        ).map { ($0.start, $0.useWarmupPrefix) }
    }

    internal func mergeTokenWindowsForTesting(
        left: [(token: Int, timestamp: Int, confidence: Float, duration: Int)],
        right: [(token: Int, timestamp: Int, confidence: Float, duration: Int)],
        spliceSafeTokenIds: Set<Int>? = nil,
        caseVariantIds: [Int: Int]? = nil
    ) -> [(token: Int, timestamp: Int, confidence: Float, duration: Int)] {
        mergeChunks(left, right, spliceSafeTokenIds: spliceSafeTokenIds, caseVariantIds: caseVariantIds)
    }
    #endif

    /// Initialize with a streaming audio sample source for memory-efficient processing.
    init(sampleSource: AudioSampleSource) {
        self.sampleSource = sampleSource
        self.totalSamples = sampleSource.sampleCount
    }

    /// Convenience initializer for in-memory audio samples.
    init(audioSamples: [Float]) {
        self.init(sampleSource: ArrayAudioSampleSource(samples: audioSamples))
    }

    func process(
        using manager: AsrManager,
        startTime: Date,
        progressHandler: ((Double) async -> Void)? = nil,
        language: Language? = nil
    ) async throws -> ASRResult {
        let requestedConcurrency = max(1, await manager.parallelChunkConcurrency)
        let workers = await makeWorkerPool(using: manager, count: requestedConcurrency) ?? [manager]
        let decoderLayers = await manager.decoderLayerCount
        let maxModelSamples = self.maxModelSamples
        // Issue #594: opt-out of PR #264's 80ms mel-context prepend. For v3,
        // no-mel uses real-audio warmup plus silence-aligned chunk starts.
        let melChunkContext = await manager.melChunkContext
        let modelVersion = await manager.modelVersion
        let dualDecodeArbitration = await manager.dualDecodeArbitration

        // Dual-decode opt-in (only effective for v3 + no-mel; other paths
        // are not changed by the flag).
        if dualDecodeArbitration, !melChunkContext, modelVersion == .v3 {
            return try await processWithDualDecodeArbitration(
                using: manager,
                workers: workers,
                decoderLayers: decoderLayers,
                maxModelSamples: maxModelSamples,
                modelVersion: modelVersion,
                startTime: startTime,
                progressHandler: progressHandler,
                language: language
            )
        }

        let layout = chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
        let melContextSamples = layout.melContextSamples
        let warmupPrefixSamples = layout.warmupPrefixSamples
        let chunkSamples = layout.chunkSamples
        let strideSamples = layout.strideSamples
        let chunkStarts = try self.chunkStarts(
            warmupPrefixSamples: warmupPrefixSamples,
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            preferSilenceAlignment: !melChunkContext && modelVersion == .v3
        )

        var chunkOutputs: [[TokenWindow]?] = []
        var availableWorkers = Array(workers.indices)
        var inFlight = 0
        var chunkDecision = chunkStarts.first ?? ChunkStartDecision(start: 0, useWarmupPrefix: false)
        var chunkStart = chunkDecision.start
        var chunkIndex = 0
        let endAligned = Self.supportsSuppressedPrefix(modelVersion)
        // The final window must end at the last speech-bearing frame, not
        // EOF — see `speechEndSamples()`.
        let speechEnd = endAligned ? try speechEndSamples() : totalSamples

        func collectNextResult(
            _ group: inout ThrowingTaskGroup<TaskResult, Error>
        ) async throws {
            guard inFlight > 0 else { return }
            guard let finished = try await group.next() else { return }
            chunkOutputs[finished.index] = finished.tokens
            availableWorkers.append(finished.workerIndex)
            inFlight -= 1
        }

        try await withThrowingTaskGroup(of: TaskResult.self) { group in
            while chunkStart < totalSamples {
                try Task.checkCancellation()
                let defaultWarmupSamples =
                    chunkIndex > 0 && chunkDecision.useWarmupPrefix
                    ? min(warmupPrefixSamples, chunkStart) : 0
                // A short final chunk fills its window backwards with real
                // audio instead of zero padding (issue #747); V2-family
                // decoders can't suppress the prefix and keep the old layout.
                let warmupSamples =
                    endAligned
                    ? Self.lastChunkWarmupSamples(
                        chunkStart: chunkStart,
                        defaultWarmupSamples: defaultWarmupSamples,
                        chunkSamples: chunkSamples,
                        totalSamples: totalSamples,
                        speechEndSamples: speechEnd)
                    : defaultWarmupSamples
                let visibleChunkSamples = max(
                    ASRConstants.samplesPerEncoderFrame,
                    chunkSamples - warmupSamples
                )
                let candidateEnd = chunkStart + visibleChunkSamples
                let isLastChunk = candidateEnd >= totalSamples
                let chunkEnd = isLastChunk ? totalSamples : candidateEnd

                if chunkEnd <= chunkStart {
                    break
                }
                // The final window's audio stops at the last speech-bearing
                // frame — a window ending inside a dead-silence run decodes
                // degenerately. A pure-silence tail has nothing to decode.
                let audioEnd = isLastChunk && endAligned ? min(chunkEnd, speechEnd) : chunkEnd
                if audioEnd <= chunkStart {
                    break
                }

                // In the default path, contextSamples means mel/STFT context
                // and is skipped by the decoder. In v3/no-mel mode, the
                // warmup prefix is decoded from frame 0 and only its emitted
                // tokens are suppressed.
                let contextSamples = warmupSamples > 0 ? 0 : (chunkIndex > 0 ? melContextSamples : 0)
                let contextStart = chunkStart - max(warmupSamples, contextSamples)
                let chunkLengthWithContext = audioEnd - contextStart
                let chunkSamplesArray = try readSamples(offset: contextStart, count: chunkLengthWithContext)
                let emitTokensAfterFrame =
                    warmupSamples > 0 ? chunkStart / ASRConstants.samplesPerEncoderFrame : nil

                if availableWorkers.isEmpty {
                    try await collectNextResult(&group)
                }
                if availableWorkers.isEmpty {
                    availableWorkers.append(0)
                }

                let workerIndex = availableWorkers.removeFirst()
                let worker = workers[workerIndex]
                let index = chunkIndex
                let chunkStartOffset = warmupSamples > 0 ? contextStart : chunkStart
                chunkOutputs.append(nil)

                group.addTask {
                    var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
                    decoderState.reset()

                    let (windowTokens, windowTimestamps, windowConfidences, windowDurations) =
                        try await Self
                        .transcribeChunk(
                            samples: chunkSamplesArray,
                            contextSamples: contextSamples,
                            chunkStart: chunkStartOffset,
                            isLastChunk: isLastChunk,
                            using: worker,
                            decoderState: &decoderState,
                            maxModelSamples: maxModelSamples,
                            language: language,
                            emitTokensAfterFrame: emitTokensAfterFrame,
                            initialTimeIndexOverride: emitTokensAfterFrame == nil ? nil : 0
                        )

                    guard
                        windowTokens.count == windowTimestamps.count
                            && windowTokens.count == windowConfidences.count
                    else {
                        throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
                    }

                    let durations =
                        windowDurations.count == windowTokens.count
                        ? windowDurations : Array(repeating: 0, count: windowTokens.count)

                    let windowData: [TokenWindow] = zip(
                        zip(zip(windowTokens, windowTimestamps), windowConfidences), durations
                    ).map {
                        (token: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
                    }

                    return TaskResult(index: index, tokens: windowData, workerIndex: workerIndex)
                }
                inFlight += 1
                chunkIndex += 1

                if let progressHandler, !isLastChunk {
                    let progress = min(1.0, max(0.0, Double(chunkEnd) / Double(totalSamples)))
                    await progressHandler(progress)
                }

                if isLastChunk {
                    break
                }

                if chunkIndex < chunkStarts.count {
                    chunkDecision = chunkStarts[chunkIndex]
                    chunkStart = chunkDecision.start
                } else {
                    chunkStart += strideSamples
                    chunkDecision = ChunkStartDecision(start: chunkStart, useWarmupPrefix: false)
                }

                if availableWorkers.isEmpty && inFlight > 0 {
                    try await collectNextResult(&group)
                }
            }

            while inFlight > 0 {
                try Task.checkCancellation()
                try await collectNextResult(&group)
            }
        }

        let orderedChunkOutputs = chunkOutputs.compactMap { $0 }

        guard var mergedTokens = orderedChunkOutputs.first else {
            return await manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSampleCount: totalSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if orderedChunkOutputs.count > 1 {
            let vocabulary = await manager.vocabulary
            let spliceSafeTokenIds = Self.spliceSafeTokenIds(vocabulary: vocabulary)
            let caseVariantIds = Self.caseVariantCanonicalIds(vocabulary: vocabulary)
            for chunk in orderedChunkOutputs.dropFirst() {
                mergedTokens = mergeChunks(
                    mergedTokens,
                    chunk,
                    spliceSafeTokenIds: spliceSafeTokenIds,
                    caseVariantIds: caseVariantIds
                )
            }
            if mergedTokens.count > 1 {
                mergedTokens.sort { $0.timestamp < $1.timestamp }
            }
            mergedTokens = collapseSeamWordDuplicates(mergedTokens, vocabulary: vocabulary)
        } else if mergedTokens.count > 1 {
            mergedTokens.sort { $0.timestamp < $1.timestamp }
        }

        // Post-merge repair pass re-decodes seam gaps the merger dropped
        // (issue #758) — see "Post-Merge Repair Passes" in
        // Documentation/ASR/LongTranscription.md.
        if orderedChunkOutputs.count > 1, mergedTokens.count > 1, await manager.seamGapRepair {
            let vocabulary = await manager.vocabulary
            let spliceSafeTokenIds = Self.spliceSafeTokenIds(vocabulary: vocabulary)
            let minGapSeconds = await manager.seamGapRepairMinGapSeconds
            let speechRmsThreshold = try adaptiveSpeechRmsThreshold()

            mergedTokens = try await repairSeamGaps(
                in: mergedTokens,
                using: workers[0],
                decoderLayers: decoderLayers,
                maxModelSamples: maxModelSamples,
                minGapSeconds: minGapSeconds,
                speechRmsThreshold: speechRmsThreshold,
                spliceSafeTokenIds: spliceSafeTokenIds,
                vocabulary: vocabulary,
                language: language
            )
        }

        let allTokens = mergedTokens.map { $0.token }
        let allTimestamps = mergedTokens.map { $0.timestamp }
        let allConfidences = mergedTokens.map { $0.confidence }
        let allDurations = mergedTokens.map { $0.duration }

        return await manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            confidences: allConfidences,
            tokenDurations: allDurations,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSampleCount: totalSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    private func makeWorkerPool(using manager: AsrManager, count: Int) async -> [AsrManager]? {
        guard count > 0 else { return nil }
        var workers: [AsrManager] = [manager]
        if count == 1 {
            return workers
        }
        for _ in 1..<count {
            guard let clone = await manager.makeWorkerClone() else {
                return nil
            }
            workers.append(clone)
        }
        logger.debug("ChunkProcessor using worker pool of size \(workers.count)")
        return workers
    }

    func readSamples(offset: Int, count: Int) throws -> [Float] {
        var buffer = [Float](repeating: 0, count: count)
        try buffer.withUnsafeMutableBufferPointer { pointer in
            try sampleSource.copySamples(into: pointer.baseAddress!, offset: offset, count: count)
        }
        return buffer
    }

    static func transcribeChunk(
        samples: [Float],
        contextSamples: Int,
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState,
        maxModelSamples: Int,
        language: Language? = nil,
        emitTokensAfterFrame: Int? = nil,
        initialTimeIndexOverride: Int? = nil
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int]) {
        guard !samples.isEmpty else { return ([], [], [], []) }

        let paddedChunk = manager.padAudioIfNeeded(samples, targetLength: maxModelSamples)

        // Calculate frame count for the ACTUAL audio (excluding prepended context)
        let actualAudioSamples = samples.count - contextSamples
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: actualAudioSamples)

        // Global frame offset is based on original chunkStart (not context-adjusted start)
        let globalFrameOffset = chunkStart / ASRConstants.samplesPerEncoderFrame

        // Context frame adjustment tells decoder to skip the prepended context frames
        let contextFrames = contextSamples / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: samples.count,  // Full length including context
            actualAudioFrames: actualFrameCount,  // Only actual audio frames (excluding context)
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrames,  // Skip context frames in decoder
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset,
            language: language,
            emitTokensAfterGlobalFrame: emitTokensAfterFrame,
            initialTimeIndexOverride: initialTimeIndexOverride
        )

        if hypothesis.isEmpty || encoderSequenceLength == 0 {
            return ([], [], [], [])
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, hypothesis.tokenDurations)
    }

    /// Token IDs that may safely start a seam splice: word-initial (`▁`) or
    /// punctuation-only pieces (issue #683). Nil for an empty vocabulary.
    static func spliceSafeTokenIds(vocabulary: [Int: String]) -> Set<Int>? {
        guard !vocabulary.isEmpty else { return nil }
        var ids = Set<Int>()
        for (id, piece) in vocabulary where isSpliceSafePiece(piece) {
            ids.insert(id)
        }
        return ids
    }

    /// Maps token IDs with a case-only twin to a shared canonical ID so the
    /// overlap matcher aligns `▁Meeting`/`▁meeting` at seams (issue #706) —
    /// see "Case-Folded Matching" in Documentation/ASR/LongTranscription.md.
    /// Nil for an empty vocabulary.
    static func caseVariantCanonicalIds(vocabulary: [Int: String]) -> [Int: Int]? {
        guard !vocabulary.isEmpty else { return nil }
        var groups: [String: [Int]] = [:]
        for (id, piece) in vocabulary {
            groups[piece.lowercased(), default: []].append(id)
        }
        var canon: [Int: Int] = [:]
        for (folded, ids) in groups where ids.count > 1 {
            // Lower-case variant is canonical so the collapse knows which
            // copy of a seam duplicate to keep.
            let canonical = ids.first { vocabulary[$0] == folded } ?? ids.min()!
            for id in ids { canon[id] = canonical }
        }
        return canon.isEmpty ? nil : canon
    }

    /// Drops an adjacent case-only duplicate of a seam word ("…have Have a…")
    /// left by a false sentence start, at word granularity (issue #706) — see
    /// "Case-Folded Matching" in Documentation/ASR/LongTranscription.md for
    /// the collapse conditions and what is deliberately left alone.
    func collapseSeamWordDuplicates(
        _ tokens: [TokenWindow],
        vocabulary: [Int: String]
    ) -> [TokenWindow] {
        guard !vocabulary.isEmpty, tokens.count > 1 else { return tokens }
        let overlapFrames = Int((overlapSeconds / ASRConstants.secondsPerEncoderFrame).rounded())

        func piece(_ id: Int) -> String { vocabulary[id] ?? "" }
        func startsWord(_ id: Int) -> Bool {
            let p = piece(id)
            return p.hasPrefix(ASRConstants.sentencePieceWordBoundary) || p.hasPrefix(" ")
        }

        struct Word {
            var tokens: [TokenWindow]
            var core: String
            var startTimestamp: Int
            var endsSentence: Bool
        }

        // Segment the token stream into words on word-initial pieces.
        var words: [Word] = []
        for token in tokens {
            if words.isEmpty || startsWord(token.token) {
                words.append(Word(tokens: [token], core: "", startTimestamp: token.timestamp, endsSentence: false))
            } else {
                words[words.count - 1].tokens.append(token)
            }
        }

        let strippable = CharacterSet.punctuationCharacters.union(.whitespaces)
        for index in words.indices {
            var text = ""
            for token in words[index].tokens {
                text += stripWordBoundaryPrefix(piece(token.token))
            }
            words[index].core = text.trimmingCharacters(in: strippable)
            if let last = text.last { words[index].endsSentence = ".?!:".contains(last) }
        }

        var keep = [Bool](repeating: true, count: words.count)
        var lastKept = -1
        for index in words.indices {
            guard lastKept >= 0 else {
                lastKept = index
                continue
            }
            let previous = words[lastKept]
            let current = words[index]
            let previousCore = previous.core
            let currentCore = current.core

            let isSeamDuplicate =
                !previousCore.isEmpty && !currentCore.isEmpty
                && previousCore != currentCore
                && previousCore.lowercased() == currentCore.lowercased()
                && currentCore.first?.isLetter == true
                && !previous.endsSentence
                && current.startTimestamp - previous.startTimestamp <= overlapFrames

            guard isSeamDuplicate else {
                lastKept = index
                continue
            }

            // Keep the lower-cased copy; if neither is lower-case keep the
            // earlier (left-context) one.
            if currentCore == currentCore.lowercased(), previousCore != previousCore.lowercased() {
                keep[lastKept] = false
                lastKept = index
            } else {
                keep[index] = false
            }
        }

        var result: [TokenWindow] = []
        result.reserveCapacity(tokens.count)
        for index in words.indices where keep[index] {
            result.append(contentsOf: words[index].tokens)
        }
        return result
    }

    /// A piece is splice-safe when decoding it right after another word does
    /// not glue two words together: it either starts a new word (`▁`/space
    /// prefix) or is pure punctuation/symbols.
    static func isSpliceSafePiece(_ piece: String) -> Bool {
        guard !piece.isEmpty else { return false }
        if isWordBoundary(piece) { return true }
        return piece.unicodeScalars.allSatisfy { scalar in
            CharacterSet.punctuationCharacters.contains(scalar)
                || CharacterSet.symbols.contains(scalar)
        }
    }

    func mergeChunks(
        _ left: [TokenWindow],
        _ right: [TokenWindow],
        spliceSafeTokenIds: Set<Int>? = nil,
        caseVariantIds: [Int: Int]? = nil
    ) -> [TokenWindow] {
        if left.isEmpty { return right }
        if right.isEmpty { return left }

        let frameDuration = ASRConstants.secondsPerEncoderFrame
        let overlapDuration = overlapSeconds
        let halfOverlapWindow = overlapDuration / 2

        func startTime(of token: TokenWindow) -> Double {
            Double(token.timestamp) * frameDuration
        }

        func endTime(of token: TokenWindow) -> Double {
            startTime(of: token) + frameDuration
        }

        let leftEndTime = endTime(of: left.last!)
        let rightStartTime = startTime(of: right.first!)

        if leftEndTime <= rightStartTime {
            return left + right
        }

        let overlapLeft: [IndexedToken] = left.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            let end = start + frameDuration
            guard end > rightStartTime - overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: end)
        }

        let overlapRight: [IndexedToken] = right.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            guard start < leftEndTime + overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: start + frameDuration)
        }

        guard overlapLeft.count >= 2 && overlapRight.count >= 2 else {
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration, spliceSafeTokenIds: spliceSafeTokenIds)
        }

        let minimumPairs = max(overlapLeft.count / 2, 1)

        // EXTRACTED: Contiguous matching using SequenceMatcher
        let timeTolerantMatcher: (IndexedToken, IndexedToken) -> Bool = { [self] l, r in
            tokensMatch(l, r, tolerance: halfOverlapWindow, caseVariantIds: caseVariantIds)
        }

        let contiguousMatches = SequenceMatcher.findContiguousMatches(
            left: overlapLeft,
            right: overlapRight,
            matcher: timeTolerantMatcher
        )

        // Convert SequenceMatch results to index pairs
        let contiguousPairs = contiguousMatches.map { ($0.leftStartIndex, $0.rightStartIndex) }

        if contiguousPairs.count >= minimumPairs {
            return mergeUsingMatches(
                matches: contiguousPairs,
                overlapLeft: overlapLeft,
                overlapRight: overlapRight,
                left: left,
                right: right,
                spliceSafeTokenIds: spliceSafeTokenIds
            )
        }

        // EXTRACTED: LCS fallback using SequenceMatcher
        let lcsMatches = SequenceMatcher.findLongestCommonSubsequence(
            left: overlapLeft,
            right: overlapRight,
            matcher: timeTolerantMatcher
        )

        guard !lcsMatches.isEmpty else {
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration, spliceSafeTokenIds: spliceSafeTokenIds)
        }

        // Map LCS matches directly to pairs (no consolidation)
        // mergeUsingMatches requires one pair per matched element to function correctly
        let lcsPairs = lcsMatches.map { ($0.leftStartIndex, $0.rightStartIndex) }

        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right,
            spliceSafeTokenIds: spliceSafeTokenIds
        )
    }

    private func tokensMatch(
        _ left: IndexedToken,
        _ right: IndexedToken,
        tolerance: Double,
        caseVariantIds: [Int: Int]? = nil
    ) -> Bool {
        guard tokenIdsMatch(left.token.token, right.token.token, caseVariantIds: caseVariantIds) else {
            return false
        }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    /// Token IDs match when equal or case-only variants of the same piece
    /// (issue #706), so a falsely capitalized seam word still anchors.
    private func tokenIdsMatch(_ left: Int, _ right: Int, caseVariantIds: [Int: Int]?) -> Bool {
        if left == right { return true }
        guard let caseVariantIds, let lhs = caseVariantIds[left], let rhs = caseVariantIds[right] else {
            return false
        }
        return lhs == rhs
    }

    private func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow],
        spliceSafeTokenIds: Set<Int>?
    ) -> [TokenWindow] {
        let leftIndices = matches.map { overlapLeft[$0.0].index }
        let rightIndices = matches.map { overlapRight[$0.1].index }

        var result: [TokenWindow] = []

        if let firstLeft = leftIndices.first, firstLeft > 0 {
            result.append(contentsOf: left[..<firstLeft])
        }

        for idx in 0..<matches.count {
            let leftIndex = leftIndices[idx]
            let rightIndex = rightIndices[idx]

            result.append(left[leftIndex])

            guard idx < matches.count - 1 else { continue }

            let nextLeftIndex = leftIndices[idx + 1]
            let nextRightIndex = rightIndices[idx + 1]

            let gapLeft = nextLeftIndex > leftIndex + 1 ? Array(left[(leftIndex + 1)..<nextLeftIndex]) : []
            let gapRight = nextRightIndex > rightIndex + 1 ? Array(right[(rightIndex + 1)..<nextRightIndex]) : []

            if gapRight.count > gapLeft.count {
                result.append(contentsOf: gapRight)
            } else {
                result.append(contentsOf: gapLeft)
            }
        }

        if let lastRight = rightIndices.last, lastRight + 1 < right.count {
            let tail = right[(lastRight + 1)...]
            if let safeIds = spliceSafeTokenIds,
                let firstTail = tail.first,
                !safeIds.contains(firstTail.token)
            {
                // Issue #683: the splice lands mid-word — re-splice at a word
                // boundary so exactly one window segments the seam word. See
                // "Word-Boundary Splice Repair" in LongTranscription.md.
                if let wordStart = wordInitialIndex(in: right, endingAt: lastRight, safeIds: safeIds),
                    popSeamWord(from: &result, safeIds: safeIds)
                {
                    // Right heard the seam word from its start — adopt its
                    // segmentation of the whole word.
                    result.append(contentsOf: right[wordStart...])
                } else {
                    // Right begins mid-word: left owns the seam word; resume
                    // right at its next word-initial piece instead of gluing.
                    if let lastLeft = leftIndices.last {
                        var cursor = lastLeft + 1
                        while cursor < left.count, !safeIds.contains(left[cursor].token) {
                            result.append(left[cursor])
                            cursor += 1
                        }
                    }
                    if let resume = tail.firstIndex(where: { safeIds.contains($0.token) }) {
                        result.append(contentsOf: tail[resume...])
                    } else {
                        // No word-initial piece in the tail: keep it verbatim
                        // — a possible glue beats dropping a word (PR #759).
                        result.append(contentsOf: tail)
                    }
                }
            } else {
                result.append(contentsOf: tail)
            }
        }

        return result
    }

    /// Index of the word-initial (or punctuation) piece starting the word
    /// that contains `anchor`, or nil when the stream begins mid-word.
    private func wordInitialIndex(
        in stream: [TokenWindow],
        endingAt anchor: Int,
        safeIds: Set<Int>
    ) -> Int? {
        var index = anchor
        while index >= 0 {
            if safeIds.contains(stream[index].token) { return index }
            index -= 1
        }
        return nil
    }

    /// Remove the trailing seam word from `result` so the right window's
    /// segmentation can replace it; false (untouched) when `result` has no
    /// word-initial piece. Unbounded scan — a piece cap false-negatives on
    /// long seam words (PR #759).
    private func popSeamWord(from result: inout [TokenWindow], safeIds: Set<Int>) -> Bool {
        var cursor = result.count - 1
        while cursor >= 0 {
            if safeIds.contains(result[cursor].token) {
                result.removeLast(result.count - cursor)
                return true
            }
            cursor -= 1
        }
        return false
    }

    private func mergeByMidpoint(
        left: [TokenWindow],
        right: [TokenWindow],
        leftEndTime: Double,
        rightStartTime: Double,
        frameDuration: Double,
        spliceSafeTokenIds: Set<Int>?
    ) -> [TokenWindow] {
        let cutoff = (leftEndTime + rightStartTime) / 2
        // Token streams are emitted in timestamp order, so the cutoff filter
        // is equivalent to a prefix/suffix split.
        var leftEnd = left.firstIndex { Double($0.timestamp) * frameDuration >= cutoff } ?? left.count
        var rightStart = right.firstIndex { Double($0.timestamp) * frameDuration >= cutoff } ?? right.count
        if let safeIds = spliceSafeTokenIds {
            // Issue #683: a pure time cutoff can split a word — let left
            // finish its word, drop orphaned continuation pieces from right.
            if leftEnd > 0 {
                while leftEnd < left.count, !safeIds.contains(left[leftEnd].token) {
                    leftEnd += 1
                }
            }
            // Adopt the advanced cutoff only if a splice-safe token exists
            // ahead — otherwise the whole right window would be discarded
            // (PR #759); fall back to the cutoff-based split.
            var scanIndex = rightStart
            while scanIndex < right.count, !safeIds.contains(right[scanIndex].token) {
                scanIndex += 1
            }
            if scanIndex < right.count {
                rightStart = scanIndex
            }
        }
        return Array(left[..<leftEnd]) + Array(right[rightStart...])
    }

    // MARK: - Seam-gap repair (issue #758)

    /// Probe budget per file — backstop against pathological inputs; sizing
    /// rationale in "Post-Merge Repair Pass" (LongTranscription.md).
    private var maxSeamGapRepairs: Int { 32 }

    // Adaptive speech-gate constants. Per-value rationale (dBFS anchors,
    // clamp structure, percentile choice) lives in the constants table under
    // "Adaptive Speech-Energy Gate" in Documentation/ASR/LongTranscription.md.

    /// Ceiling (~-42 dBFS): the pre-adaptive fixed gate.
    static let speechRmsCeiling: Float = 0.008

    /// Floor (~-66 dBFS): above dither/room tone, below quiet speech.
    static let speechRmsFloor: Float = 0.0005

    /// ~-10.5 dB under the speech reference.
    static let speechRmsReferenceScale: Float = 0.3

    /// Reference percentile over non-digital-silence frames.
    static let speechRmsReferencePercentile = 0.75

    /// Speech-energy threshold scaled to the recording's own level — an
    /// absolute gate can never fire on quiet audio (issue #747).
    static func adaptiveSpeechRmsThreshold(referenceRms: Float, floor: Float, ceiling: Float) -> Float {
        min(ceiling, max(floor, referenceRms * speechRmsReferenceScale))
    }

    /// Recording-level threshold from the reference percentile of per-frame
    /// RMS. All-zero frames are excluded — digital silence is no recording
    /// and drags the percentile to the floor on silence-heavy files.
    private func adaptiveSpeechRmsThreshold() throws -> Float {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        var frameRms: [Float] = []
        frameRms.reserveCapacity(totalSamples / frameSamples + 1)
        var offset = 0
        while offset + frameSamples <= totalSamples {
            let samples = try readSamples(offset: offset, count: frameSamples)
            var sum: Float = 0
            for sample in samples {
                sum += sample * sample
            }
            if sum > 0 {
                frameRms.append(sqrt(sum / Float(samples.count)))
            }
            offset += frameSamples
        }
        guard !frameRms.isEmpty else { return Self.speechRmsCeiling }
        frameRms.sort()
        let referenceIndex = min(
            frameRms.count - 1,
            Int(Double(frameRms.count) * Self.speechRmsReferencePercentile))
        let referenceRms = frameRms[referenceIndex]
        return Self.adaptiveSpeechRmsThreshold(
            referenceRms: referenceRms, floor: Self.speechRmsFloor, ceiling: Self.speechRmsCeiling)
    }

    /// Minimum cumulative speech-like audio inside a gap before it is
    /// probed. Genuine pauses with a stray cough stay untouched.
    private var seamGapMinSpeechSeconds: Double { 0.5 }

    /// A piece consisting solely of punctuation/symbol scalars (e.g. the
    /// "." in "else." = ▁else + .). Skipped when resolving the word bordering
    /// a gap for edge dedupe.
    static func isPunctuationOnlyPiece(_ id: Int, vocabulary: [Int: String]) -> Bool {
        guard let piece = vocabulary[id], !piece.isEmpty else { return false }
        return piece.unicodeScalars.allSatisfy { scalar in
            CharacterSet.punctuationCharacters.contains(scalar)
                || CharacterSet.symbols.contains(scalar)
        }
    }

    /// The token of the word bordering a gap, walking past punctuation-only
    /// pieces (`step` -1 walks left from the token before the gap, +1 walks
    /// right from the token after it).
    static func wordNeighbor(
        in stream: [TokenWindow],
        from index: Int,
        step: Int,
        vocabulary: [Int: String]
    ) -> TokenWindow {
        var neighborIndex = index
        while neighborIndex + step >= 0,
            neighborIndex + step < stream.count,
            isPunctuationOnlyPiece(stream[neighborIndex].token, vocabulary: vocabulary)
        {
            neighborIndex += step
        }
        return stream[neighborIndex]
    }

    /// Filter a probe window's tokens to the spliceable run: strictly
    /// in-gap, word-initial start, edge dedupe of re-heard border words —
    /// rules and tolerances in "Post-Merge Repair Pass"
    /// (LongTranscription.md).
    static func spliceCandidate(
        windowTokens: [Int],
        windowTimestamps: [Int],
        windowConfidences: [Float],
        windowDurations: [Int],
        gapStartFrame: Int,
        gapEndFrame: Int,
        leadNeighbor: TokenWindow,
        tailNeighbor: TokenWindow,
        spliceSafeTokenIds: Set<Int>?,
        vocabulary: [Int: String]
    ) -> [TokenWindow] {
        let edgeToleranceFrames = 6
        func samePiece(_ a: Int, _ b: Int) -> Bool {
            if a == b { return true }
            guard let pieceA = vocabulary[a], let pieceB = vocabulary[b] else { return false }
            return pieceA.lowercased() == pieceB.lowercased()
        }

        var candidate: [TokenWindow] = []
        for tokenIndex in 0..<windowTokens.count {
            let timestamp = windowTimestamps[tokenIndex]
            guard timestamp > gapStartFrame, timestamp < gapEndFrame - 1 else { continue }
            candidate.append(
                (
                    token: windowTokens[tokenIndex],
                    timestamp: timestamp,
                    confidence: windowConfidences[tokenIndex],
                    duration: windowDurations[tokenIndex]
                )
            )
        }

        if let safeIds = spliceSafeTokenIds {
            while let first = candidate.first, !safeIds.contains(first.token) {
                candidate.removeFirst()
            }
        }

        while let first = candidate.first,
            samePiece(first.token, leadNeighbor.token),
            abs(first.timestamp - leadNeighbor.timestamp) <= edgeToleranceFrames
        {
            candidate.removeFirst()
        }
        while let last = candidate.last,
            samePiece(last.token, tailNeighbor.token),
            abs(tailNeighbor.timestamp - last.timestamp) <= edgeToleranceFrames
        {
            candidate.removeLast()
        }
        // Removing an edge token can expose continuation pieces (or orphaned
        // punctuation) at the head — re-trim.
        if let safeIds = spliceSafeTokenIds {
            while let first = candidate.first,
                !safeIds.contains(first.token) || isPunctuationOnlyPiece(first.token, vocabulary: vocabulary)
            {
                candidate.removeFirst()
            }
        }

        return candidate
    }

    #if DEBUG
    internal func speechLikeSecondsForTesting(from startSample: Int, to endSample: Int) throws -> Double {
        try speechLikeSeconds(from: startSample, to: endSample, threshold: adaptiveSpeechRmsThreshold())
    }

    internal func adaptiveSpeechRmsThresholdForTesting() throws -> Float {
        try adaptiveSpeechRmsThreshold()
    }
    #endif

    /// Re-decode inter-token gaps that plausibly contain dropped speech with
    /// a fresh seam-free window and splice in only in-gap tokens (issue
    /// #758). Genuine silence splices nothing. See "Post-Merge Repair Pass"
    /// in Documentation/ASR/LongTranscription.md.
    private func repairSeamGaps(
        in tokens: [TokenWindow],
        using manager: AsrManager,
        decoderLayers: Int,
        maxModelSamples: Int,
        minGapSeconds: Double,
        speechRmsThreshold: Float,
        spliceSafeTokenIds: Set<Int>?,
        vocabulary: [Int: String],
        language: Language?
    ) async throws -> [TokenWindow] {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let frameDuration = ASRConstants.secondsPerEncoderFrame
        let minGapFrames = max(2, Int(minGapSeconds / frameDuration))
        // Same frame-aligned usable window size the chunker uses (no context
        // reservation — the repair window is decoded standalone).
        let windowSamples = max(
            frameSamples,
            (maxModelSamples - ASRConstants.melHopSize) / frameSamples * frameSamples
        )

        var working = tokens
        var probes = 0
        var probedGapStarts = Set<Int>()
        // Iterate so a partial recovery's residual gap (start has moved)
        // gets its own probe; yielded-nothing gaps are memoized and skipped.
        for _ in 0..<3 {
            var inserts: [TokenWindow] = []

            for index in 0..<(working.count - 1) {
                guard probes < maxSeamGapRepairs else { break }

                let current = working[index]
                let next = working[index + 1]
                // Conservative end of the current token: its decoded duration
                // when present, else one frame (mirrors mergeChunks).
                let gapStartFrame = current.timestamp + max(1, current.duration)
                let gapEndFrame = next.timestamp
                guard gapEndFrame - gapStartFrame >= minGapFrames else { continue }
                guard !probedGapStarts.contains(gapStartFrame) else { continue }

                let gapStartSample = gapStartFrame * frameSamples
                let gapEndSample = min(gapEndFrame * frameSamples, totalSamples)
                guard gapEndSample > gapStartSample else { continue }

                let speechSeconds = try speechLikeSeconds(
                    from: gapStartSample, to: gapEndSample, threshold: speechRmsThreshold)
                guard speechSeconds >= seamGapMinSpeechSeconds else { continue }
                probedGapStarts.insert(gapStartFrame)
                probes += 1

                // Cold-start AT the gap first (replaying the pre-gap noise
                // can re-blank), gap-centred fallback — see "Placement
                // matters" in LongTranscription.md.
                let gapCenterSample = (gapStartSample + gapEndSample) / 2
                let placements = [gapStartSample, gapCenterSample - windowSamples / 2]
                var recovered: [TokenWindow] = []

                for placement in placements {
                    var windowStart = max(0, min(placement, totalSamples - windowSamples))
                    windowStart = windowStart / frameSamples * frameSamples
                    let windowEnd = min(windowStart + windowSamples, totalSamples)
                    guard windowEnd > windowStart else { continue }

                    var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
                    decoderState.reset()
                    let windowAudio = try readSamples(offset: windowStart, count: windowEnd - windowStart)
                    let (windowTokens, windowTimestamps, windowConfidences, windowDurations) =
                        try await Self.transcribeChunk(
                            samples: windowAudio,
                            contextSamples: 0,
                            chunkStart: windowStart,
                            isLastChunk: windowEnd >= totalSamples,
                            using: manager,
                            decoderState: &decoderState,
                            maxModelSamples: maxModelSamples,
                            language: language
                        )

                    guard windowTokens.count == windowTimestamps.count,
                        windowTokens.count == windowConfidences.count
                    else { continue }
                    let durations =
                        windowDurations.count == windowTokens.count
                        ? windowDurations : Array(repeating: 0, count: windowTokens.count)

                    let candidate = Self.spliceCandidate(
                        windowTokens: windowTokens,
                        windowTimestamps: windowTimestamps,
                        windowConfidences: windowConfidences,
                        windowDurations: durations,
                        gapStartFrame: gapStartFrame,
                        gapEndFrame: gapEndFrame,
                        leadNeighbor: Self.wordNeighbor(in: working, from: index, step: -1, vocabulary: vocabulary),
                        tailNeighbor: Self.wordNeighbor(in: working, from: index + 1, step: 1, vocabulary: vocabulary),
                        spliceSafeTokenIds: spliceSafeTokenIds,
                        vocabulary: vocabulary
                    )

                    if !candidate.isEmpty {
                        recovered = candidate
                        break
                    }
                }

                guard !recovered.isEmpty else { continue }
                logger.info(
                    "Seam-gap repair: recovered \(recovered.count) tokens in "
                        + String(format: "%.2fs", Double(gapStartFrame) * frameDuration) + "–"
                        + String(format: "%.2fs", Double(gapEndFrame) * frameDuration) + " gap"
                )
                inserts.append(contentsOf: recovered)
            }

            guard !inserts.isEmpty else { break }
            working.append(contentsOf: inserts)
            working.sort { $0.timestamp < $1.timestamp }
        }

        return working
    }

    /// Cumulative duration of speech-like audio (per-frame RMS above the
    /// given adaptive threshold) between two sample offsets.
    private func speechLikeSeconds(from startSample: Int, to endSample: Int, threshold: Float) throws -> Double {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        var speechFrames = 0
        var offset = startSample
        while offset + frameSamples <= endSample {
            let samples = try readSamples(offset: offset, count: frameSamples)
            var sum: Float = 0
            for sample in samples {
                sum += sample * sample
            }
            if sqrt(sum / Float(samples.count)) > threshold {
                speechFrames += 1
            }
            offset += frameSamples
        }
        return Double(speechFrames) * ASRConstants.secondsPerEncoderFrame
    }
}
