import Foundation

struct ChunkProcessor {
    let sampleSource: AudioSampleSource
    let totalSamples: Int

    private let logger = AppLogger(category: "ChunkProcessor")
    private typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float, duration: Int)
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
    private struct ChunkStartDecision {
        let start: Int
        let useWarmupPrefix: Bool
    }

    // Stateless chunking aligned with CoreML reference:
    // - process ~14.96s of audio per window (frame-aligned) to stay under encoder limit
    // - 2.0s overlap (frame-aligned) to give the decoder slack when merging windows
    private let overlapSeconds: Double = 2.0

    /// Context samples prepended from previous chunk for mel spectrogram stability (80ms = 1 encoder frame).
    /// The FastConformer encoder's depthwise convolutions need left context for stable output.
    /// Without this, the first frames of a chunk may produce features that cause all-blank predictions.
    ///
    /// Issue #594: on `parakeet-tdt-0.6b-v3-coreml` multilingual long-form
    /// audio this prepend can shift the encoder's first-frame distribution
    /// enough to make the SOS-primed decoder drift to its English-biased prior.
    /// Callers can opt out via `ASRConfig.melChunkContext = false` to
    /// use the v3/no-mel boundary warmup path below.
    private let melContextSamples: Int = ASRConstants.samplesPerEncoderFrame  // 1280 samples = 80ms

    /// Short real-audio warmup prefix for v3/no-mel chunks. The prefix is
    /// decoded to condition the predictor on real left audio, but prefix
    /// tokens are suppressed before chunk merging.
    private let noMelWarmupPrefixFrames: Int = 7

    /// Short-circuit threshold for the dual-decode arbitration path. When
    /// path A (regular stride / no warmup) emits a chunk whose mean
    /// per-token joint log-probability is at or above this value, path B
    /// (silence-aligned + warmup) is not run for that chunk. Tuned so
    /// "clean" chunks skip the second decode while "uncertain" chunks
    /// (typical of #594-style seam drift or G2-suppression content drop)
    /// fall through to the dual-decode comparison.
    ///
    /// 0.90 was picked empirically: typical TDT v3 per-token confidence
    /// on clean French/English/Spanish ranges 0.93–0.99, and chunks with
    /// any low-confidence patch (BPE artifact, OOD seam, suppressed
    /// content) average lower. Adjusting this knob trades runtime for
    /// quality.
    private let dualDecodeConfidenceShortCircuit: Float = 0.90

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

    private func chunkLayout(
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

    private func regularChunkStarts(strideSamples: Int) -> [ChunkStartDecision] {
        var starts = [ChunkStartDecision(start: 0, useWarmupPrefix: false)]
        var start = strideSamples
        while start < totalSamples {
            starts.append(ChunkStartDecision(start: start, useWarmupPrefix: false))
            start += strideSamples
        }
        return starts
    }

    private func silenceAlignedChunkStarts(
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
        right: [(token: Int, timestamp: Int, confidence: Float, duration: Int)]
    ) -> [(token: Int, timestamp: Int, confidence: Float, duration: Int)] {
        mergeChunks(left, right)
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
                let warmupSamples =
                    chunkIndex > 0 && chunkDecision.useWarmupPrefix
                    ? min(warmupPrefixSamples, chunkStart) : 0
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

                // In the default path, contextSamples means mel/STFT context
                // and is skipped by the decoder. In v3/no-mel mode, the
                // warmup prefix is decoded from frame 0 and only its emitted
                // tokens are suppressed.
                let contextSamples = warmupSamples > 0 ? 0 : (chunkIndex > 0 ? melContextSamples : 0)
                let contextStart = chunkStart - max(warmupSamples, contextSamples)
                let chunkLengthWithContext = chunkEnd - contextStart
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
                audioSamples: [],
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if orderedChunkOutputs.count > 1 {
            for chunk in orderedChunkOutputs.dropFirst() {
                mergedTokens = mergeChunks(mergedTokens, chunk)
            }
        }

        if mergedTokens.count > 1 {
            mergedTokens.sort { $0.timestamp < $1.timestamp }
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
            audioSamples: [],
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    /// Dual-decode arbitration path. For each chunk index ≥ 1, decode the
    /// chunk with the regular stride/no-warmup ("path A", the simple PR
    /// #596 shape) first. If path A produces a chunk whose mean per-token
    /// joint log-probability is above `dualDecodeConfidenceShortCircuit`
    /// (typical for well-decoded seams), path A is accepted without
    /// running path B at all — keeping cost at ~1× for the easy majority
    /// of chunks. Otherwise path B (silence-aligned start + 7-frame
    /// real-audio warmup prefix, the PR #604 shape) is decoded as well
    /// and the two hypotheses are arbitrated by mean confidence: whichever
    /// chunk scores higher becomes the chunk's contribution before the
    /// existing LCS+midpoint merger runs.
    ///
    /// Chunk 0 is decoded once with the regular shape (no warmup is ever
    /// applied to chunk 0 in either path, so the two would be identical).
    ///
    /// Cost: ~1× per-chunk encoder+decoder for high-confidence chunks,
    /// ~2× for chunks whose path-A confidence falls under the short-
    /// circuit threshold. Empirically (over the primary 20 + 16 broad-
    /// stress fixtures) the average is closer to 1.1–1.3× than to 2×.
    /// Mechanism is language-agnostic — arbitration uses raw acoustic
    /// confidence with no text inspection, no vocabulary/script/token
    /// filtering, and no language hints.
    private func processWithDualDecodeArbitration(
        using manager: AsrManager,
        workers: [AsrManager],
        decoderLayers: Int,
        maxModelSamples: Int,
        modelVersion: AsrModelVersion?,
        startTime: Date,
        progressHandler: ((Double) async -> Void)?,
        language: Language?
    ) async throws -> ASRResult {
        // Layout is computed in the no-mel shape because both decode paths
        // run with `melChunkContext == false` semantics; the only difference
        // is whether silence-alignment + warmup-prefix is active for this
        // chunk's decode.
        let layout = chunkLayout(melChunkContext: false, modelVersion: modelVersion)
        let warmupPrefixSamples = layout.warmupPrefixSamples
        let chunkSamples = layout.chunkSamples
        let strideSamples = layout.strideSamples

        let regularStarts = regularChunkStarts(strideSamples: strideSamples)
        let silenceAlignedStarts = try silenceAlignedChunkStarts(
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            canUseWarmupPrefix: warmupPrefixSamples > 0
        )

        let chunkCount = max(regularStarts.count, silenceAlignedStarts.count)
        var chunkOutputs: [[TokenWindow]] = []
        chunkOutputs.reserveCapacity(chunkCount)

        let worker = workers.first ?? manager

        for chunkIndex in 0..<chunkCount {
            try Task.checkCancellation()

            let regularStart = chunkIndex < regularStarts.count
                ? regularStarts[chunkIndex].start
                : min(regularStarts.last?.start ?? 0 + strideSamples, totalSamples)
            let silenceDecision = chunkIndex < silenceAlignedStarts.count
                ? silenceAlignedStarts[chunkIndex]
                : ChunkStartDecision(
                    start: min(silenceAlignedStarts.last?.start ?? 0 + strideSamples, totalSamples),
                    useWarmupPrefix: false
                )

            // Path A: regular (no warmup prefix, no silence-alignment).
            // Always decoded; this is the simple PR #596 shape and the
            // fallback when path B's silence-aligned start would coincide.
            let regularTokens = try await decodeOneChunk(
                chunkStart: regularStart,
                chunkIndex: chunkIndex,
                chunkSamples: chunkSamples,
                warmupSamples: 0,
                using: worker,
                decoderLayers: decoderLayers,
                maxModelSamples: maxModelSamples,
                language: language
            )
            let regularConf = meanConfidence(regularTokens)

            // Path B: silence-aligned start; warmup prefix when the
            // chunkDecision says so (and we are not on chunk 0). Compute
            // the would-be warmup so we can decide whether path B is
            // structurally distinct from path A.
            let warmupSamplesForB =
                chunkIndex > 0 && silenceDecision.useWarmupPrefix
                ? min(warmupPrefixSamples, silenceDecision.start) : 0
            let pathBCoincident =
                chunkIndex == 0
                || (silenceDecision.start == regularStart && warmupSamplesForB == 0)

            // Short-circuit: skip path B when path A is already confident,
            // unless path B would have been the same chunk anyway (in which
            // case there is nothing to compare against). Empirically this
            // keeps cost near 1× for clean seams.
            let skipPathB =
                pathBCoincident
                || (regularTokens.count > 0 && regularConf >= dualDecodeConfidenceShortCircuit)

            let chosen: [TokenWindow]
            if skipPathB {
                chosen = regularTokens
            } else {
                let silenceTokens = try await decodeOneChunk(
                    chunkStart: silenceDecision.start,
                    chunkIndex: chunkIndex,
                    chunkSamples: chunkSamples,
                    warmupSamples: warmupSamplesForB,
                    using: worker,
                    decoderLayers: decoderLayers,
                    maxModelSamples: maxModelSamples,
                    language: language
                )
                let silenceConf = meanConfidence(silenceTokens)
                chosen = silenceConf > regularConf ? silenceTokens : regularTokens
            }
            chunkOutputs.append(chosen)

            if let progressHandler {
                let progress = min(1.0, Double(chunkIndex + 1) / Double(chunkCount))
                await progressHandler(progress)
            }
        }

        guard var mergedTokens = chunkOutputs.first else {
            return await manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: [],
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if chunkOutputs.count > 1 {
            for chunk in chunkOutputs.dropFirst() {
                mergedTokens = mergeChunks(mergedTokens, chunk)
            }
        }

        if mergedTokens.count > 1 {
            mergedTokens.sort { $0.timestamp < $1.timestamp }
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
            encoderSequenceLength: 0,
            audioSamples: [],
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    /// Decode a single chunk under the given start + warmup parameters.
    /// Mirrors the per-chunk decode inside the main `process()` loop, but
    /// without the task-group / worker-pool plumbing so it can be called
    /// twice per chunk index from the dual-decode arbitration path.
    private func decodeOneChunk(
        chunkStart: Int,
        chunkIndex: Int,
        chunkSamples: Int,
        warmupSamples: Int,
        using manager: AsrManager,
        decoderLayers: Int,
        maxModelSamples: Int,
        language: Language?
    ) async throws -> [TokenWindow] {
        let visibleChunkSamples = max(
            ASRConstants.samplesPerEncoderFrame,
            chunkSamples - warmupSamples
        )
        let candidateEnd = chunkStart + visibleChunkSamples
        let isLastChunk = candidateEnd >= totalSamples
        let chunkEnd = isLastChunk ? totalSamples : candidateEnd

        if chunkEnd <= chunkStart {
            return []
        }

        // In dual-decode mode `melChunkContext` is `false` for both paths,
        // so the regular path has no mel-context prepend and the warmup
        // path uses the real-audio prefix instead. Therefore
        // `contextSamples` is always 0 here.
        let contextSamples = 0
        let contextStart = chunkStart - warmupSamples
        let chunkLengthWithContext = chunkEnd - contextStart
        let chunkSamplesArray = try readSamples(offset: contextStart, count: chunkLengthWithContext)
        let emitTokensAfterFrame =
            warmupSamples > 0 ? chunkStart / ASRConstants.samplesPerEncoderFrame : nil
        let chunkStartOffset = warmupSamples > 0 ? contextStart : chunkStart

        var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
        decoderState.reset()

        let (windowTokens, windowTimestamps, windowConfidences, windowDurations) =
            try await Self.transcribeChunk(
                samples: chunkSamplesArray,
                contextSamples: contextSamples,
                chunkStart: chunkStartOffset,
                isLastChunk: isLastChunk,
                using: manager,
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

        return zip(
            zip(zip(windowTokens, windowTimestamps), windowConfidences), durations
        ).map {
            (token: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
        }
    }

    /// Mean per-token confidence over emitted (non-blank) tokens.
    /// Empty chunks score `-.infinity` so they always lose against any
    /// non-empty chunk.
    private func meanConfidence(_ tokens: [TokenWindow]) -> Float {
        guard !tokens.isEmpty else { return -.infinity }
        var sum: Float = 0
        for t in tokens {
            sum += t.confidence
        }
        return sum / Float(tokens.count)
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

    private func readSamples(offset: Int, count: Int) throws -> [Float] {
        var buffer = [Float](repeating: 0, count: count)
        try buffer.withUnsafeMutableBufferPointer { pointer in
            try sampleSource.copySamples(into: pointer.baseAddress!, offset: offset, count: count)
        }
        return buffer
    }

    private static func transcribeChunk(
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

    private func mergeChunks(
        _ left: [TokenWindow],
        _ right: [TokenWindow]
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
                frameDuration: frameDuration)
        }

        let minimumPairs = max(overlapLeft.count / 2, 1)

        // EXTRACTED: Contiguous matching using SequenceMatcher
        let timeTolerantMatcher: (IndexedToken, IndexedToken) -> Bool = { [self] l, r in
            tokensMatch(l, r, tolerance: halfOverlapWindow)
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
                right: right
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
                frameDuration: frameDuration)
        }

        // Map LCS matches directly to pairs (no consolidation)
        // mergeUsingMatches requires one pair per matched element to function correctly
        let lcsPairs = lcsMatches.map { ($0.leftStartIndex, $0.rightStartIndex) }

        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right
        )
    }

    private func tokensMatch(_ left: IndexedToken, _ right: IndexedToken, tolerance: Double) -> Bool {
        guard left.token.token == right.token.token else { return false }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    private func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow]
    ) -> [TokenWindow] {
        let leftIndices = matches.map { overlapLeft[$0.0].index }
        let rightIndices = matches.map { overlapRight[$0.1].index }

        var result: [TokenWindow] = []

        if let firstLeft = leftIndices.first, let firstRight = rightIndices.first {
            let leftPrefix = firstLeft > 0 ? Array(left[..<firstLeft]) : []
            let rightPrefix = firstRight > 0 ? Array(right[..<firstRight]) : []

            if let rightStartTimestamp = rightPrefix.first?.timestamp ?? right.first?.timestamp {
                let stableLeftEnd =
                    leftPrefix.firstIndex {
                        $0.timestamp + max($0.duration, 1) > rightStartTimestamp
                    }
                    ?? leftPrefix.count
                result.append(contentsOf: leftPrefix[..<stableLeftEnd])
                result.append(
                    contentsOf: preferredLeadingGap(
                        left: Array(leftPrefix[stableLeftEnd...]),
                        right: rightPrefix
                    )
                )
            } else {
                result.append(contentsOf: leftPrefix)
            }
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

            result.append(contentsOf: preferredGap(left: gapLeft, right: gapRight))
        }

        if let lastRight = rightIndices.last, lastRight + 1 < right.count {
            result.append(contentsOf: right[(lastRight + 1)...])
        }

        return result
    }

    private func preferredGap(left gapLeft: [TokenWindow], right gapRight: [TokenWindow]) -> [TokenWindow] {
        if gapRight.count > gapLeft.count { return gapRight }
        if gapLeft.count > gapRight.count { return gapLeft }
        if gapLeft.isEmpty { return gapRight }

        let leftConfidence = gapLeft.reduce(Float(0)) { $0 + $1.confidence }
        let rightConfidence = gapRight.reduce(Float(0)) { $0 + $1.confidence }
        return rightConfidence > leftConfidence ? gapRight : gapLeft
    }

    private func preferredLeadingGap(left gapLeft: [TokenWindow], right gapRight: [TokenWindow]) -> [TokenWindow] {
        if gapLeft.isEmpty { return gapRight }
        if gapRight.isEmpty { return gapLeft }

        let leftConfidence = gapLeft.reduce(Float(0)) { $0 + $1.confidence }
        let rightConfidence = gapRight.reduce(Float(0)) { $0 + $1.confidence }

        if gapLeft.count == gapRight.count {
            let newerChunkSlack = Float(gapLeft.count) * 0.12
            return rightConfidence + newerChunkSlack >= leftConfidence ? gapRight : gapLeft
        }

        let leftAverage = leftConfidence / Float(gapLeft.count)
        let rightAverage = rightConfidence / Float(gapRight.count)
        return rightAverage > leftAverage + 0.20 ? gapRight : gapLeft
    }

    private func mergeByMidpoint(
        left: [TokenWindow],
        right: [TokenWindow],
        leftEndTime: Double,
        rightStartTime: Double,
        frameDuration: Double
    ) -> [TokenWindow] {
        let cutoff = (leftEndTime + rightStartTime) / 2
        let trimmedLeft = left.filter { Double($0.timestamp) * frameDuration < cutoff }
        let trimmedRight = right.filter { Double($0.timestamp) * frameDuration >= cutoff }
        return trimmedLeft + trimmedRight
    }
}
