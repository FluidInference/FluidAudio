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
    private let noMelWarmupPrefixFrames: Int = 0

    /// Number of non-first chunks to dual-decode as a probe before
    /// committing to a single primary path for the remainder of the file.
    /// 2 is the empirical sweet spot: ~30s of dual-decoded audio is
    /// enough for one path's mean confidence to dominate cleanly while
    /// keeping amortized cost ≤1.2× a single decode on typical files.
    private let probeChunkCount: Int = 3

    /// Minimum mean per-token confidence margin path B must beat path A
    /// by, in the probe, to commit the file to path B. Below this margin
    /// the file commits to path A (the content-safe default). Empirical
    /// separation observed in probe traces: rare cases where path B is
    /// the correct global choice (path A suffers a cross-script burst
    /// against a wrong-script language prior) show margins near 5%;
    /// cases where the two paths agree on content show margins of
    /// 0.1–0.7%. A 1% gate cleanly separates them.
    private let pathBSwitchMargin: Float = 0.01

    /// Maximum acceptable ratio of path-B-emitted tokens to path-A-emitted
    /// tokens (over the full probe) below which path B is suspected of
    /// content suppression and the file commits to path A regardless of
    /// any confidence advantage. Known failure mode: warmup-driven priming
    /// suppressing must-keep content drops the probe ratio near 0.45;
    /// healthy probes sit in [0.7, 1.05]. 0.6 sits between.
    private let pathBSuppressionRatio: Float = 0.6

    /// Minimum ratio of path-C-emitted tokens to path-A-emitted tokens
    /// over the probe at which path C is selected as a content-recovery
    /// fallback. Path C uses fixed-stride boundaries (no silence-snap)
    /// and recovers seam tokens that the silence-snap can move outside
    /// any single chunk's decoder span. The recovery is real when path C
    /// reliably emits more tokens than path A over the probe; below this
    /// ratio the two paths effectively agree on content and the safer
    /// silence-aligned default wins to avoid mid-word starts that bias
    /// the decoder toward an English-prior wrong-language burst on
    /// non-English audio.
    private let pathCContentRatio: Float = 1.01

    /// Maximum acceptable confidence headroom path C can have above path
    /// A before path C is suspected of being a wrong-language drift.
    /// Cross-language drift typically shows artificially high per-token
    /// confidence because the decoder is settling into its default
    /// (English) prior; real seam recovery shows similar confidence to
    /// path A.
    private let pathCDriftConfidenceCeiling: Float = 0.03

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

    /// Per-file probe arbitration over a single, shared set of silence-
    /// aligned chunk boundaries. Both paths consider the *same* visible
    /// chunk range, and the file commits to ONE path globally based on
    /// dual-decoded probe chunks — so no chunk inside a single file is
    /// ever stitched across paths, and the LCS+midpoint merger never
    /// sees inter-path BPE divergence at its overlap windows.
    ///
    /// Path A: silence-aligned start, NO warmup prefix. The "G1-only"
    /// shape from A3-bisect — verified safe for content preservation
    /// on user2 (`journaliste de l'AFP`, `19 membres`, `Affaires
    /// étrangères néerlandais` all survive because no warmup is biasing
    /// the predictor away from them).
    ///
    /// Path B: silence-aligned start, WITH the 7-frame real-audio warmup
    /// prefix when the silence-alignment heuristic asked for it. The
    /// "G1+G2" shape (PR #604) — load-bearing for cross-script bias
    /// avoidance on Slovenian and similar low-resource Latin-script
    /// languages where SOS-priming alone reverts to a wrong-script prior.
    ///
    /// Algorithm:
    ///   1. Decode chunk 0 with path A (warmup never applies to chunk 0).
    ///   2. Probe phase: for the next `probeChunkCount` chunks, decode
    ///      both paths. Track per-path mean per-token confidence across
    ///      the probe. Chunks where the heuristic declined warmup count
    ///      structurally as identical (path B reuses path A's tokens).
    ///   3. Decide globally: whichever path has higher mean confidence
    ///      across the probe becomes the primary path for the remainder
    ///      of the file. Ties (or all-identical probes) go to path A.
    ///   4. Decode remaining chunks single-path (the chosen primary).
    ///
    /// This eliminates the per-chunk arbitration's inter-path stitching
    /// artifacts (cf. T1/T2/T3 buenaventura: `siem Spre`, `mir Mandoir`,
    /// `Qu díinasce`, `nue Nvoue`, etc.) at the cost of giving up the
    /// ability to recover one bad chunk mid-file via the other path.
    /// In practice, the fixtures whose path preference flips mid-file
    /// are exceptionally rare; the dominant per-file regression modes
    /// (user2 content suppression; sl_si Cyrillic burst) are global
    /// properties of the file's acoustic distribution that the probe
    /// already captures.
    ///
    /// Mechanism is language-agnostic — the probe uses raw acoustic
    /// confidence only, with no text inspection, no vocabulary/script/
    /// token filtering, and no language hints.
    ///
    /// Cost: chunk 0 is 1×; probe chunks are up to 2× (1× when path B
    /// would have been identical to path A); remaining chunks are 1×.
    /// Amortized on a 10-chunk file ≈ 1.2×; on a 20-chunk file ≈ 1.1×.
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
        // between path A and path B for a given chunk is the warmup prefix.
        let layout = chunkLayout(melChunkContext: false, modelVersion: modelVersion)
        let chunkSamples = layout.chunkSamples
        let strideSamples = layout.strideSamples
        // In dual-decode mode `noMelWarmupPrefixFrames` is 0, so
        // layout.warmupPrefixSamples is 0. Path B retains warmup capability
        // via an explicit per-path size (7 encoder frames; matches the
        // upstream warmup default for the non-arbitrated no-mel path).
        let pathBWarmupSamples = 7 * ASRConstants.samplesPerEncoderFrame

        // Each path uses its OWN start system:
        //
        //   Path A: silence-aligned boundaries, no warmup prefix. The
        //   content-preserving default — silence-anchored seams avoid
        //   mid-word cuts that would seed an English-prior wrong-language
        //   burst on non-English audio, and skipping warmup avoids the
        //   priming that can suppress must-keep content on continuous
        //   narration.
        //
        //   Path B: silence-aligned boundaries WITH warmup capability.
        //   Load-bearing when the SOS-primed decoder reverts to a
        //   wrong-script prior on cross-script audio; the warmup
        //   conditions the predictor on real left audio before emitting.
        //
        //   Path C: regular fixed-stride boundaries, no warmup prefix.
        //   Seam-recovery option for audio where the silence-snap moves
        //   the boundary past a syllable that no single chunk's decoder
        //   then sees with right-context. Selected only when its probe
        //   emits materially more tokens than path A AND its mean
        //   per-token confidence is comparable to path A (a strongly
        //   higher confidence with similar token count signals a
        //   wrong-language drift on non-English audio, not a content
        //   recovery, and the suspect-drift gate rejects it).
        //
        // Per-file commit ensures the chosen path's boundaries are used
        // throughout the file, so the merger never sees mixed-boundary
        // chunks within a single transcript.
        let pathAStartsDecisions = try silenceAlignedChunkStarts(
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            canUseWarmupPrefix: false
        )
        let pathBStartsDecisions = try silenceAlignedChunkStarts(
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            canUseWarmupPrefix: true
        )
        let pathCStartsDecisions = regularChunkStarts(strideSamples: strideSamples)
        // Chunk count may differ between paths in pathological cases;
        // probe and commit only over the shorter prefix where both
        // disagree, and use the chosen path's full layout for the rest.
        let pathACount = pathAStartsDecisions.count
        let pathBCount = pathBStartsDecisions.count
        let pathCCount = pathCStartsDecisions.count
        // For chunk 0 both paths agree (start = 0, no warmup), so use
        // path A's count as the working chunk count for the probe.
        let chunkCount = pathACount

        var chunkOutputs: [[TokenWindow]] = []
        chunkOutputs.reserveCapacity(chunkCount)

        let worker = workers.first ?? manager

        func reportProgress(through chunkIndex: Int) async {
            if let progressHandler {
                let progress = min(1.0, Double(chunkIndex + 1) / Double(chunkCount))
                await progressHandler(progress)
            }
        }

        // ----- Chunk 0 (always path A; warmup never applies). -----
        if chunkCount == 0 {
            return await manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: [],
                processingTime: Date().timeIntervalSince(startTime)
            )
        }
        // Chunk 0 boundary is start=0 in both paths and warmup never
        // applies on chunk 0, so the decode is shared across both paths.
        let chunk0Decision = pathAStartsDecisions[0]
        let chunk0Tokens = try await decodeOneChunk(
            chunkStart: chunk0Decision.start,
            chunkIndex: 0,
            chunkSamples: chunkSamples,
            warmupSamples: 0,
            using: worker,
            decoderLayers: decoderLayers,
            maxModelSamples: maxModelSamples,
            language: language
        )
        chunkOutputs.append(chunk0Tokens)
        await reportProgress(through: 0)

        // ----- Probe phase. -----
        // Run all three paths on chunks 1..probeEnd; accumulate per-path
        // confidence stats; remember outputs so we don't re-decode the
        // winning path's probe chunks.
        let probeEnd = min(probeChunkCount, chunkCount - 1)
        var pathAProbeOutputs: [[TokenWindow]] = []
        var pathBProbeOutputs: [[TokenWindow]] = []
        var pathCProbeOutputs: [[TokenWindow]] = []
        var pathAConfSum: Float = 0
        var pathBConfSum: Float = 0
        var pathCConfSum: Float = 0
        var pathATokenCount: Int = 0
        var pathBTokenCount: Int = 0
        var pathCTokenCount: Int = 0

        if probeEnd >= 1 {
            for chunkIndex in 1...probeEnd {
                try Task.checkCancellation()

                // Path A decode under path A's own silence-aligned start.
                let pathADecision = pathAStartsDecisions[chunkIndex]
                let pathATokens = try await decodeOneChunk(
                    chunkStart: pathADecision.start,
                    chunkIndex: chunkIndex,
                    chunkSamples: chunkSamples,
                    warmupSamples: 0,
                    using: worker,
                    decoderLayers: decoderLayers,
                    maxModelSamples: maxModelSamples,
                    language: language
                )
                pathAProbeOutputs.append(pathATokens)
                for t in pathATokens { pathAConfSum += t.confidence }
                pathATokenCount += pathATokens.count

                // Path B decode under path B's own silence-aligned start
                // (and its own per-chunk warmup decision). When path B's
                // start coincides with path A's AND warmup wouldn't apply,
                // the two are structurally identical and we reuse path A.
                if chunkIndex < pathBCount {
                    let pathBDecision = pathBStartsDecisions[chunkIndex]
                    let warmupSamplesForB =
                        pathBDecision.useWarmupPrefix
                        ? min(pathBWarmupSamples, pathBDecision.start) : 0
                    if pathBDecision.start == pathADecision.start && warmupSamplesForB == 0 {
                        pathBProbeOutputs.append(pathATokens)
                        for t in pathATokens { pathBConfSum += t.confidence }
                        pathBTokenCount += pathATokens.count
                    } else {
                        let pathBTokens = try await decodeOneChunk(
                            chunkStart: pathBDecision.start,
                            chunkIndex: chunkIndex,
                            chunkSamples: chunkSamples,
                            warmupSamples: warmupSamplesForB,
                            using: worker,
                            decoderLayers: decoderLayers,
                            maxModelSamples: maxModelSamples,
                            language: language
                        )
                        pathBProbeOutputs.append(pathBTokens)
                        for t in pathBTokens { pathBConfSum += t.confidence }
                        pathBTokenCount += pathBTokens.count
                    }
                } else {
                    // Path B has fewer chunks here; reuse path A and let
                    // the probe still produce a stable signal.
                    pathBProbeOutputs.append(pathATokens)
                    for t in pathATokens { pathBConfSum += t.confidence }
                    pathBTokenCount += pathATokens.count
                }

                // Path C decode under regular fixed-stride start. When
                // path C's start coincides with path A's (silence-snap
                // landed on the regular-stride frame), reuse path A.
                if chunkIndex < pathCCount {
                    let pathCDecision = pathCStartsDecisions[chunkIndex]
                    if pathCDecision.start == pathADecision.start {
                        pathCProbeOutputs.append(pathATokens)
                        for t in pathATokens { pathCConfSum += t.confidence }
                        pathCTokenCount += pathATokens.count
                    } else {
                        let pathCTokens = try await decodeOneChunk(
                            chunkStart: pathCDecision.start,
                            chunkIndex: chunkIndex,
                            chunkSamples: chunkSamples,
                            warmupSamples: 0,
                            using: worker,
                            decoderLayers: decoderLayers,
                            maxModelSamples: maxModelSamples,
                            language: language
                        )
                        pathCProbeOutputs.append(pathCTokens)
                        for t in pathCTokens { pathCConfSum += t.confidence }
                        pathCTokenCount += pathCTokens.count
                    }
                } else {
                    pathCProbeOutputs.append(pathATokens)
                    for t in pathATokens { pathCConfSum += t.confidence }
                    pathCTokenCount += pathATokens.count
                }
            }
        }

        let pathAMean =
            pathATokenCount > 0 ? pathAConfSum / Float(pathATokenCount) : -Float.infinity
        let pathBMean =
            pathBTokenCount > 0 ? pathBConfSum / Float(pathBTokenCount) : -Float.infinity
        let pathCMean =
            pathCTokenCount > 0 ? pathCConfSum / Float(pathCTokenCount) : -Float.infinity
        let tokenRatioB: Float =
            pathATokenCount > 0
            ? Float(pathBTokenCount) / Float(pathATokenCount) : 1.0
        let tokenRatioC: Float =
            pathATokenCount > 0
            ? Float(pathCTokenCount) / Float(pathATokenCount) : 1.0

        // Decide globally. Path A is the content-safe default. The order
        // is path C first (seam recovery), then path B (cross-script
        // recovery), then default to path A.
        //
        //   Path C is selected when it emits materially more tokens than
        //   path A AND its mean per-token confidence is close to path A's
        //   (within a drift ceiling). The token-count signal isolates the
        //   case where the silence-snap dropped a seam syllable that the
        //   fixed-stride boundary preserved; the confidence-ceiling
        //   signal rejects wrong-language drifts, which present as
        //   suspiciously high-confidence emissions against an English
        //   prior on non-English audio.
        //
        //   Path B is selected by the v26 suppression-guard / margin
        //   logic: it must not be suppressing content and must beat
        //   path A's confidence by at least `pathBSwitchMargin`.
        let pathBSuppressionGuardTripped =
            pathATokenCount > 0 && tokenRatioB < pathBSuppressionRatio
        let usePathC =
            pathATokenCount > 0
            && tokenRatioC >= pathCContentRatio
            && pathCMean <= pathAMean + pathCDriftConfidenceCeiling
            && pathCMean >= pathAMean - pathCDriftConfidenceCeiling
        let usePathB =
            !usePathC
            && !pathBSuppressionGuardTripped
            && pathBMean > pathAMean + pathBSwitchMargin

        let chosenPath: String = usePathC ? "C" : (usePathB ? "B" : "A")
        logger.debug(
            "[dual-decode probe] A=(n=\(pathATokenCount), conf=\(pathAMean)) B=(n=\(pathBTokenCount), conf=\(pathBMean)) C=(n=\(pathCTokenCount), conf=\(pathCMean)) B/A=\(tokenRatioB) C/A=\(tokenRatioC) → \(chosenPath)"
        )

        // Commit probe outputs in chunk order.
        let chosenProbeOutputs: [[TokenWindow]]
        if usePathC {
            chosenProbeOutputs = pathCProbeOutputs
        } else if usePathB {
            chosenProbeOutputs = pathBProbeOutputs
        } else {
            chosenProbeOutputs = pathAProbeOutputs
        }
        chunkOutputs.append(contentsOf: chosenProbeOutputs)
        if probeEnd >= 1 {
            await reportProgress(through: probeEnd)
        }

        // ----- Post-probe phase: single-path decode for remaining chunks.
        let chosenDecisions: [ChunkStartDecision]
        if usePathC {
            chosenDecisions = pathCStartsDecisions
        } else if usePathB {
            chosenDecisions = pathBStartsDecisions
        } else {
            chosenDecisions = pathAStartsDecisions
        }
        let postProbeEnd = chosenDecisions.count
        if probeEnd + 1 < postProbeEnd {
            for chunkIndex in (probeEnd + 1)..<postProbeEnd {
                try Task.checkCancellation()

                let decision = chosenDecisions[chunkIndex]
                let warmupSamples =
                    usePathB && decision.useWarmupPrefix
                    ? min(pathBWarmupSamples, decision.start) : 0

                let tokens = try await decodeOneChunk(
                    chunkStart: decision.start,
                    chunkIndex: chunkIndex,
                    chunkSamples: chunkSamples,
                    warmupSamples: warmupSamples,
                    using: worker,
                    decoderLayers: decoderLayers,
                    maxModelSamples: maxModelSamples,
                    language: language
                )
                chunkOutputs.append(tokens)
                await reportProgress(through: chunkIndex)
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
            result.append(contentsOf: right[(lastRight + 1)...])
        }

        return result
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
