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

    func mergeChunks(
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
