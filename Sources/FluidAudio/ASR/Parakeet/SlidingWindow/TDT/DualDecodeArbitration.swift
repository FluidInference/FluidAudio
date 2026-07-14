import Foundation

private struct DualDecodeArbitrationConfig {
    /// Number of non-first chunks to probe before choosing one path for the file.
    let probeChunkCount: Int = 3

    /// Path B's hidden real-audio warmup prefix.
    let pathBWarmupFrames: Int = 7

    /// Minimum mean confidence edge path B needs over path A.
    let pathBSwitchMargin: Float = 0.001

    /// Maximum B/A token ratio for path B to be considered drift suppression.
    let pathBMaxContentRatio: Float = 0.9

    /// B/A token ratio below which path B is treated as content suppression.
    let pathBSuppressionRatio: Float = 0.6

    /// Minimum C/A token ratio for fixed-stride path C to be content recovery.
    let pathCContentRatio: Float = 1.15

    /// Maximum allowed confidence distance between path C and path A.
    let pathCDriftConfidenceCeiling: Float = 0.03

    /// Minimum exact-token agreement required between path C and path A.
    let pathCAgreementRatio: Float = 0.75

    var pathBWarmupSamples: Int {
        pathBWarmupFrames * ASRConstants.samplesPerEncoderFrame
    }
}

extension ChunkProcessor {
    /// Per-file probe arbitration over silence-aligned and fixed-stride chunk
    /// layouts. The file commits to one path globally, so the chunk merger never
    /// has to stitch together mixed-path BPE output inside one transcript.
    ///
    /// Path A: silence-aligned starts, no warmup prefix.
    /// Path B: silence-aligned starts, with hidden 7-frame real-audio warmup.
    /// Path C: regular fixed-stride starts, no warmup prefix.
    ///
    /// Mechanism is language-agnostic: it uses acoustic chunk geometry and
    /// model confidences only. It does not inspect text, vocab, script, or
    /// language identity.
    func processWithDualDecodeArbitration(
        using manager: AsrManager,
        workers: [AsrManager],
        decoderLayers: Int,
        maxModelSamples: Int,
        modelVersion: AsrModelVersion?,
        startTime: Date,
        progressHandler: ((Double) async -> Void)?,
        language: Language?
    ) async throws -> ASRResult {
        let config = DualDecodeArbitrationConfig()
        let logger = AppLogger(category: "ChunkProcessor")

        // Layout is computed in the no-mel shape because both decode paths
        // run with `melChunkContext == false` semantics; the only difference
        // between path A and path B for a given chunk is the warmup prefix.
        let layout = chunkLayout(melChunkContext: false, modelVersion: modelVersion)
        let chunkSamples = layout.chunkSamples
        let strideSamples = layout.strideSamples

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

        let pathBCount = pathBStartsDecisions.count
        let pathCCount = pathCStartsDecisions.count
        let chunkCount = pathAStartsDecisions.count

        var chunkOutputs: [[TokenWindow]] = []
        chunkOutputs.reserveCapacity(chunkCount)

        let worker = workers.first ?? manager

        func reportProgress(through chunkIndex: Int) async {
            if let progressHandler {
                let progress = min(1.0, Double(chunkIndex + 1) / Double(chunkCount))
                await progressHandler(progress)
            }
        }

        if chunkCount == 0 {
            return await manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSampleCount: totalSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        // Chunk 0 boundary is start=0 in all paths and warmup never applies.
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

        let probeEnd = min(config.probeChunkCount, chunkCount - 1)
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
                for token in pathATokens { pathAConfSum += token.confidence }
                pathATokenCount += pathATokens.count

                if chunkIndex < pathBCount {
                    let pathBDecision = pathBStartsDecisions[chunkIndex]
                    let warmupSamplesForB =
                        pathBDecision.useWarmupPrefix
                        ? min(config.pathBWarmupSamples, pathBDecision.start) : 0
                    if pathBDecision.start == pathADecision.start && warmupSamplesForB == 0 {
                        pathBProbeOutputs.append(pathATokens)
                        for token in pathATokens { pathBConfSum += token.confidence }
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
                        for token in pathBTokens { pathBConfSum += token.confidence }
                        pathBTokenCount += pathBTokens.count
                    }
                } else {
                    pathBProbeOutputs.append(pathATokens)
                    for token in pathATokens { pathBConfSum += token.confidence }
                    pathBTokenCount += pathATokens.count
                }

                if chunkIndex < pathCCount {
                    let pathCDecision = pathCStartsDecisions[chunkIndex]
                    if pathCDecision.start == pathADecision.start {
                        pathCProbeOutputs.append(pathATokens)
                        for token in pathATokens { pathCConfSum += token.confidence }
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
                        for token in pathCTokens { pathCConfSum += token.confidence }
                        pathCTokenCount += pathCTokens.count
                    }
                } else {
                    pathCProbeOutputs.append(pathATokens)
                    for token in pathATokens { pathCConfSum += token.confidence }
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
            pathATokenCount > 0 ? Float(pathBTokenCount) / Float(pathATokenCount) : 1.0
        let tokenRatioC: Float =
            pathATokenCount > 0 ? Float(pathCTokenCount) / Float(pathATokenCount) : 1.0

        let agreementToleranceFrames = Int(overlapSeconds / ASRConstants.secondsPerEncoderFrame) / 2
        var pathACMatchedTokens = 0
        for chunkIndex in 0..<pathAProbeOutputs.count {
            let a = pathAProbeOutputs[chunkIndex]
            let c = chunkIndex < pathCProbeOutputs.count ? pathCProbeOutputs[chunkIndex] : []
            for aTok in a {
                for cTok in c {
                    if aTok.token == cTok.token
                        && abs(aTok.timestamp - cTok.timestamp) <= agreementToleranceFrames
                    {
                        pathACMatchedTokens += 1
                        break
                    }
                }
            }
        }
        let pathCAgreement: Float =
            pathATokenCount > 0 ? Float(pathACMatchedTokens) / Float(pathATokenCount) : 1.0

        let pathBSuppressionGuardTripped =
            pathATokenCount > 0 && tokenRatioB < config.pathBSuppressionRatio
        let usePathC =
            pathATokenCount > 0
            && tokenRatioC >= config.pathCContentRatio
            && pathCAgreement >= config.pathCAgreementRatio
            && pathCMean <= pathAMean + config.pathCDriftConfidenceCeiling
            && pathCMean >= pathAMean - config.pathCDriftConfidenceCeiling
        let usePathB =
            !usePathC
            && !pathBSuppressionGuardTripped
            && tokenRatioB <= config.pathBMaxContentRatio
            && pathBMean > pathAMean + config.pathBSwitchMargin

        let chosenPath: String = usePathC ? "C" : (usePathB ? "B" : "A")
        logger.debug(
            "[dual-decode probe] A=(n=\(pathATokenCount), conf=\(pathAMean)) B=(n=\(pathBTokenCount), conf=\(pathBMean)) C=(n=\(pathCTokenCount), conf=\(pathCMean)) B/A=\(tokenRatioB) C/A=\(tokenRatioC) C-agree=\(pathCAgreement) -> \(chosenPath)"
        )

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
                    ? min(config.pathBWarmupSamples, decision.start) : 0

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
                audioSampleCount: totalSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if chunkOutputs.count > 1 {
            let vocabulary = await manager.vocabulary
            let spliceSafeTokenIds = Self.spliceSafeTokenIds(vocabulary: vocabulary)
            let caseVariantIds = Self.caseVariantCanonicalIds(vocabulary: vocabulary)
            for chunk in chunkOutputs.dropFirst() {
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
            audioSampleCount: totalSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    /// Decode a single chunk under the given start + warmup parameters.
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
}
