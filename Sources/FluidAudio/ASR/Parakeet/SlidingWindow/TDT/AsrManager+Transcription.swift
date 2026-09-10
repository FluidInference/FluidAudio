import Foundation

extension AsrManager {

    internal func transcribeWithState(
        _ audioSamples: [Float], decoderState: inout TdtDecoderState, language: Language? = nil
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        let minimumRequiredSamples = ASRConstants.minimumRequiredSamples(forSampleRate: config.sampleRate)
        guard audioSamples.count >= minimumRequiredSamples else { throw ASRError.invalidAudioData }

        let startTime = Date()

        // Route to appropriate processing method based on audio length
        if audioSamples.count <= ASRConstants.maxModelSamples {
            let (alignedSamples, frameAlignedLength) = frameAlignedAudio(audioSamples)
            let paddedAudio: [Float] = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
            let (hypothesis, encoderSequenceLength) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: frameAlignedLength,
                actualAudioFrames: nil,  // Will be calculated from originalLength
                decoderState: &decoderState,
                isLastChunk: true,  // Single-chunk: always first and last
                language: language
            )

            let result = processTranscriptionResult(
                tokenIds: hypothesis.ySequence,
                timestamps: hypothesis.timestamps,
                confidences: hypothesis.tokenConfidences,
                tokenDurations: hypothesis.tokenDurations,
                encoderSequenceLength: encoderSequenceLength,
                audioSampleCount: audioSamples.count,
                processingTime: Date().timeIntervalSince(startTime)
            )

            return result
        }

        // ChunkProcessor handles stateless chunked transcription for long audio
        let processor = ChunkProcessor(audioSamples: audioSamples)
        let result = try await processor.process(
            using: self,
            startTime: startTime,
            progressHandler: { [weak self] progress in
                guard let self else { return }
                await self.progressEmitter.report(progress: progress)
            },
            language: language
        )

        return result
    }

    /// Cross-window emission jitter allowance for the final-window re-decode: a
    /// re-decoded token can land a few frames from its original emission, so the
    /// suppression cutoff backs off this much and dedup strips what remains.
    internal static let redecodeEmissionJitterFrames = 5

    /// Decoder-entry plan for the final streaming window (issue #855).
    ///
    /// Returns `initialTimeIndexOverride: 0` so the decoder re-decodes the window
    /// from frame 0 (a mid-window entry into a short flush window can blank out
    /// the trailing speech), plus an emission cutoff in window-local frames:
    /// tokens for audio the previous windows already emitted are suppressed at
    /// the source, leaving dedup only the jitter margin. `transcribeChunk` pairs
    /// the frame-0 entry with a *fresh* decoder state — see the note there.
    /// Non-final windows and callers without accumulated timestamps get `(nil, nil)`
    /// — the legacy navigation.
    nonisolated internal static func lastChunkRedecodePlan(
        isLastChunk: Bool,
        previousTokens: [Int],
        previousTokenTimestamps: [Int]?,
        globalFrameOffset: Int,
        lastWordStartFrame: Int? = nil
    ) -> (initialTimeIndexOverride: Int?, emitTokensAfterFrame: Int?) {
        guard isLastChunk, let previousTimestamps = previousTokenTimestamps, !previousTokens.isEmpty else {
            return (nil, nil)
        }
        // Anchor the cutoff at the previous window's last *word*, not its last
        // token: that word may have been cut by the window edge (#897), and the
        // re-decode must be free to re-emit it in full.
        let anchorGlobalFrame = lastWordStartFrame ?? (previousTimestamps.max() ?? 0)
        let cutoff = max(0, anchorGlobalFrame - globalFrameOffset - redecodeEmissionJitterFrames)
        return (0, cutoff)
    }

    /// Index of the first token of the previous window's last word — a piece
    /// carrying the SentencePiece word boundary, or the leading space the loaded
    /// vocabulary normalizes it to — or nil when the sequence has no word start
    /// after its first token (dropping index 0 would discard the whole window).
    /// Pure, for testability.
    nonisolated internal static func trailingWordStartIndex(pieces: [String]) -> Int? {
        guard
            let idx = pieces.lastIndex(where: {
                $0.hasPrefix(ASRConstants.sentencePieceWordBoundary) || $0.hasPrefix(" ")
            }), idx > 0
        else { return nil }
        return idx
    }

    /// A vocabulary piece that is nothing but punctuation once the word
    /// boundary (marker or normalized leading space) is stripped.
    nonisolated internal static func isPunctuationPiece(_ piece: String) -> Bool {
        var core = piece
        if core.hasPrefix(ASRConstants.sentencePieceWordBoundary) {
            core.removeFirst(ASRConstants.sentencePieceWordBoundary.count)
        }
        core = core.trimmingCharacters(in: .whitespaces)
        guard !core.isEmpty else { return false }
        return core.unicodeScalars.allSatisfy { CharacterSet.punctuationCharacters.contains($0) }
    }

    /// Letters/digits of a word's pieces, lower-cased: boundary markers, spaces
    /// and punctuation stripped. Empty when the pieces carry no word.
    nonisolated internal static func wordCore<S: Sequence>(_ pieces: S) -> String where S.Element == String {
        pieces.joined().lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }
            .map { String($0) }.joined()
    }

    /// The pieces of the first word in a token sequence: from the first
    /// word-start piece through the piece before the next word start.
    nonisolated internal static func firstWordPieces(_ pieces: [String]) -> [String] {
        func startsWord(_ p: String) -> Bool {
            p.hasPrefix(ASRConstants.sentencePieceWordBoundary) || p.hasPrefix(" ")
        }
        guard let start = pieces.firstIndex(where: { startsWord($0) && !isPunctuationPiece($0) }) else {
            return []
        }
        var end = start + 1
        while end < pieces.count, !startsWord(pieces[end]), !isPunctuationPiece(pieces[end]) {
            end += 1
        }
        return Array(pieces[start..<end])
    }

    /// Seam reconciliation for the final streaming window (#897).
    ///
    /// The previous window's trailing word may be a fragment cut by the window
    /// edge (`and an` for `and analyzing`); the re-decoded final window emits
    /// that word in full from `trailingWordStart`'s frame minus the jitter
    /// margin. Dedup cannot fix this — the fragment never equals the full word,
    /// a re-emitted single token is below the substring matcher's minimum run,
    /// and a boundary punctuation the decoder attaches at its emission start
    /// blocks the suffix–prefix match.
    ///
    /// Decision, from the token shapes in the #897 corpus run:
    /// 1. Leading *continuation* pieces of the re-decode (no word boundary) are
    ///    the tail of the previous last word whose start fell under the cutoff
    ///    (`box` + `x, but`). They are dropped and never justify retiring.
    /// 2. The first real word of the re-decode then decides the previous word's
    ///    fate. Same word: keep the previous copy (it carries the sentence-final
    ///    punctuation a re-decode at the audio end omits), drop the re-emission.
    ///    Previous text a strict prefix of it (`an`→`analyzing`,
    ///    `every`→`everything`): a fragment, retire. A different word that
    ///    overlaps the previous word's span and was started by the re-decode
    ///    itself (no continuation head): it disagrees with more context, retire.
    ///    Behind a continuation head the first real word is the next word by
    ///    construction, so only the prefix rule can retire. A different word
    ///    starting later: the re-decode skipped the previous word, keep it.
    /// 3. Retiring additionally requires the re-decode to reach past the
    ///    previous window's last frame; an empty or early-ending final window
    ///    keeps the previous word.
    /// 4. The re-decode's head is then stripped of seam artifacts: punctuation
    ///    at or before the previous last word's frame, and tokens — punctuation
    ///    included — that duplicate a kept previous token within
    ///    `frameTolerance` inside the jitter region.
    ///
    /// Returns how many trailing previous tokens to drop (the whole last word
    /// or none) and how many leading current tokens to drop. Pure, for
    /// testability; timestamps are global frames.
    nonisolated internal static func reconcileFinalWindowSeam(
        previousTokens: [Int],
        previousTimestamps: [Int],
        trailingWordStart: Int,
        currentTokens: [Int],
        currentTimestamps: [Int],
        currentPieces: [String] = [],
        previousPieces: [String] = [],
        punctuationTokens: [Int] = ASRConstants.punctuationTokens,
        jitterFrames: Int = redecodeEmissionJitterFrames,
        frameTolerance: Int = ASRConstants.duplicateFrameTolerance
    ) -> (droppedPrevious: Int, droppedCurrent: Int) {
        guard trailingWordStart > 0, trailingWordStart < previousTokens.count,
            previousTimestamps.count == previousTokens.count,
            currentTimestamps.count == currentTokens.count,
            !currentTokens.isEmpty
        else { return (0, 0) }

        func piece(_ index: Int) -> String { index < currentPieces.count ? currentPieces[index] : "" }
        // `ASRConstants.punctuationTokens` lists only sentence-final marks; the
        // seam artifact is usually a comma, so classify by the piece text too.
        func isPunctuation(_ index: Int) -> Bool {
            punctuationTokens.contains(currentTokens[index]) || isPunctuationPiece(piece(index))
        }
        func startsWord(_ index: Int) -> Bool {
            let p = piece(index)
            return p.hasPrefix(ASRConstants.sentencePieceWordBoundary) || p.hasPrefix(" ")
        }

        let lastWordStartFrame = previousTimestamps[trailingWordStart]
        let previousLastFrame = previousTimestamps[previousTokens.count - 1]
        let extendsBeyondPrevious =
            (currentTimestamps.max() ?? Int.min) > previousLastFrame + jitterFrames

        // 1. Leading continuation pieces: the tail of the previous last word.
        var head = 0
        while head < currentTokens.count, !startsWord(head), !isPunctuation(head) {
            head += 1
        }

        // 2. The first real word of the re-decode.
        let previousWord = wordCore(previousPieces.dropFirst(trailingWordStart))
        let firstWord = firstWordPieces(Array(currentPieces.dropFirst(head)))
        let currentWord = wordCore(firstWord)
        let firstWordIndex = (head..<currentTokens.count).first { startsWord($0) && !isPunctuation($0) }
        let firstWordFrame = firstWordIndex.map { currentTimestamps[$0] }

        let retire: Bool
        if !extendsBeyondPrevious || currentWord.isEmpty || previousWord.isEmpty {
            retire = false
        } else if currentWord == previousWord {
            retire = false
        } else if currentWord.hasPrefix(previousWord) {
            retire = true
        } else if head == 0, let frame = firstWordFrame, frame <= previousLastFrame + jitterFrames {
            // Overlapping different word, and the re-decode started it itself
            // (no continuation head): it disagrees with more context. Behind a
            // continuation head the first real word is the *next* word by
            // construction (`box` + `x, but`), so only the prefix rule applies.
            retire = true
        } else {
            retire = false
        }

        let droppedPrevious = retire ? previousTokens.count - trailingWordStart : 0
        let keptPrevious = Array(
            zip(previousTokens, previousTimestamps).prefix(retire ? trailingWordStart : previousTokens.count))
        let keptLastFrame = keptPrevious.last?.1 ?? -1

        // 4. Strip the seam artifacts from the re-decode's head.
        var droppedCurrent = head
        for index in head..<currentTokens.count {
            let id = currentTokens[index]
            let frame = currentTimestamps[index]
            if isPunctuation(index), frame <= lastWordStartFrame {
                droppedCurrent += 1
                continue
            }
            if frame <= keptLastFrame + jitterFrames,
                keptPrevious.contains(where: { $0.0 == id && abs($0.1 - frame) <= frameTolerance })
            {
                droppedCurrent += 1
                continue
            }
            break
        }
        return (droppedPrevious, droppedCurrent)
    }

    /// Chunk transcription that preserves decoder state between calls.
    /// Used by SlidingWindowAsrManager for overlapping-window processing with token deduplication.
    func transcribeChunk(
        _ chunkSamples: [Float],
        decoderState: inout TdtDecoderState,
        previousTokens: [Int] = [],
        previousTokenTimestamps: [Int]? = nil,
        globalFrameOffset: Int = 0,
        isLastChunk: Bool = false,
        language: Language? = nil
    ) async throws -> (
        tokens: [Int], timestamps: [Int], confidences: [Float], encoderSequenceLength: Int,
        droppedPreviousTokens: Int
    ) {
        let (alignedSamples, frameAlignedLength) = frameAlignedAudio(
            chunkSamples, allowAlignment: previousTokens.isEmpty)
        let padded = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
        // Last streaming window: decode from frame 0 instead of skipping the overlap.
        // Jumping mid-window into a short flush window can blank out the trailing
        // speech entirely (issue #855: the joint emits a boundary punctuation, then
        // blanks to the end, dropping the final words). Emissions for audio the
        // previous windows already covered are suppressed at the source, so dedup
        // only sees the few-frame jitter margin — a token-dense overlap cannot
        // outgrow dedup's bounded search.
        //
        // The re-decode runs on a FRESH decoder state, as the batch chunker does
        // for every chunk. Re-walking the overlap with the carried state — state
        // that already consumed that audio — leaves the decoder emitting blanks
        // for the rest of the window: on a 12 s final window it produced zero
        // tokens for 9 s of never-seen speech (#855 follow-up, three real
        // recordings). The 2 s left context is enough for a fresh state to
        // re-establish itself before the cutoff.
        // The previous window's last word is re-decoded in full by the final
        // window and dropped from the accumulated output afterwards (#897).
        let trailingWordStart: Int? =
            isLastChunk && previousTokenTimestamps?.count == previousTokens.count
            ? Self.trailingWordStartIndex(pieces: previousTokens.map { vocabulary[$0] ?? "" })
            : nil
        let redecodePlan = Self.lastChunkRedecodePlan(
            isLastChunk: isLastChunk,
            previousTokens: previousTokens,
            previousTokenTimestamps: previousTokenTimestamps,
            globalFrameOffset: globalFrameOffset,
            lastWordStartFrame: trailingWordStart.flatMap { previousTokenTimestamps?[$0] }
        )
        if redecodePlan.initialTimeIndexOverride == 0 {
            decoderState = TdtDecoderState.make(decoderLayers: decoderLayerCount)
        }
        let (hypothesis, encLen) = try await executeMLInferenceWithTimings(
            padded,
            originalLength: frameAlignedLength,
            actualAudioFrames: nil,  // Will be calculated from originalLength
            decoderState: &decoderState,
            contextFrameAdjustment: 0,  // Non-streaming chunks don't use adaptive context
            isLastChunk: isLastChunk,
            language: language,
            emitTokensAfterGlobalFrame: redecodePlan.emitTokensAfterFrame,
            initialTimeIndexOverride: redecodePlan.initialTimeIndexOverride
        )

        var currentTokens = hypothesis.ySequence
        var currentTimestamps = hypothesis.timestamps
        var currentConfidences = hypothesis.tokenConfidences
        var effectivePrevious = previousTokens
        var effectivePreviousTimestamps = previousTokenTimestamps
        var droppedPrevious = 0

        // Final window: replace the previous window's (possibly edge-cut) last
        // word with the re-decoded one, and strip the seam artifacts the
        // re-decode emits ahead of it (#897).
        if redecodePlan.initialTimeIndexOverride == 0, let trailingWordStart,
            let previousTimestamps = previousTokenTimestamps
        {
            let seam = Self.reconcileFinalWindowSeam(
                previousTokens: previousTokens,
                previousTimestamps: previousTimestamps,
                trailingWordStart: trailingWordStart,
                currentTokens: currentTokens,
                currentTimestamps: currentTimestamps.map { $0 + globalFrameOffset },
                currentPieces: currentTokens.map { vocabulary[$0] ?? "" },
                previousPieces: previousTokens.map { vocabulary[$0] ?? "" }
            )
            droppedPrevious = seam.droppedPrevious
            if seam.droppedCurrent > 0 {
                currentTokens.removeFirst(seam.droppedCurrent)
                currentTimestamps.removeFirst(seam.droppedCurrent)
                currentConfidences.removeFirst(min(seam.droppedCurrent, currentConfidences.count))
            }
            effectivePrevious = Array(previousTokens.prefix(trailingWordStart))
            effectivePreviousTimestamps = Array(previousTimestamps.prefix(trailingWordStart))
            if droppedPrevious > 0 || seam.droppedCurrent > 0 {
                logger.debug(
                    "Final-window seam: dropped \(droppedPrevious) trailing previous token(s), \(seam.droppedCurrent) leading current token(s)"
                )
            }
        }

        // Apply token deduplication if previous tokens are provided
        if !effectivePrevious.isEmpty && !currentTokens.isEmpty {
            // Convert this chunk's local frame timestamps into the same global frame
            // space as `previousTokenTimestamps` so dedup can require temporal adjacency.
            let currentGlobalTimestamps: [Int]? =
                effectivePreviousTimestamps != nil ? currentTimestamps.map { $0 + globalFrameOffset } : nil
            let (deduped, removedCount) = removeDuplicateTokenSequence(
                previous: effectivePrevious, current: currentTokens,
                previousTimestamps: effectivePreviousTimestamps,
                currentTimestamps: currentGlobalTimestamps)
            let adjustedTimestamps =
                removedCount > 0 ? Array(currentTimestamps.dropFirst(removedCount)) : currentTimestamps
            let adjustedConfidences =
                removedCount > 0
                ? Array(currentConfidences.dropFirst(removedCount)) : currentConfidences

            return (deduped, adjustedTimestamps, adjustedConfidences, encLen, droppedPrevious)
        }

        return (currentTokens, currentTimestamps, currentConfidences, encLen, droppedPrevious)
    }

    internal func processTranscriptionResult(
        tokenIds: [Int],
        timestamps: [Int] = [],
        confidences: [Float] = [],
        tokenDurations: [Int] = [],
        encoderSequenceLength: Int,
        audioSampleCount: Int,
        processingTime: TimeInterval
    ) -> ASRResult {

        let text = convertTokensToText(tokenIds)
        let duration = TimeInterval(audioSampleCount) / TimeInterval(config.sampleRate)

        let resultTimings = createTokenTimings(
            from: tokenIds, timestamps: timestamps, confidences: confidences, tokenDurations: tokenDurations)

        let confidence = calculateConfidence(
            tokenCount: tokenIds.count,
            isEmpty: text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            tokenConfidences: confidences
        )

        return ASRResult(
            text: text,
            confidence: confidence,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: resultTimings
        )
    }

}
