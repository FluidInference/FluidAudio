import Accelerate
@preconcurrency import CoreML
import Foundation

/// Blank-span rescue (issue #838).
///
/// The cache-aware encoder + RNN-T decoder carry state across chunks. After
/// certain preceding audio, the greedy decode can collapse to blank for an
/// entire short, pause-delimited word — silently dropping it. The same span
/// decodes correctly from fresh state, so the rescue re-decodes exactly the
/// spans that produced nothing:
///
///   1. Track speech spans with a per-encoder-frame (80 ms) RMS gate.
///   2. When a span closes (240 ms of silence) and no token timing falls
///      inside it, re-decode the buffered span audio on fresh encoder caches
///      and decoder state, appending any recovered tokens to the transcript.
///   3. Restore the live stream's state afterwards — the main decode path is
///      byte-identical whenever spans emit normally.
///
/// A span that was never speech (noise, clicks) re-decodes to nothing and the
/// rescue is a no-op, so a false trigger costs only the extra decode.
extension StreamingNemotronMultilingualAsrManager {

    /// RMS below this is silence for span tracking. Independent of the
    /// opt-in VAD-skip threshold; 0 disables the rescue entirely.
    nonisolated internal static let rescueRmsThreshold: Float = {
        if let raw = ProcessInfo.processInfo.environment["FLUIDAUDIO_RESCUE_RMS_THRESHOLD"],
            let value = Float(raw)
        {
            return value
        }
        return 0.0025
    }()

    nonisolated internal static let blankRescueEnabled: Bool = {
        if let raw = ProcessInfo.processInfo.environment["FLUIDAUDIO_DISABLE_BLANK_RESCUE"] {
            let lowered = raw.lowercased()
            return lowered == "0" || lowered == "false" || lowered == "no"
        }
        return true
    }()

    /// Span-tracking geometry, in 80 ms RMS windows (1 encoder frame each).
    private static let closeSilentWindows = 2  // span closes after 160 ms of silence
    private static let preRollWindows = 3  // 240 ms of audio kept before speech onset
    private static let minSpeechWindows = 2  // spans shorter than 160 ms are not rescued
    private static let maxSpanSamples = 15 * 16000  // give up past 15 s (span emitted nothing that long)

    /// `processChunk` plus blank-span bookkeeping. All streaming call sites
    /// route through this; the rescue itself calls `processChunk` directly.
    internal func processChunkTracked(_ samples: [Float], nextChunkSamples: [Float]? = nil) async throws {
        guard Self.blankRescueEnabled, Self.rescueRmsThreshold > 0, !inBlankRescue else {
            try await processChunk(samples, nextChunkSamples: nextChunkSamples)
            return
        }
        let chunkStartFrame = rescueFrameCursor
        try await processChunk(samples, nextChunkSamples: nextChunkSamples)
        rescueFrameCursor += samples.count / ASRConstants.samplesPerEncoderFrame
        try await updateRescueSpans(chunk: samples, chunkStartFrame: chunkStartFrame)
    }

    /// Close a span left open by end-of-stream. Called from `finish()` after
    /// the trailing chunk is processed and before the transcript is decoded.
    internal func finalizeRescueSpanIfNeeded() async throws {
        guard Self.blankRescueEnabled, Self.rescueRmsThreshold > 0, rescueSpanOpen else { return }
        try await closeSpanAndMaybeRescue()
    }

    /// Walk the chunk in 80 ms windows, advancing the span state machine.
    private func updateRescueSpans(chunk: [Float], chunkStartFrame: Int) async throws {
        let windowSamples = ASRConstants.samplesPerEncoderFrame
        let windowCount = chunk.count / windowSamples
        for index in 0..<windowCount {
            let start = index * windowSamples
            var rms: Float = 0
            chunk.withUnsafeBufferPointer { buf in
                vDSP_rmsqv(buf.baseAddress! + start, 1, &rms, vDSP_Length(windowSamples))
            }
            let silent = rms < Self.rescueRmsThreshold
            let frame = chunkStartFrame + index

            if rescueSpanOpen {
                if silent {
                    rescueSilentWindowRun += 1
                } else {
                    rescueSilentWindowRun = 0
                    rescueSpanLastSpeechFrame = frame
                    rescueSpanSpeechWindows += 1
                }
            } else if !silent {
                rescueSpanOpen = true
                rescueSpanStartFrame = frame
                rescueSpanLastSpeechFrame = frame
                rescueSpanSpeechWindows = 1
                rescueSilentWindowRun = 0
                rescueSpanAudio = rescuePreRollTail
                rescueSpanPreRollFrames = rescuePreRollTail.count / windowSamples
                rescueSpanOverflowed = false
            }

            if rescueSpanOpen, !rescueSpanOverflowed {
                rescueSpanAudio.append(contentsOf: chunk[start..<(start + windowSamples)])
                if rescueSpanAudio.count > Self.maxSpanSamples {
                    rescueSpanOverflowed = true
                    rescueSpanAudio = []
                }
            } else if !rescueSpanOpen {
                rescuePreRollTail.append(contentsOf: chunk[start..<(start + windowSamples)])
                let maxPreRoll = Self.preRollWindows * windowSamples
                if rescuePreRollTail.count > maxPreRoll {
                    rescuePreRollTail.removeFirst(rescuePreRollTail.count - maxPreRoll)
                }
            }

            if rescueSpanOpen, rescueSilentWindowRun >= Self.closeSilentWindows {
                try await closeSpanAndMaybeRescue()
            }
        }
    }

    /// Evaluate the just-closed span: if no token timing falls inside it,
    /// re-decode the buffered audio from fresh state.
    private func closeSpanAndMaybeRescue() async throws {
        let spanAudio = rescueSpanAudio
        let startFrame = rescueSpanStartFrame
        let lastSpeechFrame = rescueSpanLastSpeechFrame
        let speechWindows = rescueSpanSpeechWindows
        let preRollFrames = rescueSpanPreRollFrames
        let overflowed = rescueSpanOverflowed
        rescueSpanOpen = false
        rescueSpanAudio = []
        rescueSpanSpeechWindows = 0
        rescueSilentWindowRun = 0
        rescueSpanOverflowed = false
        // The silence that closed the span is this span's pre-roll history.
        rescuePreRollTail = []

        guard !overflowed, speechWindows >= Self.minSpeechWindows else { return }

        // One frame of onset slack (the RMS gate can trail the acoustics);
        // five frames (400 ms) of closing slack because RNN-T emissions lag
        // the audio — a span's token often lands a few frames into the
        // trailing silence, and counting it prevents a duplicate emission
        // from the rescue.
        let openSec =
            (Double(startFrame) - 1) * ASRConstants.secondsPerEncoderFrame
        let closeSec =
            (Double(lastSpeechFrame) + 5) * ASRConstants.secondsPerEncoderFrame
        // Only lexical tokens count as span output: with long pauses the
        // decode often spends the span's frames emitting the previous
        // sentence's terminal punctuation ("afternoon." + dropped word) —
        // punctuation alone must not mask a swallowed word.
        let spanEmitted = accumulatedTokenTimings.contains {
            $0.startTime >= openSec && $0.startTime <= closeSec
                && Self.containsLexicalContent($0.token)
        }
        guard !spanEmitted else { return }

        recordDetectedBlankSpan()
        try await rescueDecode(
            span: spanAudio,
            spanStartFrame: startFrame - preRollFrames,
            spanEndFrame: lastSpeechFrame + 1)
    }

    /// A speech span decoded to all-blank live; a rescue is being attempted.
    internal func recordDetectedBlankSpan() {
        detectedBlankSpanCount += 1
    }

    /// A rescue committed lexical content to the transcript.
    internal func recordSuccessfulRescue() {
        blankRescueCount += 1
    }

    /// Re-decode `span` on fresh encoder caches and decoder state. The trial
    /// decode is staged: its tokens are captured and removed, partial
    /// callbacks are suppressed, and the result is committed only when it
    /// contains lexical content — inserted at the span's timestamp position
    /// so words already decoded from later audio in the same chunk keep
    /// their order. Live stream state is restored afterwards.
    private func rescueDecode(span: [Float], spanStartFrame: Int, spanEndFrame: Int) async throws {
        guard encoder != nil, config.chunkSamples > 0, let tokenizer = tokenizer else { return }

        // Save the live stream's carried state. Loopback outputs are never
        // output-backed (see makePredictionOptions), so these references stay
        // valid across the rescue's predictions.
        let savedCacheChannel = cacheChannel
        let savedCacheTime = cacheTime
        let savedCacheLen = cacheLen
        let savedEncoderState = encoderState
        let savedMelCache = melCache
        let savedHState = hState
        let savedCState = cState
        let savedLastToken = lastToken
        let savedPrefetchedMel = prefetchedMel
        let savedPrefetchedEncoded = prefetchedEncoded
        let savedPrefetchedEncoderProj = prefetchedEncoderProj
        let savedPrefetchedCacheChannel = prefetchedCacheChannel
        let savedPrefetchedCacheTime = prefetchedCacheTime
        let savedPrefetchedCacheLen = prefetchedCacheLen
        let savedAbsoluteFrameBase = absoluteFrameBase
        let savedChunkCount = chunkCount
        let savedProcessedChunks = processedChunks
        let savedVadRun = vadConsecutiveLowChunks
        // Suppress partial callbacks for the duration of the trial decode —
        // they would surface unvalidated, out-of-order text. A single
        // callback fires after a successful commit instead.
        let savedPartialCallback = partialCallback
        partialCallback = nil

        inBlankRescue = true
        defer {
            cacheChannel = savedCacheChannel
            cacheTime = savedCacheTime
            cacheLen = savedCacheLen
            encoderState = savedEncoderState
            melCache = savedMelCache
            hState = savedHState
            cState = savedCState
            lastToken = savedLastToken
            prefetchedMel = savedPrefetchedMel
            prefetchedEncoded = savedPrefetchedEncoded
            prefetchedEncoderProj = savedPrefetchedEncoderProj
            prefetchedCacheChannel = savedPrefetchedCacheChannel
            prefetchedCacheTime = savedPrefetchedCacheTime
            prefetchedCacheLen = savedPrefetchedCacheLen
            absoluteFrameBase = savedAbsoluteFrameBase
            chunkCount = savedChunkCount
            processedChunks = savedProcessedChunks
            vadConsecutiveLowChunks = savedVadRun
            partialCallback = savedPartialCallback
            inBlankRescue = false
        }

        try resetStatesForRescue()
        // Sessions with a forced language re-seed the lang-tag prefix so the
        // rescue decodes in the same language as the live stream.
        try await applyForcedPrefixIfNeeded()
        // Rescue tokens keep their true position on the stream timeline.
        absoluteFrameBase = max(0, spanStartFrame)

        let chunkSamples = config.chunkSamples
        var audio = span
        let remainder = audio.count % chunkSamples
        if remainder != 0 {
            audio.append(contentsOf: repeatElement(0, count: chunkSamples - remainder))
        }
        // One trailing silent chunk so the final speech frames decode with
        // their full attention lookahead (mirrors finish()'s zero-pad).
        audio.append(contentsOf: repeatElement(0, count: chunkSamples))

        let tokensBefore = accumulatedTokenIds.count
        let timingsBefore = accumulatedTokenTimings.count
        var offset = 0
        while offset < audio.count {
            try await processChunk(Array(audio[offset..<(offset + chunkSamples)]))
            offset += chunkSamples
        }

        // Stage: pull the trial decode's output back out of the accumulators.
        // Lang-tag re-emissions carry no user content (and have no timing
        // entry) — drop them so staged ids and timings stay 1:1.
        let stagedIds = Array(accumulatedTokenIds[tokensBefore...])
            .filter { !config.langTagTokenIds.contains($0) }
        // Clamp staged timings to the span's real extent: the rescue decode's
        // own emission latency (plus the zero-pad flush) stamps trailing
        // tokens past the span end, and a timing that bleeds into the NEXT
        // span's attribution window would wrongly mask that span's drop.
        let spanEndSec = Double(spanEndFrame) * ASRConstants.secondsPerEncoderFrame
        let stagedTimings = Array(accumulatedTokenTimings[timingsBefore...]).map { t in
            t.startTime <= spanEndSec
                ? t
                : TokenTiming(
                    token: t.token, tokenId: t.tokenId, startTime: spanEndSec,
                    endTime: spanEndSec + ASRConstants.secondsPerEncoderFrame,
                    confidence: t.confidence)
        }
        accumulatedTokenIds.removeSubrange(tokensBefore...)
        accumulatedTokenTimings.removeSubrange(timingsBefore...)

        // Commit only lexical recoveries: punctuation or nothing means the
        // span is genuinely blank (noise, or unrecoverable) — leave the
        // transcript untouched.
        let hasLexical = stagedIds.contains {
            Self.containsLexicalContent(tokenizer.rawToken(for: $0) ?? "")
        }
        guard hasLexical else {
            logger.info(
                "Blank-span rescue found no lexical content in a "
                    + String(format: "%.2f", Double(span.count) / 16000.0) + "s span at frame \(spanStartFrame)"
            )
            return
        }

        let spanStartSec = Double(max(0, spanStartFrame)) * ASRConstants.secondsPerEncoderFrame
        (accumulatedTokenIds, accumulatedTokenTimings) = Self.mergeRescuedTokens(
            liveIds: accumulatedTokenIds,
            liveTimings: accumulatedTokenTimings,
            rescuedIds: stagedIds,
            rescuedTimings: stagedTimings,
            langTagTokenIds: config.langTagTokenIds,
            spanStartSec: spanStartSec
        )
        recordSuccessfulRescue()
        logger.info(
            "Blank-span rescue recovered \(stagedIds.count) token(s) from a "
                + String(format: "%.2f", Double(span.count) / 16000.0) + "s span at frame \(spanStartFrame)"
        )
        if let callback = savedPartialCallback {
            callback(tokenizer.decode(ids: accumulatedTokenIds).text)
        }
    }

    /// Insert rescued tokens at the span's timestamp position instead of
    /// appending: when a dropped span and later speech share one chunk, the
    /// later words are already in the accumulators by the time the span
    /// closes, and a plain append would order them "later-word rescued-word".
    ///
    /// `liveTimings` is 1:1 with the non-lang-tag entries of `liveIds`
    /// (lang-tag tokens never get timings — see `appendTokenTiming`);
    /// `rescuedIds`/`rescuedTimings` are 1:1 (lang tags already dropped).
    nonisolated internal static func mergeRescuedTokens(
        liveIds: [Int],
        liveTimings: [TokenTiming],
        rescuedIds: [Int],
        rescuedTimings: [TokenTiming],
        langTagTokenIds: Set<Int>,
        spanStartSec: Double
    ) -> (ids: [Int], timings: [TokenTiming]) {
        let insertKey = rescuedTimings.first?.startTime ?? spanStartSec
        let timingInsertIndex =
            liveTimings.firstIndex { $0.startTime > insertKey } ?? liveTimings.count

        // Map the timing index onto the ids array by walking past lang-tag
        // entries, which occupy an id slot but no timing slot.
        var idsInsertIndex = liveIds.count
        var timedSeen = 0
        for (index, id) in liveIds.enumerated() {
            if timedSeen == timingInsertIndex, !langTagTokenIds.contains(id) {
                idsInsertIndex = index
                break
            }
            if !langTagTokenIds.contains(id) {
                timedSeen += 1
            }
        }

        var ids = liveIds
        var timings = liveTimings
        ids.insert(contentsOf: rescuedIds, at: idsInsertIndex)
        timings.insert(contentsOf: rescuedTimings, at: timingInsertIndex)
        return (ids, timings)
    }

    /// True when the raw token piece carries letters/digits (any script) —
    /// i.e. is more than word-boundary markers and punctuation.
    nonisolated internal static func containsLexicalContent(_ rawToken: String) -> Bool {
        rawToken.unicodeScalars.contains { CharacterSet.alphanumerics.contains($0) }
    }

    /// Fresh caches + decoder state for the rescue decode. Mirrors
    /// `resetStates()` but leaves transcript accumulation, language
    /// detection, and span bookkeeping untouched.
    private func resetStatesForRescue() throws {
        let cacheConfig = EncoderCacheManager.CacheConfig(
            channelShape: config.cacheChannelShape,
            timeShape: config.cacheTimeShape,
            lenShape: [1]
        )
        let caches = try EncoderCacheManager.createInitialCaches(config: cacheConfig)
        cacheChannel = caches.channel
        cacheTime = caches.time
        cacheLen = caches.len
        if #available(macOS 15, iOS 18, *) {
            if let enc = encoder, !enc.modelDescription.stateDescriptionsByName.isEmpty {
                encoderState = enc.makeState()
            }
        }
        // Same 1-frame preamble as resetStates(): keeps the encoder's
        // slice_by_index off the zero-length-slice path.
        cacheLen?[0] = 1
        melCache = nil
        prefetchedMel = nil
        prefetchedEncoded = nil
        prefetchedEncoderProj = nil
        prefetchedCacheChannel = nil
        prefetchedCacheTime = nil
        prefetchedCacheLen = nil
        hState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden])
        cState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden])
        lastToken = Int32(config.blankIdx)
    }

    /// Reset span bookkeeping. Called from `resetStates()` on session reset.
    internal func resetRescueState() {
        rescueFrameCursor = 0
        rescueSpanOpen = false
        rescueSpanStartFrame = 0
        rescueSpanLastSpeechFrame = 0
        rescueSpanSpeechWindows = 0
        rescueSpanPreRollFrames = 0
        rescueSilentWindowRun = 0
        rescueSpanAudio = []
        rescueSpanOverflowed = false
        rescuePreRollTail = []
        inBlankRescue = false
        detectedBlankSpanCount = 0
        blankRescueCount = 0
    }
}
