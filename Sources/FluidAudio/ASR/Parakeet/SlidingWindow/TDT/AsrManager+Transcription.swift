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

    /// Cross-window emission jitter allowance for a window re-decode: a
    /// re-decoded token can land a few frames from its original emission, so the
    /// suppression cutoff backs off this much and dedup strips what remains.
    internal static let redecodeEmissionJitterFrames = 5

    /// Decoder-entry plan for a streaming window that follows accumulated
    /// tokens (issue #855 for the final window, #897 for every other one).
    ///
    /// Returns `initialTimeIndexOverride: 0` so the decoder re-decodes the window
    /// from frame 0 (a mid-window entry with the carried state can blank out the
    /// rest of the window), plus an emission cutoff in window-local frames:
    /// tokens for audio the previous windows already emitted are suppressed at
    /// the source, leaving dedup only the jitter margin. `transcribeChunk` pairs
    /// the frame-0 entry with a *fresh* decoder state — see the note there.
    /// `redecode == false` (the first window) and callers without accumulated
    /// timestamps get `(nil, nil)` — the legacy navigation.
    nonisolated internal static func redecodePlan(
        redecode: Bool,
        previousTokens: [Int],
        previousTokenTimestamps: [Int]?,
        globalFrameOffset: Int,
        lastWordStartFrame: Int? = nil
    ) -> (initialTimeIndexOverride: Int?, emitTokensAfterFrame: Int?) {
        guard redecode, let previousTimestamps = previousTokenTimestamps, !previousTokens.isEmpty else {
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

    /// Whether the model's pieces mark word starts. tdtJa's vocabulary is unspaced:
    /// a window emits a word-start piece only where its decode begins, so the
    /// word rules would read the whole previous window as its last word and
    /// retire it (#1118). It re-decodes a time span instead (`unspacedTailStart`),
    /// and no window opens mid-word — as the batch splice guard is off for it (#1032).
    nonisolated internal static func marksWordStarts(_ modelVersion: AsrModelVersion?) -> Bool {
        modelVersion != .tdtJa
    }

    /// Where the re-decode retires the previous window's last word, never inside
    /// the prefix a token subscriber was already told is stable: a continuation
    /// piece (an apostrophe suffix) can re-attach the last word to a stable one.
    /// Pure, for testability.
    nonisolated internal static func redecodeWordStart(pieces: [String], stableTokenFloor: Int) -> Int? {
        guard let start = trailingWordStartIndex(pieces: pieces) else { return nil }
        let floored = max(start, stableTokenFloor)
        return floored < pieces.count ? floored : nil
    }

    /// How far before an unspaced window's last token its tail reaches. The decoder
    /// guesses up to ~0.5 s before a window edge ("理解します。" where the next words
    /// are "スペイン語を…", #1118); 1 s covers that with margin.
    internal static let unspacedTailFrames = 12

    /// The unspaced counterpart of `redecodeWordStart`: the first token within
    /// `unspacedTailFrames` of the previous last token, at or after the stable
    /// floor and the previous window's first token (a tail reaching into an older
    /// window would not end the text state), and never before `minFrame` (where
    /// the next window's audio has warmed up — a tail it cannot re-decode is not
    /// its to retire). Nil when that would drop index 0 or nothing. Timestamps
    /// are global frames. Pure.
    nonisolated internal static func unspacedTailStart(
        timestamps: [Int], stableTokenFloor: Int, previousWindowStart: Int = 0, minFrame: Int = Int.min,
        tailFrames: Int = unspacedTailFrames
    ) -> Int? {
        guard let last = timestamps.max() else { return nil }
        let startFrame = max(last - tailFrames, minFrame)
        guard let start = timestamps.firstIndex(where: { $0 >= startFrame }), start > 0 else { return nil }
        let floored = max(start, stableTokenFloor, previousWindowStart)
        return floored < timestamps.count ? floored : nil
    }

    /// Seam reconciliation for an unspaced vocabulary (#1118). The re-decode,
    /// which hears audio on both sides of the previous window's tail, replaces
    /// the tail when it covers it: it emits inside the tail and reaches past it.
    /// Otherwise the tail stays and the re-decode's copy of it goes. Returns how
    /// many trailing previous and leading current tokens to drop. Timestamps are
    /// global frames. Pure, for testability.
    nonisolated internal static func reconcileUnspacedSeam(
        previousTimestamps: [Int],
        tailStart: Int,
        currentTimestamps: [Int],
        jitterFrames: Int = redecodeEmissionJitterFrames
    ) -> (droppedPrevious: Int, droppedCurrent: Int) {
        guard tailStart > 0, tailStart < previousTimestamps.count,
            let previousLast = previousTimestamps.max(),
            let currentFirst = currentTimestamps.first, let currentLast = currentTimestamps.max()
        else { return (0, 0) }
        if currentFirst <= previousLast, currentLast > previousLast + jitterFrames {
            return (previousTimestamps.count - tailStart, 0)
        }
        return (0, currentTimestamps.prefix(while: { $0 <= previousLast }).count)
    }

    /// Whether the first token of an unspaced re-decode is a single-token copy of
    /// the last kept previous token (`全` re-emitted ahead of a replaced tail, `る`
    /// just past a kept one), which dedup leaves: it matches runs of two or more.
    /// It matches only the last kept token; an equal character further back (the
    /// `い` of `多い` ahead of `いわれ`) is the re-decode's own. `unspacedKeptCopyCount`
    /// builds on it for the other kept tokens a re-decode re-hears (#1139).
    /// Timestamps are global frames. Pure.
    nonisolated internal static func isUnspacedJitterCopy(
        keptTokens: [Int],
        keptTimestamps: [Int],
        token: Int,
        timestamp: Int,
        jitterFrames: Int = redecodeEmissionJitterFrames,
        frameTolerance: Int = 2 * redecodeEmissionJitterFrames
    ) -> Bool {
        guard let lastToken = keptTokens.last, let lastFrame = keptTimestamps.last,
            keptTokens.count == keptTimestamps.count
        else { return false }
        return token == lastToken && abs(timestamp - lastFrame) <= frameTolerance
            && timestamp <= lastFrame + jitterFrames
    }

    /// How many leading tokens of an unspaced re-decode re-hear kept previous
    /// tokens, which dedup leaves when they are single or segmented another way
    /// (#1139). The re-decode emits from `jitterFrames` ahead of the replaced tail,
    /// so it can re-emit the last kept tokens: `心 的な` over kept `心 的`, or `的な`
    /// over kept `的 で`. A token copies the kept token it starts inside (from
    /// `jitterFrames` before it through the next previous token's frame, where a
    /// merged `的な` often lands) when both begin with the same character, unless
    /// that next token begins with it too and is at least as close (an early
    /// re-emission of a retired `でき` after a kept `で`).
    /// A jitter copy of the last kept token also counts (`isUnspacedJitterCopy`).
    /// Copies go in order and use each kept token once, so the second `こ` of
    /// `ここ` stays. `previous*` hold the accumulated tokens, of which the first
    /// `keptCount` stay. Timestamps are global frames. Pure, for testability.
    nonisolated internal static func unspacedKeptCopyCount(
        previousTokens: [Int],
        previousPieces: [String],
        previousTimestamps: [Int],
        keptCount: Int,
        currentTokens: [Int],
        currentPieces: [String],
        currentTimestamps: [Int],
        jitterFrames: Int = redecodeEmissionJitterFrames
    ) -> Int {
        guard keptCount > 0, keptCount <= previousTokens.count,
            previousPieces.count == previousTokens.count, previousTimestamps.count == previousTokens.count,
            currentPieces.count == currentTokens.count, currentTimestamps.count == currentTokens.count,
            let firstFrame = currentTimestamps.first
        else { return 0 }
        func leadingCharacter(_ piece: String) -> Character? {
            guard piece != ASRConstants.sentencePieceUnknownToken else { return nil }
            return piece.drop(while: { String($0) == ASRConstants.sentencePieceWordBoundary || $0 == " " }).first
        }
        let last = keptCount - 1
        // A copy starts no earlier than inside the last kept token before the
        // re-decode's first token.
        var next = previousTimestamps.prefix(keptCount).lastIndex(where: { $0 < firstFrame }) ?? 0
        var copies = 0
        while copies < currentTokens.count, next < keptCount {
            let frame = currentTimestamps[copies]
            let character = leadingCharacter(currentPieces[copies])
            let copied = (next..<keptCount).first { kept in
                if kept == last,
                    isUnspacedJitterCopy(
                        keptTokens: [previousTokens[last]], keptTimestamps: [previousTimestamps[last]],
                        token: currentTokens[copies], timestamp: frame)
                {
                    return true
                }
                guard let character, kept + 1 < previousTimestamps.count else { return false }
                let start = previousTimestamps[kept]
                let following = previousTimestamps[kept + 1]
                guard frame >= start - jitterFrames, frame <= following,
                    character == leadingCharacter(previousPieces[kept])
                else { return false }
                return character != leadingCharacter(previousPieces[kept + 1]) || following - frame > abs(frame - start)
            }
            guard let copied else { break }
            next = copied + 1
            copies += 1
        }
        return copies
    }

    /// How many leading current tokens re-emit the stable start of a word the
    /// floor split. The re-decode starts at the word, not at the floor, so that
    /// copy must go before reconciliation or the stable part would repeat.
    /// Timestamps are global frames. Pure, for testability.
    nonisolated internal static func reemittedStableWordHead(
        previousTokens: [Int],
        previousTimestamps: [Int],
        wordStart: Int,
        flooredStart: Int,
        currentTokens: [Int],
        currentTimestamps: [Int],
        frameTolerance: Int = 2 * redecodeEmissionJitterFrames
    ) -> Int {
        guard wordStart < flooredStart, flooredStart <= previousTokens.count,
            previousTimestamps.count == previousTokens.count,
            currentTimestamps.count == currentTokens.count
        else { return 0 }
        let head = flooredStart - wordStart
        guard currentTokens.count >= head else { return 0 }
        for offset in 0..<head {
            let previous = wordStart + offset
            guard currentTokens[offset] == previousTokens[previous],
                abs(currentTimestamps[offset] - previousTimestamps[previous]) <= frameTolerance
            else { return 0 }
        }
        return head
    }

    /// The words of `pieces[0..<upTo]` as (core text, start frame, start index),
    /// for matching re-emitted words at the seam. Pure.
    nonisolated internal static func words(
        in pieces: [String], timestamps: [Int], upTo: Int
    ) -> [(core: String, frame: Int, start: Int)] {
        var result: [(core: String, frame: Int, start: Int)] = []
        var index = 0
        let limit = min(upTo, pieces.count, timestamps.count)
        while index < limit {
            guard startsWordPiece(in: pieces, at: index) else {
                index += 1
                continue
            }
            let end = min(wordExtent(in: pieces, from: index), limit)
            result.append((wordCore(Array(pieces[index..<end])), timestamps[index], index))
            index = end
        }
        return result
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

    /// Comparison form of a word's pieces: boundary markers and surrounding
    /// whitespace/punctuation stripped, lower-cased, curly apostrophes
    /// normalized — but *interior* apostrophes and hyphens kept, so `well` and
    /// `we'll` (or `cant` and `can't`) never compare equal. Empty when the
    /// pieces carry no word.
    nonisolated internal static func wordCore<S: Sequence>(_ pieces: S) -> String where S.Element == String {
        let joined = pieces.joined()
            .replacingOccurrences(of: ASRConstants.sentencePieceWordBoundary, with: " ")
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .lowercased()
        let scalars = Array(joined.unicodeScalars)
        let isEdge: (Unicode.Scalar) -> Bool = {
            CharacterSet.whitespaces.contains($0) || CharacterSet.punctuationCharacters.contains($0)
        }
        guard let first = scalars.firstIndex(where: { !isEdge($0) }),
            let last = scalars.lastIndex(where: { !isEdge($0) })
        else { return "" }
        return String(String.UnicodeScalarView(scalars[first...last]))
    }

    /// A piece that joins two halves of one word (`we` `'` `ll`, `co` `-` `op`):
    /// a *bare* apostrophe or hyphen. The v3 vocabulary also carries
    /// boundary-marked variants (`▁'`, `▁-`, normalized to a leading space)
    /// that start a new word; those are never joiners.
    nonisolated internal static func isJoiningPunctuationPiece(_ piece: String) -> Bool {
        guard !piece.hasPrefix(ASRConstants.sentencePieceWordBoundary), !piece.hasPrefix(" ") else {
            return false
        }
        return piece == "'" || piece == "\u{2019}" || piece == "-"
    }

    /// Whether the piece at `index` starts a word: a boundary-marked piece that
    /// is not punctuation — or a boundary-marked apostrophe/hyphen immediately
    /// followed by a continuation piece (`▁'` `cause`), which is the first
    /// piece of that word rather than leading punctuation.
    nonisolated internal static func startsWordPiece(in pieces: [String], at index: Int) -> Bool {
        let p = pieces[index]
        guard p.hasPrefix(ASRConstants.sentencePieceWordBoundary) || p.hasPrefix(" ") else { return false }
        if !isPunctuationPiece(p) { return true }
        let core = p.replacingOccurrences(of: ASRConstants.sentencePieceWordBoundary, with: "")
            .trimmingCharacters(in: .whitespaces)
        guard core == "'" || core == "\u{2019}" || core == "-", index + 1 < pieces.count else { return false }
        let next = pieces[index + 1]
        return !next.hasPrefix(ASRConstants.sentencePieceWordBoundary) && !next.hasPrefix(" ")
            && !isPunctuationPiece(next)
    }

    /// Exclusive end of the word that starts at `start`: continuation pieces
    /// follow, and a joining apostrophe/hyphen piece is absorbed when a
    /// continuation piece follows it.
    nonisolated internal static func wordExtent(in pieces: [String], from start: Int) -> Int {
        func startsWord(_ p: String) -> Bool {
            p.hasPrefix(ASRConstants.sentencePieceWordBoundary) || p.hasPrefix(" ")
        }
        var end = start + 1
        while end < pieces.count {
            let p = pieces[end]
            if !startsWord(p), !isPunctuationPiece(p) {
                end += 1
            } else if isJoiningPunctuationPiece(p), end + 1 < pieces.count, !startsWord(pieces[end + 1]),
                !isPunctuationPiece(pieces[end + 1])
            {
                end += 2
            } else {
                break
            }
        }
        return end
    }

    /// The pieces of the first word in a token sequence: from the first
    /// word-start piece through the end of that word (see `wordExtent`).
    nonisolated internal static func firstWordPieces(_ pieces: [String]) -> [String] {
        func startsWord(_ p: String) -> Bool {
            p.hasPrefix(ASRConstants.sentencePieceWordBoundary) || p.hasPrefix(" ")
        }
        guard let start = pieces.indices.first(where: { startsWordPiece(in: pieces, at: $0) }) else {
            return []
        }
        return Array(pieces[start..<wordExtent(in: pieces, from: start)])
    }

    /// How far a re-reading of a kept word may start from it. The same
    /// mid-window word decoded twice lands within 2 frames (`thirty`/`30` 0,
    /// `safe`/`say` 1, `a`/`the` 2); a word the re-decode inserts after a kept one
    /// starts later (`a` 3 frames after `hedge`, `to` 4 after `went`).
    internal static let respelledWordFrames = 2

    /// The kept word (before the last word) that a differently spelled
    /// re-emitted word `core` starting at `frame` re-reads (#1113), or nil. Exact
    /// re-emissions after it anchor the alignment: in `the knowledge` over a kept
    /// `a knowledge`, `the` re-reads `a` though it starts on `knowledge`'s frame.
    /// Without an anchor, frames alone cannot tell the word before the last word
    /// from the last word itself (`Saviour` landing early), so the nearest kept
    /// word must also be nearer than the last word, within the jitter margin,
    /// and spelled alike (`spellingsAgree`); ties go to the later word. Pure.
    nonisolated internal static func respelledWord(
        at frame: Int,
        core: String,
        following: [(core: String, frame: Int)],
        keptWords: [(core: String, frame: Int, start: Int)],
        lastWord: (core: String, frame: Int),
        jitterFrames: Int = redecodeEmissionJitterFrames,
        frameTolerance: Int = 2 * redecodeEmissionJitterFrames
    ) -> (core: String, frame: Int, start: Int)? {
        let candidates = keptWords.map { (core: $0.core, frame: $0.frame) } + [lastWord]
        for (offset, word) in following.enumerated() {
            guard
                let anchor = candidates.lastIndex(where: {
                    $0.core == word.core && abs($0.frame - word.frame) <= frameTolerance
                })
            else { continue }
            // A split or merged word (`forever` → `for ever`) shifts the
            // alignment; the start frames then disagree and the nearest word decides.
            let target = anchor - offset - 1
            if target >= 0, abs(keptWords[target].frame - frame) <= respelledWordFrames { return keptWords[target] }
            break
        }
        func distance(_ word: (core: String, frame: Int, start: Int)) -> Int { abs(word.frame - frame) }
        guard
            let nearest = keptWords.min(by: {
                distance($0) < distance($1) || (distance($0) == distance($1) && $0.start > $1.start)
            }),
            distance(nearest) <= jitterFrames, distance(nearest) < abs(lastWord.frame - frame),
            spellingsAgree(core, nearest.core)
        else { return nil }
        return nearest
    }

    /// Whether two spellings could be one word read twice: one extends the
    /// other (`for` / `forever`, a split) or they differ in at most half their
    /// letters (`word` / `world`). Pure.
    nonisolated internal static func spellingsAgree(_ a: String, _ b: String) -> Bool {
        guard !a.isEmpty, !b.isEmpty else { return false }
        if min(a.count, b.count) >= 2, a.hasPrefix(b) || b.hasPrefix(a) { return true }
        return 2 * StringUtils.levenshteinDistance(a, b) <= max(a.count, b.count)
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
    ///    fate. Same word starting within the previous word's span: keep the
    ///    previous copy (it carries the sentence-final punctuation a re-decode
    ///    at the audio end omits), drop the re-emission and — when the kept copy
    ///    ends with punctuation — the re-decode's punctuation behind it. A same
    ///    word starting later is a genuine repetition and stays.
    ///    Previous text a strict prefix of it (`an`→`analyzing`,
    ///    `every`→`everything`): a fragment, retire. A different word that
    ///    overlaps the previous word's span and was started by the re-decode
    ///    itself (no continuation head): it disagrees with more context, retire.
    ///    Behind a continuation head the first real word is the next word by
    ///    construction, so only the prefix rule can retire. A different word
    ///    starting later: the re-decode skipped the previous word, keep it.
    ///    Before that, re-emitted earlier words are consumed: an identical
    ///    spelling at the same frame is dropped, and a different spelling of a
    ///    kept word (`thirty` → `30`, `respelledWord`) retires the kept words
    ///    from it on instead of only the last word (#1113).
    /// 3. Retiring additionally requires the re-decode to reach past the
    ///    previous window's last frame; an empty or early-ending final window
    ///    keeps the previous word.
    /// 4. The re-decode's head is then stripped of seam artifacts: punctuation
    ///    at or before the previous last word's frame, and tokens — punctuation
    ///    included — that duplicate a kept previous token within
    ///    `frameTolerance` inside the jitter region.
    ///
    /// Returns how many trailing previous tokens to drop (from a re-spelled
    /// earlier word, the whole last word, or none) and how many leading current
    /// tokens to drop. Nothing below `stableTokenFloor` is dropped. Pure, for
    /// testability; timestamps are global frames. Both piece arrays must be
    /// aligned with their token arrays, otherwise the result is a no-op.
    nonisolated internal static func reconcileFinalWindowSeam(
        previousTokens: [Int],
        previousTimestamps: [Int],
        trailingWordStart: Int,
        currentTokens: [Int],
        currentTimestamps: [Int],
        currentPieces: [String] = [],
        previousPieces: [String] = [],
        suppressedPieces: [String] = [],
        suppressedTimestamps: [Int] = [],
        stableTokenFloor: Int = 0,
        jitterFrames: Int = redecodeEmissionJitterFrames,
        frameTolerance: Int = 2 * redecodeEmissionJitterFrames
    ) -> (droppedPrevious: Int, droppedCurrent: Int) {
        // The decision is by piece text, so both piece arrays must be aligned
        // with their token arrays; without them the only safe answer is a no-op
        // (an unknown piece would otherwise read as a continuation and drop).
        guard trailingWordStart > 0, trailingWordStart < previousTokens.count,
            previousTimestamps.count == previousTokens.count,
            currentTimestamps.count == currentTokens.count,
            currentPieces.count == currentTokens.count,
            previousPieces.count == previousTokens.count,
            !currentTokens.isEmpty
        else { return (0, 0) }

        func piece(_ index: Int) -> String { currentPieces[index] }
        // Classify by piece text only. Token ids are model-dependent:
        // `ASRConstants.punctuationTokens` was written for an older vocabulary
        // and maps to `й` / `ó` in v3.
        func isPunctuation(_ index: Int) -> Bool { isPunctuationPiece(piece(index)) }
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
        while head < currentTokens.count {
            if !startsWord(head), !isPunctuation(head) {
                head += 1
            } else if isJoiningPunctuationPiece(piece(head)), head + 1 < currentTokens.count,
                !startsWord(head + 1), !isPunctuation(head + 1)
            {
                head += 2
            } else {
                break
            }
        }

        // 2. Re-emitted earlier words. The cutoff backs off by the jitter margin,
        // so the re-decode can re-emit the word(s) *before* the previous last
        // word (`new` ahead of `code`). Consume them as whole words when the kept
        // previous output already has that word at the same frame — otherwise
        // the retire rules below would read a re-emitted `new` as the
        // replacement for `code`, and a jitter duplicate whose continuation
        // piece falls a frame outside the margin would leave `uld` behind.
        // A re-emitted word spelled differently (`thirty` → `30`, #1113) replaces
        // the kept words from the one it re-reads (`respelledWord`): the
        // re-decode heard more context. A word a token subscriber already holds
        // as stable keeps its spelling, and the re-reading goes instead.
        let previousWords = words(in: previousPieces, timestamps: previousTimestamps, upTo: trailingWordStart)
        let lastWord = (core: wordCore(previousPieces.dropFirst(trailingWordStart)), frame: lastWordStartFrame)
        let currentWords = words(in: currentPieces, timestamps: currentTimestamps, upTo: currentPieces.count)
        var consumed = head
        var consumedKeptStart = -1
        var respelledStart: Int?
        while consumed < currentTokens.count {
            if isPunctuation(consumed), !startsWordPiece(in: currentPieces, at: consumed),
                currentTimestamps[consumed] <= lastWordStartFrame
            {
                consumed += 1
                continue
            }
            guard startsWordPiece(in: currentPieces, at: consumed) else { break }
            let end = wordExtent(in: currentPieces, from: consumed)
            let core = wordCore(Array(currentPieces[consumed..<end]))
            let frame = currentTimestamps[consumed]
            guard frame <= lastWordStartFrame + jitterFrames, !core.isEmpty else { break }
            if let kept = previousWords.last(where: { $0.core == core && abs($0.frame - frame) <= frameTolerance }) {
                consumedKeptStart = max(consumedKeptStart, kept.start)
                consumed = end
                continue
            }
            // The last word's own re-reading — the same word, or a fragment and
            // its completion — is step 3's call wherever its frame lands.
            if !lastWord.core.isEmpty, core.hasPrefix(lastWord.core) || lastWord.core.hasPrefix(core) { break }
            let following = currentWords.filter { $0.start >= end && $0.frame <= lastWordStartFrame + frameTolerance }
            guard
                let target = respelledWord(
                    at: frame, core: core, following: following.map { ($0.core, $0.frame) },
                    keptWords: previousWords, lastWord: lastWord, jitterFrames: jitterFrames,
                    frameTolerance: frameTolerance),
                target.start > consumedKeptStart
            else { break }
            // Retiring from the first accumulated token, into the stable prefix,
            // or without a re-decode that reaches past the previous window is not
            // possible: the kept spelling stays and the re-reading goes instead.
            if target.start > 0, target.start >= stableTokenFloor, extendsBeyondPrevious {
                respelledStart = target.start
                break
            }
            consumed = end
        }

        // 3. The first real word of the re-decode.
        let previousWord = wordCore(previousPieces.dropFirst(trailingWordStart))
        let firstWord = firstWordPieces(Array(currentPieces.dropFirst(consumed)))
        let currentWord = wordCore(firstWord)
        let firstWordIndex = (consumed..<currentTokens.count).first { startsWordPiece(in: currentPieces, at: $0) }
        let firstWordFrame = firstWordIndex.map { currentTimestamps[$0] }

        let overlapsPrevious = firstWordFrame.map { $0 <= previousLastFrame + jitterFrames } ?? false
        let retire: Bool
        if !extendsBeyondPrevious || currentWord.isEmpty || previousWord.isEmpty {
            retire = false
        } else if currentWord == previousWord {
            retire = false
        } else if currentWord.hasPrefix(previousWord), overlapsPrevious {
            // A fragment's replacement starts where the fragment started. A
            // later word that merely happens to extend the previous text
            // (`an` … `another`) is a new word; keep the previous one.
            retire = true
        } else if head == 0 || currentTimestamps[0] < lastWordStartFrame, overlapsPrevious {
            // Overlapping different word, and the re-decode started it itself:
            // it disagrees with more context (`Savior.` → `Saviour`). A
            // continuation head that belongs to the previous *last* word makes
            // the first real word the *next* word by construction (`box` +
            // `x, but`), so only the prefix rule applies there; a head that
            // continues an earlier word (`ist` of `Christ` ahead of `Savior`)
            // says nothing about the last word.
            retire = true
        } else {
            retire = false
        }

        let retireStart = respelledStart ?? (retire ? trailingWordStart : nil)
        let droppedPrevious = retireStart.map { previousTokens.count - $0 } ?? 0
        let keptPrevious = Array(
            zip(previousTokens, previousTimestamps).prefix(retireStart ?? previousTokens.count))
        let keptLastFrame = keptPrevious.last?.1 ?? -1

        // 4. Strip the seam artifacts from the re-decode's head. When the
        // previous word is kept because the re-decode's first word is the same
        // word, consume that word's whole piece range explicitly — its
        // segmentation, casing or punctuation may differ from the kept copy,
        // so id-level duplicate matching cannot be relied on for it.
        // The re-decode's copy of the last word can drift past the jitter margin
        // (an end-aligned final window re-emits an edge-decoded `out`@247 at 253),
        // so the same-word test compares word starts within the duplicate
        // tolerance, like the re-emitted earlier words above; a later genuine
        // repetition (`go … go again`) is further away than that. A fast
        // repetition inside the tolerance is told apart by evidence: when the
        // decoder already re-emitted the previous word *before* the cutoff (it
        // is in the suppressed list at that word's frame), the visible copy is
        // a second word and stays.
        let previousWordSuppressed =
            suppressedPieces.count == suppressedTimestamps.count
            && words(in: suppressedPieces, timestamps: suppressedTimestamps, upTo: suppressedPieces.count)
                .contains { $0.core == previousWord && abs($0.frame - lastWordStartFrame) <= frameTolerance }
        var droppedCurrent = consumed
        if retireStart == nil, currentWord == previousWord, !previousWordSuppressed, let firstIndex = firstWordIndex,
            abs(currentTimestamps[firstIndex] - lastWordStartFrame) <= frameTolerance
        {
            var end = wordExtent(in: currentPieces, from: firstIndex)
            // Punctuation policy: the kept previous copy already carries its own
            // trailing punctuation, so the re-decode's is a duplicate — drop it.
            // If the previous copy has none, the re-decoded punctuation is the
            // only one and stays.
            // Never a boundary-marked piece that begins the next word (`▁'` `cause`).
            let previousEndsWithPunctuation = previousPieces.last.map { isPunctuationPiece($0) } ?? false
            while previousEndsWithPunctuation, end < currentTokens.count, isPunctuation(end),
                !startsWordPiece(in: currentPieces, at: end)
            {
                end += 1
            }
            droppedCurrent = end
        }
        // Jitter duplicates of kept previous tokens are stripped word by word: a
        // word-start piece goes only together with its continuation pieces, and
        // only when every piece of the word matches (`S` alone must not go and
        // leave `aviour` behind).
        var index = droppedCurrent
        while index < currentTokens.count {
            let frame = currentTimestamps[index]
            if isPunctuation(index), !startsWordPiece(in: currentPieces, at: index), frame <= lastWordStartFrame {
                index += 1
                droppedCurrent = index
                continue
            }
            guard frame <= keptLastFrame + jitterFrames else { break }
            let end =
                startsWordPiece(in: currentPieces, at: index) ? wordExtent(in: currentPieces, from: index) : index + 1
            let duplicated = (index..<end).allSatisfy { position in
                keptPrevious.contains(where: {
                    $0.0 == currentTokens[position] && abs($0.1 - currentTimestamps[position]) <= frameTolerance
                })
            }
            guard duplicated else { break }
            index = end
            droppedCurrent = index
        }
        return (droppedPrevious, droppedCurrent)
    }

    /// The decoder may retire only its trailing word. Use its exact boundary,
    /// not confidence, text suffix guesses or timestamp tolerances. Holding all
    /// tokens when no boundary exists is conservative until terminal drain.
    func stableStreamingTokenCount(_ tokens: [Int], timestamps: [Int] = []) -> Int {
        Self.stableTokenCount(
            pieces: tokens.map { vocabulary[$0] ?? "" }, timestamps: timestamps, modelVersion: modelVersion)
    }

    /// See `stableStreamingTokenCount`; an unspaced model retires its time tail
    /// instead. Pure, for testability.
    nonisolated internal static func stableTokenCount(
        pieces: [String], timestamps: [Int], modelVersion: AsrModelVersion?
    ) -> Int {
        guard marksWordStarts(modelVersion) else {
            guard timestamps.count == pieces.count else { return 0 }
            return unspacedTailStart(timestamps: timestamps, stableTokenFloor: 0) ?? 0
        }
        return trailingWordStartIndex(pieces: pieces) ?? 0
    }

    /// Whether `tokens` open inside a word: their first piece carries no word
    /// boundary (a continuation such as the "y" of "Andre|y", or punctuation),
    /// so the token stream glues it onto the previous token (#912).
    func startsMidWord(_ tokens: [Int]) -> Bool {
        Self.opensMidWord(pieces: tokens.compactMap { vocabulary[$0] }, modelVersion: modelVersion)
    }

    /// See `startsMidWord`. Pure, for testability.
    nonisolated internal static func opensMidWord(pieces: [String], modelVersion: AsrModelVersion?) -> Bool {
        guard marksWordStarts(modelVersion),
            let first = pieces.first(where: { !$0.isEmpty && $0 != ASRConstants.sentencePieceUnknownToken })
        else { return false }
        return !first.hasPrefix(ASRConstants.sentencePieceWordBoundary) && !first.hasPrefix(" ")
    }

    /// The text of the trailing word of `tokens`, from its last word-start piece.
    func trailingWordText(_ tokens: [Int]) -> String {
        convertTokensToText(Array(tokens[stableStreamingTokenCount(tokens)...]))
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
        language: Language? = nil,
        stableTokenFloor: Int = 0,
        previousWindowTokenStart: Int = 0
    ) async throws -> (
        tokens: [Int], timestamps: [Int], confidences: [Float], encoderSequenceLength: Int,
        droppedPreviousTokens: Int
    ) {
        let (alignedSamples, frameAlignedLength) = frameAlignedAudio(
            chunkSamples, allowAlignment: previousTokens.isEmpty)
        let padded = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
        // Every streaming window after the first decodes from frame 0 on a FRESH
        // decoder state, as the batch chunker does for every chunk, with emissions
        // for audio the previous windows already covered suppressed at the source
        // (issue #855 for the final window, #897 for the interior ones). Entering
        // a window mid-way with the carried state — state that already consumed
        // the overlap — is not safe anywhere: after a sentence-final token it can
        // blank across the rest of the window (10 s of speech lost at chunk 7 on
        // a real recording), and re-walking the overlap re-emits it with a
        // different segmentation that dedup cannot match (`environment` vs
        // `air environment`). The 2 s left context plus the previous right
        // context give the fresh state 4 s to re-establish itself before the
        // cutoff. The previous window's last word is re-decoded in full and
        // reconciled afterwards (#897).
        let redecodeWindow = isLastChunk || !previousTokens.isEmpty
        let marksWordStarts = Self.marksWordStarts(modelVersion)
        let trailingWordStart: Int?
        if redecodeWindow, let previousTimestamps = previousTokenTimestamps,
            previousTimestamps.count == previousTokens.count
        {
            trailingWordStart =
                marksWordStarts
                ? Self.redecodeWordStart(
                    pieces: previousTokens.map { vocabulary[$0] ?? "" }, stableTokenFloor: stableTokenFloor)
                // The tail needs a second of this window's own audio ahead of it.
                : Self.unspacedTailStart(
                    timestamps: previousTimestamps, stableTokenFloor: stableTokenFloor,
                    previousWindowStart: previousWindowTokenStart,
                    minFrame: globalFrameOffset + Self.unspacedTailFrames)
        } else {
            trailingWordStart = nil
        }
        let redecodePlan = Self.redecodePlan(
            redecode: redecodeWindow,
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
        var stripsJitterCopies = false

        // An unspaced model replaces the previous window's time tail with the
        // re-decoded one (#1118).
        if redecodePlan.initialTimeIndexOverride == 0, !marksWordStarts, let trailingWordStart,
            let previousTimestamps = previousTokenTimestamps
        {
            let seam = Self.reconcileUnspacedSeam(
                previousTimestamps: previousTimestamps,
                tailStart: trailingWordStart,
                currentTimestamps: currentTimestamps.map { $0 + globalFrameOffset }
            )
            stripsJitterCopies = true
            droppedPrevious = seam.droppedPrevious
            if seam.droppedCurrent > 0 {
                currentTokens.removeFirst(seam.droppedCurrent)
                currentTimestamps.removeFirst(seam.droppedCurrent)
                currentConfidences.removeFirst(min(seam.droppedCurrent, currentConfidences.count))
            }
            if droppedPrevious > 0 {
                effectivePrevious = Array(previousTokens.prefix(trailingWordStart))
                effectivePreviousTimestamps = Array(previousTimestamps.prefix(trailingWordStart))
            }
            if droppedPrevious > 0 || seam.droppedCurrent > 0 {
                logger.debug(
                    "Window seam: dropped \(droppedPrevious) trailing previous token(s), \(seam.droppedCurrent) leading current token(s)"
                )
            }
        } else if redecodePlan.initialTimeIndexOverride == 0, let trailingWordStart,
            let previousTimestamps = previousTokenTimestamps
        {
            // Replace the previous window's (possibly edge-cut) last word with the
            // re-decoded one, and strip the seam artifacts the re-decode emits
            // ahead of it (#897).
            let stableCopy = Self.reemittedStableWordHead(
                previousTokens: previousTokens,
                previousTimestamps: previousTimestamps,
                wordStart: Self.trailingWordStartIndex(pieces: previousTokens.map { vocabulary[$0] ?? "" })
                    ?? trailingWordStart,
                flooredStart: trailingWordStart,
                currentTokens: currentTokens,
                currentTimestamps: currentTimestamps.map { $0 + globalFrameOffset }
            )
            if stableCopy > 0 {
                currentTokens.removeFirst(stableCopy)
                currentTimestamps.removeFirst(stableCopy)
                currentConfidences.removeFirst(min(stableCopy, currentConfidences.count))
            }
            let seam = Self.reconcileFinalWindowSeam(
                previousTokens: previousTokens,
                previousTimestamps: previousTimestamps,
                trailingWordStart: trailingWordStart,
                currentTokens: currentTokens,
                currentTimestamps: currentTimestamps.map { $0 + globalFrameOffset },
                currentPieces: currentTokens.map { vocabulary[$0] ?? "" },
                previousPieces: previousTokens.map { vocabulary[$0] ?? "" },
                suppressedPieces: hypothesis.suppressedTokens.map { vocabulary[$0] ?? "" },
                suppressedTimestamps: hypothesis.suppressedTimestamps,
                stableTokenFloor: stableTokenFloor
            )
            droppedPrevious = seam.droppedPrevious
            if seam.droppedCurrent > 0 {
                currentTokens.removeFirst(seam.droppedCurrent)
                currentTimestamps.removeFirst(seam.droppedCurrent)
                currentConfidences.removeFirst(min(seam.droppedCurrent, currentConfidences.count))
            }
            let keptCount = min(trailingWordStart, previousTokens.count - droppedPrevious)
            effectivePrevious = Array(previousTokens.prefix(keptCount))
            effectivePreviousTimestamps = Array(previousTimestamps.prefix(keptCount))
            if droppedPrevious > 0 || seam.droppedCurrent > 0 {
                logger.debug(
                    "Window seam: dropped \(droppedPrevious) trailing previous token(s), \(seam.droppedCurrent) leading current token(s)"
                )
            }
        }

        // Apply token deduplication if previous tokens are provided
        if !effectivePrevious.isEmpty && !currentTokens.isEmpty {
            // Convert this chunk's local frame timestamps into the same global frame
            // space as `previousTokenTimestamps` so dedup can require temporal adjacency.
            let currentGlobalTimestamps: [Int]? =
                effectivePreviousTimestamps != nil ? currentTimestamps.map { $0 + globalFrameOffset } : nil
            // A re-decoded window only leaks duplicates inside the jitter margin
            // (suppression handles the rest), so the matcher must not reach
            // across it: with the legacy 2 s tolerance a word repeated within
            // 2 s of the seam (`old code and the net new code`) matched its
            // earlier copy and dedup chopped the whole re-decoded prefix.
            let tolerance =
                redecodePlan.initialTimeIndexOverride == 0
                ? 2 * Self.redecodeEmissionJitterFrames : ASRConstants.duplicateFrameTolerance
            var (deduped, removedCount) = removeDuplicateTokenSequence(
                previous: effectivePrevious, current: currentTokens,
                previousTimestamps: effectivePreviousTimestamps,
                currentTimestamps: currentGlobalTimestamps,
                frameTolerance: tolerance,
                punctuationTokens: punctuationTokenIds)
            // Copies dedup left behind are single tokens or re-segmented ones
            // (#1139); after a matched run, an equal token is the next character
            // (`こ` of `ここ`).
            if stripsJitterCopies, removedCount == 0, let previousTimestamps = previousTokenTimestamps,
                let currentGlobalTimestamps
            {
                let copies = Self.unspacedKeptCopyCount(
                    previousTokens: previousTokens,
                    previousPieces: previousTokens.map { vocabulary[$0] ?? "" },
                    previousTimestamps: previousTimestamps,
                    keptCount: effectivePrevious.count,
                    currentTokens: deduped,
                    currentPieces: deduped.map { vocabulary[$0] ?? "" },
                    currentTimestamps: currentGlobalTimestamps)
                deduped.removeFirst(copies)
                removedCount = copies
            }
            let adjustedTimestamps =
                removedCount > 0 ? Array(currentTimestamps.dropFirst(removedCount)) : currentTimestamps
            let adjustedConfidences =
                removedCount > 0
                ? Array(currentConfidences.dropFirst(removedCount)) : currentConfidences

            return (deduped, adjustedTimestamps, adjustedConfidences, encLen, droppedPrevious)
        }

        return (currentTokens, currentTimestamps, currentConfidences, encLen, droppedPrevious)
    }

    /// Timings for a window's raw decoder tokens, one per token in decoder order. Unlike
    /// `processTranscriptionResult`, this keeps `<unk>`: the token-revision stream counts the same tokens
    /// the sliding window accumulates.
    internal func rawTokenTimings(tokenIds: [Int], timestamps: [Int], confidences: [Float]) -> [TokenTiming] {
        createTokenTimings(from: tokenIds, timestamps: timestamps, confidences: confidences)
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

        let (tokenIds, timestamps, confidences, tokenDurations) = droppingUnknownTokens(
            tokenIds: tokenIds, timestamps: timestamps, confidences: confidences, tokenDurations: tokenDurations)
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
