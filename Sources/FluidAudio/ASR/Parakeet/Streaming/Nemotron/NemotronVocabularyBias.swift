@preconcurrency import CoreML
import Foundation

/// Decode-time shallow-fusion hotword biasing for the Nemotron streaming
/// RNN-T greedy decode loop.
///
/// The sliding-window (Parakeet TDT) path boosts custom vocabulary by
/// rescoring decoded text against CTC log-probabilities
/// (`SlidingWindow/CustomVocabulary/`). The Nemotron multilingual export
/// ships no CTC head, so this biases inside the greedy decode instead: at
/// every emission step the tokens that would *extend a partially matched
/// vocabulary term* receive a flat log-prob bonus, and the boosted best is
/// emitted when it beats the unbiased argmax. Adding a constant before the
/// argmax is equivalent to boosting the token's log-softmax probability by
/// the same amount (the normalizer is shared), so the bonus lives on the
/// same "context-biasing weight" scale as the CTC path's — see
/// `defaultBoost`.
///
/// Matching is on *piece text*, not on one fixed token segmentation:
/// whichever segmentation of the term the decoder is drifting toward stays
/// boostable, because every vocab piece that continues the term from a
/// viable match offset is a candidate. Three details carry the
/// multilingual weight:
///
/// - Comparison runs in NFC **Unicode-scalar space**, because SentencePiece
///   pieces split grapheme clusters (a Devanagari conjunct arrives as
///   `स्` + `ते`, an Arabic diacritic as a lone mark) and a cluster-level
///   tail could never re-join them.
/// - **Every viable match offset** is tracked, not just the longest —
///   a term that repeats its leading word ("ab ab c" hearing "ab ab ab c")
///   stays boostable through the shorter overlap.
/// - Space-delimited scripts anchor at word starts (the term's piece form
///   leads with `▁`, and `▁` appears only at word boundaries), which is
///   what keeps "ran" from matching into "CRAN" — the over-fire mode
///   issue #702 documents; the `minTermLength` guard applies here too.
///   CJK terms are indexed *unanchored* instead (the shipped vocab marks
///   almost no CJK piece with `▁`) and accept 2-cluster terms (a
///   two-character CJK word is a full word, not a short keyword).
///
/// Scaling: the fresh-start candidate set (offset 0 for every term) does
/// not depend on decode state, so it is computed once at init. Per step,
/// only terms whose prefix matches a suffix of the emitted tail need work;
/// those are found through a trie over term piece-forms, so the per-step
/// cost is bounded by the tail length and the number of *live* partial
/// matches — not by the vocabulary size.
///
/// Greedy-decoding caveat: this biases a greedy argmax, so once a boosted
/// token wins it is committed — there is no beam to recover an over-fire.
/// Chunk boundaries are NOT a barrier, unlike the sliding-window CTC path:
/// the decoder state and this match tail both persist across chunks (only
/// `reset()`/`finish()` clear them), so a term whose audio spans a
/// boundary keeps its bias. Verified by a boundary-phase sweep — leading
/// silence stepped through a full chunk period leaves term recovery intact
/// at every phase.
final class NemotronVocabularyBias {

    /// One boostable token continuation at the current match state.
    struct Candidate {
        let tokenId: Int
        let boost: Float
    }

    private struct Entry {
        /// NFC scalars of the term's piece form: `"▁steve▁jobs"`, or the
        /// bare characters for an unanchored (CJK-initial) term.
        let pieceForm: [Unicode.Scalar]
        let boost: Float
    }

    /// Trie over entry piece-forms. Each node at depth `d` records every
    /// entry whose form starts with the node's path, so one walk of a tail
    /// suffix yields all entries continuable from offset `d`.
    private final class TrieNode {
        var children: [UInt32: TrieNode] = [:]
        var entryIndices: [Int] = []
    }

    /// Terms shorter than this (letters, not counting whitespace) are
    /// skipped — the CTC rescorer's own guard against short-keyword
    /// over-firing. CJK surfaces accept two clusters (see `isUsable`).
    static let minTermLength = 3

    /// Per-token log-prob bonus for terms without their own weight.
    /// Measured on the issue #841 rig (8 rare-term clips + 4 phonetic
    /// neighbours + 2 neutrals): 3.0 recalled 2/8, 4.5 recalled 5/8, 6.0
    /// fell back to 4/8 with over-boost artifacts ("build today" splitting
    /// into "build to day"); every weight left neighbours and neutrals
    /// untouched. 4.5 is also the cbw the tuned CTC rescorer settled on
    /// (`ContextBiasingConstants.rescorerConfig`).
    static let defaultBoost: Float = 4.5

    /// Ceiling for per-term `weight` overrides. `CustomVocabularyTerm.weight`
    /// predates this engine and is set to 10.0 by the simple text-list
    /// loader — a CTC-rescoring scale, not a per-token logit bonus. Applied
    /// raw it would over-bias badly (degradation is measurable from 6.0),
    /// so weights above this fall back to `defaultBoost` (see
    /// `effectiveBoost`).
    static let maxBoost: Float = 6.0

    private let entries: [Entry]
    private let trieRoot: TrieNode
    /// Offset-0 candidates for every term — decode-state independent, so
    /// computed once. `candidates()` starts from a copy of this map.
    private let freshStartCandidates: [Int: Float]
    /// Lowercased NFC piece text → every token id whose piece folds to it
    /// (`"▁the"` → ids of `"▁The"`, `"▁the"`, …).
    private let idsByPiece: [String: [Int]]
    /// Lowercased NFC piece per id, for tracking emissions. Special pieces
    /// (`<unk>`, lang tags) are absent — they never advance a match.
    private let pieceById: [Int: String]
    /// Lowercased NFC scalars of the most recent emissions, long enough to
    /// hold any term's partial match.
    private var tail: [Unicode.Scalar] = []
    private let tailCap: Int
    private var cached: [Candidate]?

    /// Builds the biasing state, or `nil` when no usable term remains.
    ///
    /// - Parameters:
    ///   - terms: vocabulary; `weight` is a per-term flat log-prob bonus
    ///     override (clamped to `maxBoost`), `aliases` are additional
    ///     surface forms boosted (and emitted) as themselves.
    ///   - pieces: the model's id → piece table (`▁` markers intact, case
    ///     preserved).
    ///   - defaultBoost: bonus for terms without a `weight`.
    init?(
        terms: [CustomVocabularyTerm],
        pieces: [Int: String],
        defaultBoost: Float = NemotronVocabularyBias.defaultBoost
    ) {
        var entries: [Entry] = []
        var maxLen = 0
        for term in terms {
            let boost = Self.effectiveBoost(of: term, defaultBoost: defaultBoost)
            for surface in Self.usableSurfaces(of: term, defaultBoost: defaultBoost) {
                let form = Self.pieceForm(surface)
                entries.append(Entry(pieceForm: form, boost: boost))
                maxLen = max(maxLen, form.count)
            }
        }
        guard !entries.isEmpty else { return nil }
        self.entries = entries
        self.tailCap = maxLen

        var idsByPiece: [String: [Int]] = [:]
        var pieceById: [Int: String] = [:]
        for (id, piece) in pieces {
            // Special tokens (`<unk>`, `<en-US>`, …) are not text the
            // decoder spells words with; letting one into the index would
            // boost it, and letting one into the tail would corrupt the
            // match position.
            if piece.hasPrefix("<") && piece.hasSuffix(">") { continue }
            let folded = piece.lowercased().precomposedStringWithCanonicalMapping
            idsByPiece[folded, default: []].append(id)
            pieceById[id] = folded
        }
        self.idsByPiece = idsByPiece
        self.pieceById = pieceById

        let root = TrieNode()
        for (index, entry) in entries.enumerated() {
            var node = root
            for scalar in entry.pieceForm {
                let next =
                    node.children[scalar.value]
                    ?? {
                        let created = TrieNode()
                        node.children[scalar.value] = created
                        return created
                    }()
                next.entryIndices.append(index)
                node = next
            }
        }
        self.trieRoot = root

        var fresh: [Int: Float] = [:]
        for entry in entries {
            Self.accumulate(from: entry, offset: 0, idsByPiece: idsByPiece, into: &fresh)
        }
        self.freshStartCandidates = fresh
    }

    /// Whether a term would survive this engine's hygiene — the one copy of
    /// the rule callers use to keep their "applied" accounting honest.
    static func isUsable(_ term: CustomVocabularyTerm) -> Bool {
        !usableSurfaces(of: term, defaultBoost: defaultBoost).isEmpty
    }

    /// The per-token bonus a term will actually receive: its `weight` when
    /// set and within `(0, maxBoost]`, the default otherwise. A weight above
    /// `maxBoost` was tuned for the CTC rescoring scale (the simple text-list
    /// loader assigns 10.0 to every term), not for this engine — treating it
    /// as "untuned" and using the measured default beats pinning those terms
    /// to the aggressive ceiling.
    static func effectiveBoost(of term: CustomVocabularyTerm, defaultBoost: Float = defaultBoost) -> Float {
        guard let weight = term.weight, weight <= maxBoost else { return defaultBoost }
        return weight
    }

    /// The surfaces of `term` (text + aliases) that pass the length and
    /// weight guards.
    private static func usableSurfaces(
        of term: CustomVocabularyTerm, defaultBoost: Float
    ) -> [String] {
        guard effectiveBoost(of: term, defaultBoost: defaultBoost) > 0 else { return [] }
        return ([term.text] + (term.aliases ?? [])).filter { surface in
            let letters = surface.filter { !$0.isWhitespace }
            let minimum = surface.unicodeScalars.contains(where: isCJK) ? 2 : minTermLength
            return letters.count >= minimum
        }
    }

    /// The SentencePiece word-boundary marker.
    private static let marker: Unicode.Scalar = "\u{2581}"

    /// Han, kana and Hangul — scripts the vocab writes with unmarked
    /// pieces, where `▁`-anchoring would make every term inert.
    private static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3040...0x30FF,  // hiragana + katakana
            0x3400...0x4DBF, 0x4E00...0x9FFF,  // Han
            0xAC00...0xD7AF, 0xF900...0xFAFF:  // Hangul syllables, compat Han
            return true
        default:
            return false
        }
    }

    /// `"Steve Jobs"` → the NFC scalars of `"▁steve▁jobs"`; a CJK-initial
    /// term stays unanchored (no markers at all), because the vocab spells
    /// CJK with unmarked pieces.
    private static func pieceForm(_ text: String) -> [Unicode.Scalar] {
        let folded = text.lowercased().precomposedStringWithCanonicalMapping
        let anchored = !(folded.unicodeScalars.first.map(isCJK) ?? false)
        var form: [Unicode.Scalar] = []
        var atBoundary = true
        for ch in folded {
            if ch.isWhitespace {
                atBoundary = true
                continue
            }
            if atBoundary {
                if anchored { form.append(marker) }
                atBoundary = false
            }
            form.append(contentsOf: ch.unicodeScalars)
        }
        return form
    }

    /// Record an emitted token so the match state follows the decode.
    /// Call for every committed non-blank emission, in order.
    func observe(_ tokenId: Int) {
        guard let piece = pieceById[tokenId] else { return }
        tail.append(contentsOf: piece.unicodeScalars)
        if tail.count > tailCap {
            tail.removeFirst(tail.count - tailCap)
        }
        cached = nil
    }

    /// Forget the match state (stream reset / finish). The vocabulary
    /// itself survives, like the selected language does.
    func resetMatchState() {
        tail.removeAll(keepingCapacity: true)
        cached = nil
    }

    /// The boostable continuations at the current match state, one entry
    /// per token id with the strongest boost that reaches it. Cached until
    /// the next `observe`/`resetMatchState`.
    ///
    /// For each term, every viable offset is live: the fresh start (a new
    /// word can begin at any time, precomputed) plus every strict prefix of
    /// the term the emitted tail currently ends with (found by walking each
    /// tail suffix through the piece-form trie). The candidates are every
    /// vocab piece equal to a prefix of the term's remaining text from any
    /// of those offsets.
    func candidates() -> [Candidate] {
        if let cached { return cached }
        var best = freshStartCandidates
        if !tail.isEmpty {
            let maxOffset = min(tail.count, tailCap)
            for start in (tail.count - maxOffset)..<tail.count {
                var node = trieRoot
                var matched = true
                for i in start..<tail.count {
                    guard let next = node.children[tail[i].value] else {
                        matched = false
                        break
                    }
                    node = next
                }
                guard matched else { continue }
                let offset = tail.count - start
                for index in node.entryIndices {
                    Self.accumulate(
                        from: entries[index], offset: offset, idsByPiece: idsByPiece, into: &best)
                }
            }
        }
        let result = best.map { Candidate(tokenId: $0.key, boost: $0.value) }
        cached = result
        return result
    }

    /// Merge the boostable pieces that continue `entry` from `offset` —
    /// every vocab piece equal to a prefix of `pieceForm[offset...]`.
    private static func accumulate(
        from entry: Entry, offset: Int, idsByPiece: [String: [Int]], into best: inout [Int: Float]
    ) {
        let form = entry.pieceForm
        guard offset < form.count else { return }
        var piece = ""
        var letters = 0
        for scalar in form[offset...] {
            piece.unicodeScalars.append(scalar)
            if scalar != marker { letters += 1 }
            // The bare `▁` piece is a prefix of every word and narrows
            // nothing; a standing boost on it would sprinkle stray
            // word-boundary tokens through unrelated speech.
            guard piece != String(marker), let ids = idsByPiece[piece] else {
                continue
            }
            // Entry evidence: opening a fresh word needs at least two
            // letters. Single-letter word-initial pieces ("▁a", "▁n") are
            // among the vocab's most common tokens, and a standing boost on
            // one inserts stray word starts through unrelated speech —
            // every step of every utterance is a fresh-start opportunity.
            // A continuation (offset > 0) may still narrow an open match
            // one letter at a time, and an unanchored CJK term's single
            // characters stay (each is a selective full syllable).
            if offset == 0, form.first == marker, letters < 2 { continue }
            for id in ids where entry.boost > best[id, default: -.infinity] {
                best[id] = entry.boost
            }
        }
    }
}

/// `FLUIDAUDIO_BIAS_LOG=1` traces every boosted flip on stderr — which
/// piece the plain argmax wanted, which the vocabulary promoted, and both
/// logits. Off by default; diagnostic rigs read it to attribute
/// split/insert artifacts to the exact flip that caused them.
let nemotronBiasLogEnabled: Bool = {
    let value = ProcessInfo.processInfo.environment["FLUIDAUDIO_BIAS_LOG"] ?? ""
    return !(value.isEmpty || value == "0" || value.lowercased() == "false")
}()

extension StreamingNemotronMultilingualAsrManager {

    /// Configure decode-time hotword biasing (the Nemotron counterpart of
    /// the sliding-window path's vocabulary boosting; no CTC models
    /// needed). Pass an empty list to disable. Survives `reset()`; takes
    /// effect from the next emission. May be called before models load —
    /// the vocabulary is (re)bound whenever a tokenizer becomes available.
    ///
    /// `weight` is a per-token log-prob bonus; most terms should omit it
    /// (default 4.5). Weights above `NemotronVocabularyBias.maxBoost` are
    /// treated as legacy CTC-scale values and fall back to the default.
    /// Biasing requires a logits-producing step decoder: one is loaded
    /// lazily from the model directory when the tier priority skipped it,
    /// and when none exists the vocabulary is rejected with an error log
    /// rather than silently half-applied.
    public func setCustomVocabulary(_ terms: [CustomVocabularyTerm]) async {
        vocabularyTerms = terms
        await rebuildVocabularyBias()
    }

    /// Bind the stored terms to the loaded tokenizer. Called from the model
    /// load path and from `setCustomVocabulary`.
    internal func rebuildVocabularyBias() async {
        guard !vocabularyTerms.isEmpty, let tokenizer else {
            vocabularyBias = nil
            return
        }
        let anyStepDecoder =
            decoderJointNoEncProj != nil || decoderJointArgmax != nil || decoderJoint != nil
            || (decoder != nil && joint != nil)
        var logitsStepDecoder =
            decoderJointNoEncProj != nil || decoderJoint != nil || (decoder != nil && joint != nil)
        if anyStepDecoder && !logitsStepDecoder {
            // The normal load path skips the logits-producing fused decoders
            // when B2 wins the tier priority, but the bundle may still ship
            // one — bring up B1 (plain `encoder` input, works with every
            // encoder) before rejecting.
            await loadLogitsStepDecoderForVocabulary()
            logitsStepDecoder = decoderJoint != nil
        }
        if anyStepDecoder && !logitsStepDecoder {
            // B2-only asset set: the fused-argmax model never exposes
            // logits, so no decode site can be biased. All-or-nothing —
            // a silently unbiased decode reads exactly like a weak boost.
            vocabularyBias = nil
            logger.error(
                "Custom vocabulary requires a logits-producing step decoder, but only the fused-argmax "
                    + "(decoder_joint_argmax) asset is loaded. Vocabulary biasing is DISABLED. Ship the "
                    + "decoder_joint_noencproj or decoder_joint asset to enable it.")
            return
        }
        let legacyWeights = vocabularyTerms.filter { ($0.weight ?? 0) > NemotronVocabularyBias.maxBoost }
        if !legacyWeights.isEmpty {
            logger.warning(
                "Custom vocabulary: \(legacyWeights.count) term(s) with weight > "
                    + "\(NemotronVocabularyBias.maxBoost) use the \(NemotronVocabularyBias.defaultBoost) "
                    + "default instead (a weight that large is a CTC rescoring value, not a per-token "
                    + "logit bonus)")
        }
        // The multilingual wrapper hides the base vocabulary map, but ids
        // are contiguous and `rawToken(for:)` is exact — rebuild the table.
        var pieces: [Int: String] = [:]
        pieces.reserveCapacity(config.vocabSize)
        for id in 0..<config.vocabSize {
            if let piece = tokenizer.rawToken(for: id) {
                pieces[id] = piece
            }
        }
        vocabularyBias = NemotronVocabularyBias(terms: vocabularyTerms, pieces: pieces)
        if vocabularyBias == nil {
            logger.warning("Custom vocabulary: no usable terms after length/weight guards")
        } else if decoderJointNoEncProj == nil && decoderJointArgmax != nil {
            logger.info(
                "Custom vocabulary active: preferring the logits-producing step decoder over the "
                    + "fused-argmax (decoder_joint_argmax) asset so emissions can be biased")
        }
    }

    /// Load the B1 fused decoder (`decoder_joint`) from the remembered model
    /// directory so an active vocabulary has a logits-producing step decoder
    /// on a bundle where B2 won the tier priority. B1 is the safe choice:
    /// it takes the plain `encoder` step every decode path already has,
    /// whereas B3 (`decoder_joint_noencproj`) also needs the encoder to emit
    /// `encoder_proj` — loading a model the decode loop can't feed would
    /// silently fall back to unbiased B2, the exact failure this avoids.
    private func loadLogitsStepDecoderForVocabulary() async {
        guard decoderJoint == nil, let directory = modelDirectory else { return }
        do {
            guard
                let fusedURL = try await locateOptionalModelBundle(
                    in: directory, compiled: "decoder_joint.mlmodelc",
                    uncompiled: "decoder_joint.mlpackage")
            else { return }
            decoderJoint = try await MLModel.load(contentsOf: fusedURL, configuration: mlConfiguration)
            logger.info(
                "Loaded decoder_joint for custom vocabulary — the fused-argmax (B2) path exposes no "
                    + "logits, so biased decoding uses B1")
        } catch {
            logger.warning(
                "Custom vocabulary: failed to load decoder_joint from \(directory.path): "
                    + "\(error.localizedDescription)")
        }
    }

    /// When hotword biasing is active the fused-argmax step decoder (B2)
    /// must yield to a logits-producing alternative — no logits, no bias.
    internal var vocabularyBiasPrefersLogits: Bool {
        vocabularyBias != nil && (decoderJoint != nil || (decoder != nil && joint != nil))
    }

    /// Argmax with hotword shallow fusion, for decode sites whose logits
    /// are the contiguous Float32 layout `findMaxIndex` assumes. Returns
    /// the plain argmax unless a vocabulary candidate's boosted logit
    /// beats it.
    internal func selectToken(_ logits: MLMultiArray) -> Int {
        let plain = findMaxIndex(logits)
        guard let bias = vocabularyBias else { return plain }
        let candidates = bias.candidates()
        guard !candidates.isEmpty else { return plain }
        let count = logits.count
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: count)
        let picked = Self.pickBiased(plain: plain, candidates: candidates, count: count) { ptr[$0] }
        if picked != plain {
            traceBiasFlip(plain: plain, picked: picked, plainLogit: ptr[plain], pickedLogit: ptr[picked])
        }
        return picked
    }

    /// The comparison itself, layout-agnostic: `logit` reads one vocab
    /// index. Shared by `selectToken` and the speculative scan (whose
    /// logits are strided and possibly Float16).
    internal static func pickBiased(
        plain: Int,
        candidates: [NemotronVocabularyBias.Candidate],
        count: Int,
        logit: (Int) -> Float
    ) -> Int {
        var bestId = plain
        var bestScore = logit(plain)
        for candidate in candidates where candidate.tokenId < count {
            let score = logit(candidate.tokenId) + candidate.boost
            if score > bestScore {
                bestScore = score
                bestId = candidate.tokenId
            }
        }
        return bestId
    }

    /// Emit one `FLUIDAUDIO_BIAS_LOG` trace line for a boosted flip. Every
    /// biased site reports through here — the speculative scan included,
    /// which is where most flips happen (it is the one site that always
    /// has logits); a drains-only trace undercounts and mis-attributes.
    internal func traceBiasFlip(plain: Int, picked: Int, plainLogit: Float, pickedLogit: Float) {
        guard nemotronBiasLogEnabled else { return }
        let plainPiece =
            plain == config.blankIdx ? "<blank>" : (tokenizer?.rawToken(for: plain) ?? "?")
        let pickedPiece = tokenizer?.rawToken(for: picked) ?? "?"
        FileHandle.standardError.write(
            Data("bias-flip: '\(plainPiece)'(\(plainLogit)) -> '\(pickedPiece)'(\(pickedLogit))\n".utf8))
    }
}
