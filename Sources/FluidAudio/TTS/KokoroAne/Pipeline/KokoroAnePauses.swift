#if TTS
import Foundation

/// Punctuation pause overrides (``KokoroAneManager/setPunctuationPauses(_:)``).
///
/// Kokoro's pause at a punctuation mark is the duration PostAlbert predicts
/// for its token (Mandarin `、` / `，` get 50–125 ms). An override rewrites
/// that duration. Stretching the token alone does not make a pause: Prosody
/// keeps predicting voiced F0 across it and the vocoder holds a sound. So the
/// stretched frames are also rendered as silence — F0 0 and energy at the
/// level of the BOS/EOS frames — keeping one predicted frame at each edge so
/// the decoder shapes the transitions.
enum KokoroAnePauses {

    /// One acoustic frame: 600 samples at 24 kHz.
    static let frameMilliseconds = 25.0

    /// Tokens (indices into the chain's input ids, BOS = 0) whose summed
    /// duration becomes `frames`; `scalars` is where they sit in the
    /// phoneme string (Unicode-scalar offsets).
    struct Span: Equatable, Sendable {
        let tokens: Range<Int>
        let frames: Int32
        var scalars: Range<Int> = 0..<0
    }

    /// `duration` in acoustic frames, rounded, at least 1.
    static func frames(for duration: Duration) -> Int32 {
        let components = duration.components
        let milliseconds = Double(components.seconds) * 1000 + Double(components.attoseconds) / 1e15
        let frames = (milliseconds / frameMilliseconds).rounded()
        return Int32(min(max(frames, 1), Double(KokoroAneConstants.maxAcousticFrames)))
    }

    /// The spans `pauses` asks for in `phonemes`. `sources` maps the
    /// Unicode-scalar offset of each mark that came from the input text to
    /// the mark as written; given it, a mark the frontend added (the
    /// Mandarin `/` between words) is never paused, and `nil` treats every
    /// mark as written. A mark is looked up as written, then as itself.
    /// Whitespace extends a span but is not a key. Consecutive marks and the
    /// whitespace after them form one span that takes the longest of their
    /// pauses. Scalars the vocab drops are skipped, as `KokoroAneVocab.encode`
    /// does.
    static func spans(
        phonemes: String, vocab: KokoroAneVocab, sources: [Int: Character]?, pauses: [Character: Int32]
    ) -> [Span] {
        guard !pauses.isEmpty else { return [] }
        var spans: [Span] = []
        var open: (start: Int, end: Int, frames: Int32, scalars: Range<Int>)?
        func close() {
            if let run = open {
                spans.append(Span(tokens: run.start..<run.end, frames: run.frames, scalars: run.scalars))
            }
            open = nil
        }
        var token = 0
        for (offset, scalar) in phonemes.unicodeScalars.enumerated() {
            let mark = Character(scalar)
            guard vocab.map[mark] != nil else { continue }
            token += 1
            if mark.isWhitespace {
                if let run = open {
                    open = (run.start, token + 1, run.frames, run.scalars.lowerBound..<(offset + 1))
                }
                continue
            }
            let written = sources == nil ? mark : sources?[offset]
            if let written, let frames = pauses[written] ?? pauses[mark] {
                open = (
                    open?.start ?? token, token + 1, max(open?.frames ?? 0, frames),
                    (open?.scalars.lowerBound ?? offset)..<(offset + 1)
                )
            } else {
                close()
            }
        }
        close()
        return spans
    }

    /// `phonemes` cut after the pause span nearest its middle, each part with
    /// its `sources` rebased: a pass is capped at
    /// ``KokoroAneConstants/maxAcousticFrames``, and a cut at a pause falls
    /// in silence. `nil` when no span ends inside the string.
    static func split(
        _ phonemes: String, atPauseNearestMiddle spans: [Span], sources: [Int: Character]?
    ) -> [(phonemes: String, sources: [Int: Character]?)]? {
        let scalars = phonemes.unicodeScalars.map(Character.init)
        let middle = scalars.count / 2
        let cuts = spans.map(\.scalars.upperBound).filter { $0 > 0 && $0 < scalars.count }
        guard let cut = cuts.min(by: { abs($0 - middle) < abs($1 - middle) }) else { return nil }
        return [0..<cut, cut..<scalars.count].map { piece(of: scalars, $0, sources: sources) }
    }

    /// `scalars[range]` as a string, with the `sources` inside it rebased.
    static func piece(
        of scalars: [Character], _ range: Range<Int>, sources: [Int: Character]?
    ) -> (phonemes: String, sources: [Int: Character]?) {
        let rebased = sources.map { all in
            Dictionary(
                uniqueKeysWithValues: all.compactMap {
                    range.contains($0.key) ? ($0.key - range.lowerBound, $0.value) : nil
                })
        }
        return (String(scalars[range]), rebased)
    }

    /// Rewrite `durations` so each span sums to its frame count (every token
    /// keeps at least one frame; a shorter pause trims from the first token
    /// on). Returns the acoustic-frame ranges to render as silence: the
    /// lengthened spans less one frame at each edge.
    static func apply(_ spans: [Span], to durations: inout [Int32]) -> [Range<Int>] {
        var lengthened: [Range<Int>] = []
        for span in spans where !span.tokens.isEmpty && span.tokens.upperBound <= durations.count {
            let predicted = span.tokens.reduce(0) { $0 + Int(durations[$1]) }
            let target = max(Int(span.frames), span.tokens.count)
            if target > predicted {
                durations[span.tokens.lowerBound] += Int32(target - predicted)
                lengthened.append(span.tokens)
            } else {
                var excess = predicted - target
                for token in span.tokens where excess > 0 {
                    let cut = min(excess, Int(durations[token]) - 1)
                    durations[token] -= Int32(cut)
                    excess -= cut
                }
            }
        }
        guard !lengthened.isEmpty else { return [] }
        var offsets = [0]
        for duration in durations { offsets.append(offsets[offsets.count - 1] + Int(duration)) }
        return lengthened.compactMap { tokens in
            let start = offsets[tokens.lowerBound] + 1
            let end = offsets[tokens.upperBound] - 1
            return start < end ? start..<end : nil
        }
    }

    /// Set `f0` to 0 and `n` to the median energy of the BOS and EOS frames
    /// over `frames`. Prosody's curves run at a whole multiple of the frame
    /// rate given by `durations` (two values per frame).
    static func silence(f0: inout [Float], n: inout [Float], frames: [Range<Int>], durations: [Int32]) throws {
        let total = durations.reduce(0) { $0 + Int($1) }
        guard let first = durations.first, let last = durations.last, total > 0, f0.count == n.count,
            !f0.isEmpty, f0.count % total == 0
        else {
            throw KokoroAneError.unexpectedOutputShape(
                stage: KokoroAneStage.prosody.rawValue, expected: "F0/N at a multiple of \(total) frames",
                got: "F0 \(f0.count), N \(n.count)")
        }
        let perFrame = f0.count / total
        let edges = (n.prefix(Int(first) * perFrame) + n.suffix(Int(last) * perFrame)).sorted()
        let level = edges[edges.count / 2]
        for range in frames {
            for index in (range.lowerBound * perFrame)..<min(range.upperBound * perFrame, f0.count) {
                f0[index] = 0
                n[index] = level
            }
        }
    }
}
#endif
