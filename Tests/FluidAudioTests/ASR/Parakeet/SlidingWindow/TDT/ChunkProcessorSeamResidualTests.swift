import Foundation
import XCTest

@testable import FluidAudio

/// Issue #683's `spliceSafeTokenIds` word-boundary-aware re-splice closed the
/// original glue/hybrid-word bugs, but three deterministic holes remain in
/// `ChunkProcessor.mergeChunks` that still drop words at chunk seams:
///
/// - Hole A: `popSeamWord`'s `maxPiecesPerWord` cap (12) false-negatives on a
///   seam word whose word-initial piece is more than 12 pieces back, forcing
///   the else-branch fallback which drops the final continuation piece.
/// - Hole B: when the `right` window ends mid-word (no further word-initial
///   piece anywhere in `tail`), `mergeUsingMatches`'s else-branch has no
///   fallback and silently drops every remaining continuation piece.
/// - Hole C: `mergeByMidpoint`'s safe-token scan can walk `rightStart` all the
///   way to `right.count`, discarding the entire right window.
///
/// Each token stream below is fully in the SentencePiece domain: word-initial
/// pieces carry the `▁` boundary marker, continuation pieces do not, and
/// `decode` reconstructs the merged text the same way production detokenizes
/// (word-initial pieces get a leading space, everything else attaches
/// directly to the previous piece).
final class ChunkProcessorSeamResidualTests: XCTestCase {

    private typealias Token = (token: Int, timestamp: Int, confidence: Float, duration: Int)

    private func createMockAudioSamples(durationSeconds: Double, sampleRate: Int = 16000) -> [Float] {
        let sampleCount = Int(durationSeconds * Double(sampleRate))
        return (0..<sampleCount).map { Float($0) / Float(sampleCount) }
    }

    /// Vocabulary covering all five cases below. Token 1 is a plain filler
    /// word reused everywhere as a leading anchor with no seam involvement.
    private let vocabulary: [Int: String] = [
        1: "▁hello",

        // Hole A: seam word segmented by left as "▁super" + 11 continuation
        // pieces (300 + 301...311), with 312 the matched anchor — 13 pieces
        // back to the word-initial, one past popSeamWord's 12-piece cap.
        300: "▁super",
        301: "c1", 302: "c2", 303: "c3", 304: "c4", 305: "c5", 306: "c6",
        307: "c7", 308: "c8", 309: "c9", 310: "c10", 311: "c11",
        312: "c12",
        320: "▁xyz",  // right's own word-initial segmentation of the same word
        313: "c13",  // right's tail continuation piece — currently dropped
        330: "▁there",

        // Hole B: "▁fan" + "tas" (anchor) + "ti" + "cal" == "fantastical".
        // The right window ends immediately after "cal" — no further
        // word-initial piece anywhere in the stream.
        424: "▁fan",
        425: "tas",
        426: "ti",
        427: "cal",

        // Hole C: left is two complete words; right is four continuation-only
        // pieces with no splice-safe token anywhere.
        511: "▁world",
        549: "pre",
        550: "ne",
        551: "ta",
        552: "tion",

        // Control 4: punctuation-adjacent seam, already handled correctly.
        601: "▁word",
        602: "end",
        690: ",",
        640: "▁friend",

        // Control 5: short seam word (well under the 12-piece cap), two
        // legitimate segmentations that must still resolve to right's view.
        724: "▁Gre",
        725: "nl",
        726: "and",
        727: "▁Green",
        728: "andia",
        730: "▁there",
    ]

    private var safeIds: Set<Int> {
        ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary)!
    }

    /// Reconstructs merged text the way production detokenizes: word-initial
    /// pieces start a new (space-separated) word, everything else — including
    /// punctuation — attaches directly to the previous piece.
    private func decode(_ tokens: [Token]) -> String {
        var text = ""
        for entry in tokens {
            let piece = vocabulary[entry.token] ?? ""
            if isWordBoundary(piece) {
                let word = stripWordBoundaryPrefix(piece)
                text += text.isEmpty ? word : " " + word
            } else {
                text += piece
            }
        }
        return text
    }

    // MARK: - Hole A: popSeamWord's 12-piece cap false-negatives on long words

    func testLongSeamWordPastPopSeamWordCapIsNotDropped() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 22.0))

        let left: [Token] = [
            (token: 1, timestamp: 90, confidence: 0.98, duration: 1),
            (token: 300, timestamp: 91, confidence: 0.97, duration: 1),
            (token: 301, timestamp: 92, confidence: 0.97, duration: 1),
            (token: 302, timestamp: 93, confidence: 0.97, duration: 1),
            (token: 303, timestamp: 94, confidence: 0.97, duration: 1),
            (token: 304, timestamp: 95, confidence: 0.97, duration: 1),
            (token: 305, timestamp: 96, confidence: 0.97, duration: 1),
            (token: 306, timestamp: 97, confidence: 0.97, duration: 1),
            (token: 307, timestamp: 98, confidence: 0.97, duration: 1),
            (token: 308, timestamp: 99, confidence: 0.97, duration: 1),
            (token: 309, timestamp: 100, confidence: 0.97, duration: 1),
            (token: 310, timestamp: 101, confidence: 0.97, duration: 1),
            (token: 311, timestamp: 102, confidence: 0.97, duration: 1),
            (token: 312, timestamp: 103, confidence: 0.96, duration: 1),  // anchor
        ]
        let right: [Token] = [
            (token: 320, timestamp: 99, confidence: 0.95, duration: 1),
            (token: 312, timestamp: 103, confidence: 0.96, duration: 1),  // anchor
            (token: 313, timestamp: 104, confidence: 0.95, duration: 1),  // must survive
            (token: 330, timestamp: 105, confidence: 0.97, duration: 1),
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds
        )

        // Right heard the seam word from its own start ("▁xyz" + "c12" +
        // "c13"); the merge must adopt that segmentation in full, including
        // the trailing "c13" continuation piece, instead of keeping left's
        // truncated 13-piece-back view and losing "c13" in the process.
        XCTAssertEqual(merged.map(\.token), [1, 320, 312, 313, 330])
        XCTAssertEqual(decode(merged), "hello xyzc12c13 there")
    }

    // MARK: - Hole B: right window ends mid-word, no further word-initial piece

    func testRightWindowEndingMidWordKeepsTrailingContinuationPieces() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 22.0))

        let left: [Token] = [
            (token: 1, timestamp: 120, confidence: 0.98, duration: 1),
            (token: 424, timestamp: 130, confidence: 0.97, duration: 1),
            (token: 425, timestamp: 131, confidence: 0.96, duration: 1),  // anchor
        ]
        let right: [Token] = [
            (token: 425, timestamp: 131, confidence: 0.96, duration: 1),  // anchor
            (token: 426, timestamp: 132, confidence: 0.95, duration: 1),
            (token: 427, timestamp: 133, confidence: 0.95, duration: 1),  // stream ends here
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds
        )

        // The right window's chunk simply ended mid-word; "ti" and "cal" are
        // real continuation pieces of "fantastical" that left never heard,
        // not glue candidates to be discarded because no further
        // word-initial piece happens to follow them.
        XCTAssertEqual(merged.map(\.token), [1, 424, 425, 426, 427])
        XCTAssertEqual(decode(merged), "hello fantastical")
    }

    // MARK: - Hole C: mergeByMidpoint's safe-token scan can drop the entire right window

    func testMidpointMergeWithNoSafeTokenInRightKeepsRightWindow() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 22.0))

        // Disjoint token IDs on both sides force the LCS/contiguous matchers
        // to find nothing, so mergeChunks falls back to mergeByMidpoint.
        let left: [Token] = [
            (token: 1, timestamp: 120, confidence: 0.98, duration: 1),
            (token: 511, timestamp: 140, confidence: 0.97, duration: 1),
        ]
        let right: [Token] = [
            (token: 549, timestamp: 140, confidence: 0.90, duration: 1),  // before cutoff
            (token: 550, timestamp: 141, confidence: 0.91, duration: 1),  // past cutoff, unsafe
            (token: 551, timestamp: 142, confidence: 0.91, duration: 1),  // past cutoff, unsafe
            (token: 552, timestamp: 143, confidence: 0.91, duration: 1),  // past cutoff, unsafe
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds
        )

        // None of right's post-cutoff pieces are splice-safe, so the
        // safe-token scan must not be allowed to walk off the end of
        // `right` and discard the whole window — it should fall back to the
        // original cutoff-based split instead of returning nothing.
        XCTAssertEqual(merged.map(\.token), [1, 511, 550, 551, 552])
        XCTAssertEqual(decode(merged), "hello worldnetation")
    }

    // MARK: - Control: punctuation-adjacent seam must stay unaffected

    func testPunctuationAdjacentSeamIsUnaffected() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 22.0))

        let left: [Token] = [
            (token: 1, timestamp: 120, confidence: 0.98, duration: 1),
            (token: 601, timestamp: 130, confidence: 0.97, duration: 1),
            (token: 602, timestamp: 131, confidence: 0.96, duration: 1),  // anchor
        ]
        let right: [Token] = [
            (token: 602, timestamp: 131, confidence: 0.96, duration: 1),  // anchor
            (token: 690, timestamp: 132, confidence: 0.97, duration: 1),  // punctuation, safe
            (token: 640, timestamp: 134, confidence: 0.98, duration: 1),
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds
        )

        XCTAssertEqual(merged.map(\.token), [1, 601, 602, 690, 640])
        XCTAssertEqual(decode(merged), "hello wordend, friend")
    }

    // MARK: - Control: short disagreeing segmentation must stay unaffected

    func testShortSeamWordDisagreeingSegmentationIsUnaffected() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 22.0))

        let left: [Token] = [
            (token: 1, timestamp: 120, confidence: 0.98, duration: 1),
            (token: 724, timestamp: 130, confidence: 0.97, duration: 1),
            (token: 725, timestamp: 131, confidence: 0.96, duration: 1),  // anchor
            (token: 726, timestamp: 132, confidence: 0.95, duration: 1),
        ]
        let right: [Token] = [
            (token: 727, timestamp: 130, confidence: 0.97, duration: 1),
            (token: 725, timestamp: 131, confidence: 0.96, duration: 1),  // anchor
            (token: 728, timestamp: 132, confidence: 0.95, duration: 1),
            (token: 730, timestamp: 134, confidence: 0.97, duration: 1),
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds
        )

        // Well within the 12-piece cap, so both before and after removing it
        // the merge must adopt right's segmentation of the whole word.
        XCTAssertEqual(merged.map(\.token), [1, 727, 725, 728, 730])
        XCTAssertEqual(decode(merged), "hello Greennlandia there")
    }
}
