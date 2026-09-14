import XCTest

@testable import FluidAudioCLI

/// Tests for the CLI ASR scoring normalizer (issue #911).
///
/// Swift dictionaries iterate in a per-instance random order, so before the
/// tables were applied longest-key-first, overlapping keys ("n't" vs "'t")
/// could expand "don't" differently between the reference and hypothesis
/// calls of a single WER computation — moving CI benchmark rows between runs
/// with no code change.
final class AsrTextNormalizerTests: XCTestCase {

    /// The exact record from issue #911 that scored WER 0% or 10% depending
    /// on dictionary iteration order.
    func testIssue911RecordScoresZero() {
        let hypothesis = "You don't mean that you thought me so silly."
        let reference = "YOU DON'T MEAN THAT YOU THOUGHT ME SO SILLY"

        for _ in 0..<100 {
            let metrics = WERCalculator.calculateWERAndCER(
                hypothesis: hypothesis, reference: reference)
            XCTAssertEqual(metrics.wer, 0.0)
            XCTAssertEqual(metrics.cer, 0.0)
        }
    }

    /// Repeated calls must produce byte-identical output. Each call builds
    /// fresh replacement dictionaries, so any order-dependence shows up as
    /// call-to-call drift even within one process.
    func testNormalizeIsDeterministicAcrossCalls() {
        let inputs = [
            "You don't mean that you thought me so silly.",
            "I'd been there before and it's been a long day",
            "Can't you see he's got it? Let's go, she'll come, won't she?",
            "Mr Smith met Dr Jones at one hundred and twenty three Main St",
        ]
        for input in inputs {
            let first = TextNormalizer.normalize(input)
            for _ in 0..<100 {
                XCTAssertEqual(TextNormalizer.normalize(input), first)
            }
        }
    }

    /// Longest-key-first application: the specific rule must win over its
    /// substring ("n't" over "'t", "'d been" over "'d", "can't" over "n't").
    func testOverlappingContractionsExpandCorrectly() {
        XCTAssertEqual(TextNormalizer.normalize("don't"), "do not")
        XCTAssertEqual(TextNormalizer.normalize("can't"), "can not")
        XCTAssertEqual(TextNormalizer.normalize("won't"), "will not")
        XCTAssertEqual(TextNormalizer.normalize("let's"), "let us")
        XCTAssertEqual(TextNormalizer.normalize("I'd been there"), "i had been there")
        XCTAssertEqual(TextNormalizer.normalize("It's been a long day"), "it has been a long day")
        XCTAssertEqual(TextNormalizer.normalize("he's got it"), "he has got it")
        XCTAssertEqual(TextNormalizer.normalize("she'll come"), "she will come")
    }
}
