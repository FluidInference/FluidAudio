import XCTest

@testable import FluidAudio

/// Prompt-layout logic checked with a deterministic scalar-per-token encoder; the real tokenizer is
/// covered by `LayaIntegrationTests` against fixtures generated from the HuggingFace tokenizer.
final class LayaSequenceBuilderTests: XCTestCase {
    private static let mask = 4
    private static let cls = 2
    private static let sep = 1

    /// Every Unicode scalar becomes one id offset by 1000, so token counts equal character counts.
    private func builder(headMaxLength: Int = 256) -> LayaSequenceBuilder {
        LayaSequenceBuilder(
            encode: { text in text.unicodeScalars.map { Int($0.value) + 1000 } },
            maskTokenId: Self.mask, clsTokenId: Self.cls, sepTokenId: Self.sep, maskToken: "<mask>",
            headMaxLength: headMaxLength)
    }

    private func ids(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) + 1000 }
    }

    func testLayoutMatchesUpstreamFormat() throws {
        let question = LayaQuestion.choice("Pick", options: ["a", "bb"])
        let sequence = try builder().sequence(state: "state", question: question, maximumLength: 512)
        let head = [Self.cls] + ids("choice question: Pick") + [Self.sep]
        let optionA = [Self.mask] + ids(" a")
        let optionB = [Self.mask] + ids(" bb")
        let expected = head + optionA + optionB + [Self.sep] + ids("state") + [Self.sep]
        XCTAssertEqual(sequence.ids, expected)
        XCTAssertEqual(sequence.markers, [head.count, head.count + optionA.count])
        XCTAssertFalse(sequence.stateWasTruncated)
    }

    func testRenderedOptionsPerType() {
        XCTAssertEqual(
            LayaQuestion.choice("q", options: [.init("yes", description: "do it"), .init("no", description: "")])
                .renderedOptions,
            ["yes: do it", "no"])
        XCTAssertEqual(
            LayaQuestion.score("q", levels: ["low", "high"]).renderedOptions, ["level 0: low", "level 1: high"])
        XCTAssertEqual(
            LayaQuestion.noul("q").renderedOptions,
            ["false: no, the statement does not hold", "true: yes, the statement holds"])
        XCTAssertEqual(
            LayaQuestion.noul("q", falseDescription: "nope", trueDescription: "yep").renderedOptions,
            ["false: nope", "true: yep"])
        XCTAssertEqual(LayaQuestion.noul("q").labels, ["false", "true"])
        XCTAssertEqual(LayaQuestion.score("q", levels: ["a", "b", "c"]).labels, ["0", "1", "2"])
    }

    func testMaskLiteralsAreNeutralised() throws {
        let question = LayaQuestion.noul("Is <mask> here?")
        let sequence = try builder().sequence(state: "x <mask> y", question: question, maximumLength: 512)
        XCTAssertEqual(sequence.ids.filter { $0 == Self.mask }.count, 2)
        XCTAssertEqual(sequence.markers.count, 2)
    }

    func testStateIsTruncatedOnTheRightToFitTheBucket() throws {
        let question = LayaQuestion.noul("q")
        let parts = try builder().parts(state: String(repeating: "s", count: 300), question: question)
        let sequence = try builder().sequence(from: parts, maximumLength: 128)
        XCTAssertEqual(sequence.ids.count, 128)
        XCTAssertEqual(sequence.ids.last, Self.sep)
        XCTAssertEqual(sequence.ids[parts.head.count..<127].count, 128 - parts.head.count - 1)
        XCTAssertTrue(sequence.stateWasTruncated)
        XCTAssertEqual(parts.untruncatedLength, parts.head.count + 300 + 1)
    }

    func testOptionsAreCappedAndBudgetShrinksEvenly() throws {
        // 48-token cap per option: a 60-char option keeps 48 ids after the marker.
        let long = String(repeating: "o", count: 60)
        let capped = try builder().sequence(
            state: "s", question: .choice("q", options: [long, "b"]), maximumLength: 512)
        XCTAssertEqual(capped.markers[1] - capped.markers[0], 1 + 48)

        // Budget of 64 with 8 options of 40 chars leaves < 16, so each option shrinks to (64-16)/8 = 6 ids.
        let options = (0..<8).map { _ in String(repeating: "x", count: 40) }
        let shrunk = try builder(headMaxLength: 64).sequence(
            state: "s", question: .choice("q", options: options), maximumLength: 512)
        for pair in zip(shrunk.markers, shrunk.markers.dropFirst()) {
            XCTAssertEqual(pair.1 - pair.0, 6)
        }
        // Instructions keep at least 8 ids when the option budget is exhausted.
        XCTAssertEqual(shrunk.markers[0], 1 + 8 + 1)
    }

    func testQuestionThatCannotFitTheBucketIsRejected() throws {
        let options = (0..<32).map { "option \($0) with some words" }
        XCTAssertThrowsError(
            try builder().sequence(state: "s", question: .choice("q", options: options), maximumLength: 128)
        ) { error in
            guard case LayaError.promptTooLong = error else { return XCTFail("unexpected \(error)") }
        }
    }

    func testValidation() {
        XCTAssertThrowsError(try LayaQuestion.noul("").validate()) {
            XCTAssertEqual($0 as? LayaError, .emptyInstructions)
        }
        XCTAssertThrowsError(try LayaQuestion.choice("q", options: ["only"]).validate()) {
            XCTAssertEqual($0 as? LayaError, .invalidOptionCount(1))
        }
        XCTAssertThrowsError(try LayaQuestion.score("q", levels: Array(repeating: "l", count: 33)).validate()) {
            XCTAssertEqual($0 as? LayaError, .invalidOptionCount(33))
        }
        XCTAssertThrowsError(try LayaQuestion.choice("q", options: ["a", ""]).validate()) {
            XCTAssertEqual($0 as? LayaError, .emptyOption(1))
        }
        XCTAssertNoThrow(try LayaQuestion.noul("q").validate())
    }

    func testTemperatureBucketsMirrorUpstream() {
        XCTAssertEqual(LayaManager.temperatureBucket(.noul, optionCount: 2), "noul:2")
        XCTAssertEqual(LayaManager.temperatureBucket(.choice, optionCount: 5), "choice:3-5")
        XCTAssertEqual(LayaManager.temperatureBucket(.choice, optionCount: 10), "choice:6-10")
        XCTAssertEqual(LayaManager.temperatureBucket(.score, optionCount: 11), "score:11+")
    }

    func testAnswerConveniences() {
        let answer = LayaAnswer(
            question: .score("q", levels: ["a", "b", "c"]), probabilities: [0.2, 0.5, 0.3],
            rawProbabilities: [0.2, 0.5, 0.3],
            logits: [0, 1, 0.5], confidence: 0.1, actionProbability: 1, tokenCount: 10, bucketLength: 128,
            stateWasTruncated: false)
        XCTAssertEqual(answer.selectedIndex, 1)
        XCTAssertEqual(answer.selectedLabel, "1")
        XCTAssertEqual(answer.expectedScore!, 1.1, accuracy: 1e-6)
        XCTAssertNil(answer.noul)
        let yesNo = LayaAnswer(
            question: .noul("q"), probabilities: [0.3, 0.7], rawProbabilities: [0.3, 0.7], logits: [0, 1],
            confidence: 0.7,
            actionProbability: 1, tokenCount: 5, bucketLength: 128, stateWasTruncated: false)
        XCTAssertEqual(yesNo.noul!, 0.7, accuracy: 1e-6)
        XCTAssertEqual(yesNo.selectedLabel, "true")
    }

    func testModelRegistry() throws {
        XCTAssertEqual(Repo.laya.remotePath, "FluidInference/laya-coreml")
        XCTAssertEqual(Repo.laya.folderName, "laya-coreml")
        XCTAssertEqual(try ModelNames.Laya.modelFile(length: 128), "laya_multilingual_fp16_L128_options32.mlmodelc")
        XCTAssertThrowsError(try ModelNames.Laya.modelFile(length: 96))
        XCTAssertEqual(ModelNames.getRequiredModelNames(for: .laya, variant: nil).count, 4)
    }
}
