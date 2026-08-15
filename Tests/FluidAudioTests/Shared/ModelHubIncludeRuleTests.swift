import XCTest

@testable import FluidAudio

/// Unit tests for `ModelHub.repoIncludeRule` — the file/directory selection
/// rule behind repo downloads. For subPath repos, CoreML bundles are matched
/// all-or-nothing against the required-model patterns:
///
/// - Required bundles are taken whole: the `.json`/`.bin` metadata allowance
///   alone used to sweep in a bundle's weights and coremldata while dropping
///   `model.mil`, producing bundles that fail MIL load (#821, StyleTTS2
///   t64/t128/t256 buckets).
/// - Non-required bundles are skipped whole: the `.bin` allowance used to
///   pull every `weight.bin` inside the uncompiled `.mlpackage` published
///   next to each compiled `.mlmodelc`, so ~half of a first-run download was
///   never loaded (#826, parakeetEou/kokoroAne).
final class ModelHubIncludeRuleTests: XCTestCase {

    // MARK: - StyleTTS2-shaped repo (#821)

    private let styleSub = "iteration_3/compiled"
    private var stylePatterns: [String] {
        // Default styletts2 variant declares the unsized bundles only.
        ["iteration_3/compiled/bert_fp16.mlmodelc/", "iteration_3/compiled/ref_encoder_fp16.mlmodelc/"]
    }

    private func styleInclude(_ path: String, isDirectory: Bool = false) -> Bool {
        ModelHub.repoIncludeRule(subPath: styleSub, patterns: stylePatterns)(path, isDirectory)
    }

    func testRequiredBundleTakenWhole() {
        XCTAssertTrue(styleInclude("iteration_3/compiled/bert_fp16.mlmodelc/model.mil"))
        XCTAssertTrue(styleInclude("iteration_3/compiled/bert_fp16.mlmodelc/weights/weight.bin"))
        XCTAssertTrue(styleInclude("iteration_3/compiled/bert_fp16.mlmodelc/coremldata.bin"))
        XCTAssertTrue(styleInclude("iteration_3/compiled/bert_fp16.mlmodelc/metadata.json"))
    }

    func testNonRequiredBundleSkippedWhole() {
        // No partial bundle: the metadata allowance must not admit a
        // non-required bundle's .json/.bin while dropping model.mil.
        XCTAssertFalse(styleInclude("iteration_3/compiled/bert_fp16_t128.mlmodelc/model.mil"))
        XCTAssertFalse(styleInclude("iteration_3/compiled/bert_fp16_t128.mlmodelc/weights/weight.bin"))
        XCTAssertFalse(styleInclude("iteration_3/compiled/bert_fp16_t128.mlmodelc/coremldata.bin"))
        XCTAssertFalse(styleInclude("iteration_3/compiled/bert_fp16_t128.mlmodelc/metadata.json"))
    }

    // MARK: - parakeetEou-shaped repo (#826)

    private let eouSub = "320ms"
    private var eouPatterns: [String] {
        [
            "320ms/streaming_encoder.mlmodelc/", "320ms/decoder.mlmodelc/",
            "320ms/joint_decision.mlmodelc/", "320ms/vocab.json/",
        ]
    }

    private func eouInclude(_ path: String, isDirectory: Bool = false) -> Bool {
        ModelHub.repoIncludeRule(subPath: eouSub, patterns: eouPatterns)(path, isDirectory)
    }

    func testUncompiledMlpackageSiblingExcluded() {
        // The .mlpackage copy beside each required .mlmodelc is never loaded;
        // its weight.bin must not ride in on the .bin metadata allowance.
        XCTAssertFalse(
            eouInclude("320ms/streaming_encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin"))
        XCTAssertFalse(
            eouInclude("320ms/streaming_encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel"))
        XCTAssertFalse(eouInclude("320ms/streaming_encoder.mlpackage/Manifest.json"))
    }

    func testMlpackageDirectoryPruned() {
        // Directory traversal skips excluded bundles instead of recursing.
        XCTAssertFalse(eouInclude("320ms/streaming_encoder.mlpackage", isDirectory: true))
        XCTAssertFalse(
            eouInclude("320ms/streaming_encoder.mlpackage/Data", isDirectory: true))
    }

    func testRequiredBundleDirectoriesTraversed() {
        XCTAssertTrue(eouInclude("320ms", isDirectory: true))
        XCTAssertTrue(eouInclude("320ms/streaming_encoder.mlmodelc", isDirectory: true))
        XCTAssertTrue(eouInclude("320ms/streaming_encoder.mlmodelc/weights", isDirectory: true))
    }

    func testRequiredBundleFilesIncluded() {
        XCTAssertTrue(eouInclude("320ms/streaming_encoder.mlmodelc/model.mil"))
        XCTAssertTrue(eouInclude("320ms/streaming_encoder.mlmodelc/weights/weight.bin"))
    }

    func testLooseMetadataAllowanceUnchanged() {
        // Aux files outside any bundle still ride the metadata allowance
        // (vocab.json is in patterns but with a trailing slash, so only the
        // allowance admits it; voice packs are loose .bin files).
        XCTAssertTrue(eouInclude("320ms/vocab.json"))
        XCTAssertTrue(eouInclude("320ms/streaming_encoder_metadata.json"))
        XCTAssertTrue(eouInclude("320ms/af_heart.bin"))
    }

    func testStrayNonBundleFileOutsideAllowancesExcluded() {
        XCTAssertFalse(eouInclude("320ms/convert_parakeet_eou.py"))
        XCTAssertFalse(eouInclude("320ms/.DS_Store"))
    }

    func testFileOutsideSubPathExcluded() {
        XCTAssertFalse(eouInclude("160ms/streaming_encoder.mlmodelc/model.mil"))
        XCTAssertFalse(eouInclude("README.md"))
        XCTAssertFalse(eouInclude("160ms", isDirectory: true))
    }

    func testNestedRequiredBundlePatternStillMatches() {
        // Nemotron declares a nested bundle path (encoder/encoder_int8.mlmodelc).
        let rule = ModelHub.repoIncludeRule(
            subPath: "nemotron_coreml_2240ms",
            patterns: ["nemotron_coreml_2240ms/encoder/encoder_int8.mlmodelc/"])
        XCTAssertTrue(rule("nemotron_coreml_2240ms/encoder", true))
        XCTAssertTrue(rule("nemotron_coreml_2240ms/encoder/encoder_int8.mlmodelc", true))
        XCTAssertTrue(rule("nemotron_coreml_2240ms/encoder/encoder_int8.mlmodelc/model.mil", false))
        XCTAssertTrue(
            rule("nemotron_coreml_2240ms/encoder/encoder_int8.mlmodelc/weights/weight.bin", false))
    }

    func testEmptyPatternsDownloadEverythingUnderSubPath() {
        let rule = ModelHub.repoIncludeRule(subPath: "320ms", patterns: [])
        XCTAssertTrue(rule("320ms/streaming_encoder.mlpackage", true))
        XCTAssertTrue(
            rule("320ms/streaming_encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", false))
        XCTAssertTrue(rule("320ms/streaming_encoder.mlmodelc/model.mil", false))
    }

    func testNoSubPathRepoBehaviorUnchanged() {
        let rule = ModelHub.repoIncludeRule(subPath: nil, patterns: ["Melspectrogram.mlmodelc/"])
        XCTAssertTrue(rule("Melspectrogram.mlmodelc/model.mil", false))
        XCTAssertTrue(rule("config.json", false))
        XCTAssertTrue(rule("vocab.txt", false))
        XCTAssertFalse(rule("Other.mlmodelc/model.mil", false))
    }
}
