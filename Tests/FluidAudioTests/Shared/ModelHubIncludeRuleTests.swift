import XCTest

@testable import FluidAudio

/// Unit tests for `ModelHub.repoIncludeRule` — the file/directory selection
/// rule behind repo downloads. The regression case: for subPath repos, files
/// inside a CoreML bundle that is not in the required-model patterns must be
/// taken all-or-nothing. The `.json`/`.bin` metadata allowance used to sweep
/// in such a bundle's weights and coremldata while dropping `model.mil`,
/// producing bundles that fail MIL load (StyleTTS2 t64/t128/t256 buckets).
final class ModelHubIncludeRuleTests: XCTestCase {

    private let sub = "iteration_3/compiled"
    private var patterns: [String] {
        // Default styletts2 variant declares the unsized bundles only.
        ["iteration_3/compiled/bert_fp16.mlmodelc/", "iteration_3/compiled/ref_encoder_fp16.mlmodelc/"]
    }

    private func include(_ path: String, isDirectory: Bool = false) -> Bool {
        ModelHub.repoIncludeRule(subPath: sub, patterns: patterns)(path, isDirectory)
    }

    // MARK: - CoreML bundle completeness (regression)

    func testNonRequiredBundleModelMilIsIncluded() {
        // Previously excluded (not .json/.model/.bin, bundle not in patterns).
        XCTAssertTrue(include("iteration_3/compiled/bert_fp16_t128.mlmodelc/model.mil"))
    }

    func testNonRequiredBundleWeightsAndMetadataStayIncluded() {
        XCTAssertTrue(include("iteration_3/compiled/bert_fp16_t128.mlmodelc/weights/weight.bin"))
        XCTAssertTrue(include("iteration_3/compiled/bert_fp16_t128.mlmodelc/coremldata.bin"))
        XCTAssertTrue(include("iteration_3/compiled/bert_fp16_t128.mlmodelc/metadata.json"))
    }

    func testMlpackageInternalFilesAreIncluded() {
        XCTAssertTrue(
            include("iteration_3/compiled/foo.mlpackage/Data/com.apple.CoreML/model.mlmodel"))
    }

    // MARK: - Existing behavior unchanged

    func testRequiredBundleFilesIncluded() {
        XCTAssertTrue(include("iteration_3/compiled/bert_fp16.mlmodelc/model.mil"))
    }

    func testStrayNonBundleFileOutsideAllowancesExcluded() {
        XCTAssertFalse(include("iteration_3/compiled/notes.txt"))
    }

    func testFileOutsideSubPathExcluded() {
        XCTAssertFalse(include("iteration_1/compiled/bert_fp16.mlmodelc/model.mil"))
        XCTAssertFalse(include("README.md"))
    }

    func testSubPathDirectoryTraversal() {
        XCTAssertTrue(include("iteration_3/compiled", isDirectory: true))
        XCTAssertTrue(include("iteration_3/compiled/bert_fp16.mlmodelc", isDirectory: true))
        XCTAssertFalse(include("iteration_1", isDirectory: true))
    }

    func testNoSubPathRepoBehaviorUnchanged() {
        let rule = ModelHub.repoIncludeRule(subPath: nil, patterns: ["Melspectrogram.mlmodelc/"])
        XCTAssertTrue(rule("Melspectrogram.mlmodelc/model.mil", false))
        XCTAssertTrue(rule("config.json", false))
        XCTAssertTrue(rule("vocab.txt", false))
        XCTAssertFalse(rule("Other.mlmodelc/model.mil", false))
    }
}
