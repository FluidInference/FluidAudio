import XCTest

@testable import FluidAudio

/// Guard the stateful → stateless decode rename. The HF repo
/// `FluidInference/CosyVoice3-0.5B-coreml` ships only `LLM-Decode-M768-fp16`
/// (non-stateful, external KV cache); resurrecting `-stateful` here would
/// re-break the download path and regress macOS 14 support.
final class CosyVoice3ModelNameTests: XCTestCase {

    // MARK: - ModelNames.CosyVoice3

    func testLlmDecodeIsStatelessName() {
        XCTAssertEqual(ModelNames.CosyVoice3.llmDecode, "LLM-Decode-M768-fp16")
        XCTAssertFalse(
            ModelNames.CosyVoice3.llmDecode.contains("stateful"),
            "llmDecode must not reference the dropped stateful variant")
    }

    func testLlmDecodeFileMatchesBaseName() {
        XCTAssertEqual(
            ModelNames.CosyVoice3.llmDecodeFile,
            "LLM-Decode-M768-fp16.mlmodelc")
    }

    func testRequiredModelsContainsStatelessDecode() {
        XCTAssertTrue(
            ModelNames.CosyVoice3.requiredModels.contains("LLM-Decode-M768-fp16.mlmodelc"),
            "requiredModels must list the stateless decode bundle")
        XCTAssertFalse(
            ModelNames.CosyVoice3.requiredModels.contains(
                "LLM-Decode-M768-fp16-stateful.mlmodelc"),
            "requiredModels must not list the dropped stateful bundle")
    }

    func testRequiredModelsHasFourEntries() {
        XCTAssertEqual(
            ModelNames.CosyVoice3.requiredModels.count, 4,
            "Pipeline ships exactly 4 CoreML bundles: prefill, decode, flow, hift")
    }

    // MARK: - CosyVoice3Constants.Files

    func testFilesLlmDecodeIsStatelessPackage() {
        XCTAssertEqual(
            CosyVoice3Constants.Files.llmDecode,
            "LLM-Decode-M768-fp16.mlpackage")
        XCTAssertFalse(
            CosyVoice3Constants.Files.llmDecode.contains("stateful"))
    }

    func testFilesLlmDecodeSubdirIsRenamed() {
        XCTAssertEqual(
            CosyVoice3Constants.Files.llmDecodeSubdir,
            "llm-fp16-decode",
            "Local-build subdir must be the renamed stateless directory")
        XCTAssertFalse(
            CosyVoice3Constants.Files.llmDecodeSubdir.contains("stateful"))
    }
}
