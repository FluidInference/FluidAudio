import Foundation
import XCTest

@testable import FluidAudio

final class ModelNamesTests: XCTestCase {

    // MARK: - Repo

    func testRepoRemotePathContainsOwner() {
        for repo in Repo.allCases {
            XCTAssertTrue(
                repo.remotePath.contains("FluidInference/"),
                "\(repo) remotePath should contain 'FluidInference/'"
            )
        }
    }

    func testRepoNameIsNonEmpty() {
        for repo in Repo.allCases {
            XCTAssertFalse(repo.name.isEmpty, "\(repo) should have a non-empty name")
        }
    }

    func testRepoFolderNameIsNonEmpty() {
        for repo in Repo.allCases {
            XCTAssertFalse(repo.folderName.isEmpty, "\(repo) should have a non-empty folderName")
        }
    }

    func testRepoSubPathForVariants() {
        XCTAssertEqual(Repo.parakeetEou160.subPath, "160ms")
        XCTAssertEqual(Repo.parakeetEou320.subPath, "320ms")
        XCTAssertEqual(Repo.parakeetEou1280.subPath, "1280ms")
        XCTAssertEqual(Repo.qwen3Asr.subPath, "f32")
        XCTAssertEqual(Repo.qwen3AsrInt8.subPath, "int8")
        XCTAssertNil(Repo.vad.subPath)
        XCTAssertNil(Repo.parakeet.subPath)
    }

    // MARK: - Required Models

    func testGetRequiredModelNamesReturnsNonEmpty() {
        for repo in Repo.allCases {
            let models = ModelNames.getRequiredModelNames(for: repo, variant: nil)
            XCTAssertFalse(models.isEmpty, "\(repo) should have required models")
        }
    }

    func testModelFileExtensions() {
        let validExtensions: Set<String> = [".mlmodelc", ".json", ".bin"]
        let validDirectories: Set<String> = ["constants_bin"]

        for repo in Repo.allCases {
            let models = ModelNames.getRequiredModelNames(for: repo, variant: nil)
            for model in models {
                let hasValidExtension = validExtensions.contains(where: { model.hasSuffix($0) })
                let isKnownDirectory = validDirectories.contains(model)
                XCTAssertTrue(
                    hasValidExtension || isKnownDirectory,
                    "Model '\(model)' for \(repo) should have a valid extension or be a known directory"
                )
            }
        }
    }

    func testDiarizerOfflineVariant() {
        let offlineModels = ModelNames.getRequiredModelNames(for: .diarizer, variant: "offline")
        let onlineModels = ModelNames.getRequiredModelNames(for: .diarizer, variant: nil)

        XCTAssertNotEqual(offlineModels, onlineModels, "Offline and online diarizer should have different model sets")
        XCTAssertTrue(offlineModels.contains("Segmentation.mlmodelc"))
        XCTAssertTrue(offlineModels.contains("FBank.mlmodelc"))
    }

    // MARK: - Sortformer Bundles

    func testSortformerBundleForVariant() {
        for variant in ModelNames.Sortformer.Variant.allCases {
            let bundle = ModelNames.Sortformer.bundle(for: variant)
            XCTAssertTrue(bundle.hasSuffix(".mlmodelc"), "Bundle '\(bundle)' should end in .mlmodelc")
        }
    }

    func testSortformerBundleForConfig() {
        let defaultConfig = SortformerConfig.default
        let bundle = ModelNames.Sortformer.bundle(for: defaultConfig)
        XCTAssertNotNil(bundle, "Default config should match a variant")
    }

    func testSortformerRequiredModelsMatchVariants() {
        let required = ModelNames.Sortformer.requiredModels
        XCTAssertEqual(
            required.count, ModelNames.Sortformer.Variant.allCases.count,
            "Required models count should match variant count"
        )
    }

    // MARK: - Specific Model Names

    func testASRModelNamesEndInMlmodelc() {
        for model in ModelNames.ASR.requiredModels {
            XCTAssertTrue(model.hasSuffix(".mlmodelc"), "ASR model '\(model)' should end in .mlmodelc")
        }
    }

    func testVADModelNames() {
        XCTAssertEqual(ModelNames.VAD.requiredModels.count, 1)
        XCTAssertTrue(ModelNames.VAD.requiredModels.first!.hasSuffix(".mlmodelc"))
    }

    func testQwen3ASRRequiredModels() {
        XCTAssertFalse(ModelNames.Qwen3ASR.requiredModels.isEmpty)
        XCTAssertFalse(ModelNames.Qwen3ASR.requiredModelsFull.isEmpty)
    }

    // MARK: - TDT-CTC-110M Repo Tests

    func testParakeetTdtCtc110mRepoProperties() {
        let repo = Repo.parakeetTdtCtc110m

        // Verify remote path (owner/repo)
        XCTAssertEqual(repo.remotePath, "FluidInference/parakeet-tdt-ctc-110m-coreml")

        // Verify name (repo slug with -coreml suffix)
        XCTAssertEqual(repo.name, "parakeet-tdt-ctc-110m-coreml")

        // Verify folder name (simplified local folder name)
        XCTAssertEqual(repo.folderName, "parakeet-tdt-ctc-110m")

        // Should have no subpath (not a variant repo)
        XCTAssertNil(repo.subPath)
    }

    func testParakeetTdtCtc110mVocabulary() {
        // tdtCtc110m uses array-format vocabulary
        let vocabFile = ModelNames.ASR.vocabulary(for: .parakeetTdtCtc110m)
        XCTAssertEqual(vocabFile, "parakeet_vocab.json")
        XCTAssertEqual(vocabFile, ModelNames.ASR.vocabularyFileArray)
    }

    func testParakeetTdtCtc110mUsesRequiredModelsFused() {
        // tdtCtc110m has fused preprocessor+encoder, so uses requiredModelsFused
        let models = ModelNames.getRequiredModelNames(for: .parakeetTdtCtc110m, variant: nil)

        // Should match ASR.requiredModelsFused (3 .mlmodelc files, no vocab in this set)
        XCTAssertEqual(Set(models), Set(ModelNames.ASR.requiredModelsFused))

        // Should NOT match regular ASR.requiredModels (which includes separate Encoder)
        XCTAssertNotEqual(Set(models), Set(ModelNames.ASR.requiredModels))

        // Verify it includes Preprocessor (fused with encoder)
        XCTAssertTrue(models.contains("Preprocessor.mlmodelc"))

        // Verify it does NOT include separate Encoder
        XCTAssertFalse(models.contains("Encoder.mlmodelc"))
    }

    func testParakeetTdtCtc110mRequiredModelCount() {
        let models = ModelNames.getRequiredModelNames(for: .parakeetTdtCtc110m, variant: nil)

        // Fused models have 1 less file than regular (no separate Encoder)
        // Expected: Preprocessor (fused), Decoder, JointDecision = 3 .mlmodelc files
        // Note: vocabulary is handled separately, not in requiredModelsFused
        XCTAssertEqual(models.count, 3, "tdtCtc110m should have 3 .mlmodelc files (fused preprocessor+encoder)")
    }

    func testASRRequiredModelsFusedStructure() {
        let fusedModels = ModelNames.ASR.requiredModelsFused

        // Should contain core models
        XCTAssertTrue(fusedModels.contains("Preprocessor.mlmodelc"))
        XCTAssertTrue(fusedModels.contains("Decoder.mlmodelc"))
        XCTAssertTrue(fusedModels.contains("JointDecision.mlmodelc"))

        // Should NOT contain vocabulary (handled separately)
        XCTAssertFalse(fusedModels.contains("parakeet_vocab.json"))

        // Should NOT contain separate Encoder
        XCTAssertFalse(fusedModels.contains("Encoder.mlmodelc"))

        // Should be 1 less than regular models (which has 4: Preprocessor, Encoder, Decoder, Joint)
        XCTAssertEqual(fusedModels.count, ModelNames.ASR.requiredModels.count - 1)
    }
}
