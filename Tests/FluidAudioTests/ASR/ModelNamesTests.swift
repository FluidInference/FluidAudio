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
}
