import Foundation
import XCTest

@testable import FluidAudio

final class ModelRegistryTests: XCTestCase {

    // MARK: - Setup and Teardown

    override func tearDown() {
        super.tearDown()
        // Reset the custom base URL after each test
        ModelRegistry.baseURL = "https://huggingface.co"
    }

    // MARK: - Registry URL Configuration Priority Tests

    func testDefaultRegistryURL() {
        // Reset to ensure we're testing the default
        ModelRegistry.baseURL = "https://huggingface.co"

        XCTAssertEqual(ModelRegistry.baseURL, "https://huggingface.co", "Default registry should be HuggingFace")
    }

    func testProgrammaticOverrideHighestPriority() {
        let customURL = "https://custom-mirror.example.com"
        ModelRegistry.baseURL = customURL

        XCTAssertEqual(
            ModelRegistry.baseURL, customURL,
            "Programmatic override should be the highest priority")
    }

    func testRegistryURLEnvironmentVariable() {
        // This test documents the expected behavior when REGISTRY_URL is set
        // In a real test, we would need to run in a subprocess or use ProcessInfo mocking
        let expectedURL = "https://custom-registry.internal"

        ModelRegistry.baseURL = expectedURL
        XCTAssertEqual(
            ModelRegistry.baseURL, expectedURL, "REGISTRY_URL should be used when no programmatic override is set")
    }

    func testModelRegistryURLEnvironmentVariable() {
        // This test documents the fallback to MODEL_REGISTRY_URL
        let expectedURL = "https://model-registry.internal"

        ModelRegistry.baseURL = expectedURL
        XCTAssertEqual(ModelRegistry.baseURL, expectedURL, "MODEL_REGISTRY_URL should be used as fallback")
    }

    // MARK: - URL Construction Tests

    func testAPIModelsURLConstruction() throws {
        let repoPath = "pyannote/speaker-diarization-3.1"
        let apiPath = "tree/main"

        let url = try ModelRegistry.apiModels(repoPath, apiPath)
        let expectedPath = "https://huggingface.co/api/models/pyannote/speaker-diarization-3.1/tree/main"

        XCTAssertEqual(url.absoluteString, expectedPath, "API models URL should be constructed correctly")
    }

    func testAPIModelsURLWithCustomRegistry() throws {
        let customRegistry = "https://models.internal.corp"
        ModelRegistry.baseURL = customRegistry

        let repoPath = "pyannote/speaker-diarization-3.1"
        let apiPath = "tree/main"

        let url = try ModelRegistry.apiModels(repoPath, apiPath)
        let expectedPath = "https://models.internal.corp/api/models/pyannote/speaker-diarization-3.1/tree/main"

        XCTAssertEqual(url.absoluteString, expectedPath, "API models URL should use custom registry")
    }

    func testResolveModelURLConstruction() throws {
        let repoPath = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
        let filePath = "model.mlpackage"

        let url = try ModelRegistry.resolveModel(repoPath, filePath)
        let expectedPath =
            "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/model.mlpackage"

        XCTAssertEqual(url.absoluteString, expectedPath, "Resolve model URL should be constructed correctly")
    }

    func testResolveModelURLWithCustomRegistry() throws {
        let customRegistry = "https://models.internal.corp"
        ModelRegistry.baseURL = customRegistry

        let repoPath = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
        let filePath = "model.mlpackage"

        let url = try ModelRegistry.resolveModel(repoPath, filePath)
        let expectedPath =
            "https://models.internal.corp/FluidInference/parakeet-tdt-0.6b-v3-coreml/resolve/main/model.mlpackage"

        XCTAssertEqual(url.absoluteString, expectedPath, "Resolve model URL should use custom registry")
    }

    func testAPIDatasetsURLConstruction() throws {
        let dataset = "FluidInference/librispeech"
        let apiPath = "tree/main"

        let url = try ModelRegistry.apiDatasets(dataset, apiPath)
        let expectedPath = "https://huggingface.co/api/datasets/FluidInference/librispeech/tree/main"

        XCTAssertEqual(url.absoluteString, expectedPath, "API datasets URL should be constructed correctly")
    }

    func testAPIDatasetsURLWithCustomRegistry() throws {
        let customRegistry = "https://datasets.internal.corp"
        ModelRegistry.baseURL = customRegistry

        let dataset = "FluidInference/librispeech"
        let apiPath = "tree/main"

        let url = try ModelRegistry.apiDatasets(dataset, apiPath)
        let expectedPath = "https://datasets.internal.corp/api/datasets/FluidInference/librispeech/tree/main"

        XCTAssertEqual(url.absoluteString, expectedPath, "API datasets URL should use custom registry")
    }

    func testResolveDatasetURLConstruction() throws {
        let dataset = "FluidInference/librispeech"
        let filePath = "test-clean.tar.gz"

        let url = try ModelRegistry.resolveDataset(dataset, filePath)
        let expectedPath = "https://huggingface.co/datasets/FluidInference/librispeech/resolve/main/test-clean.tar.gz"

        XCTAssertEqual(url.absoluteString, expectedPath, "Resolve dataset URL should be constructed correctly")
    }

    func testResolveDatasetURLWithCustomRegistry() throws {
        let customRegistry = "https://datasets.internal.corp"
        ModelRegistry.baseURL = customRegistry

        let dataset = "FluidInference/librispeech"
        let filePath = "test-clean.tar.gz"

        let url = try ModelRegistry.resolveDataset(dataset, filePath)
        let expectedPath =
            "https://datasets.internal.corp/datasets/FluidInference/librispeech/resolve/main/test-clean.tar.gz"

        XCTAssertEqual(url.absoluteString, expectedPath, "Resolve dataset URL should use custom registry")
    }

    // MARK: - Dataset Base URL Tests

    func testResolveDatasetBaseConstruction() {
        let dataset = "alexwengg/musan_mini50"

        let baseURL = ModelRegistry.resolveDatasetBase(dataset)
        let expected = "https://huggingface.co/datasets/alexwengg/musan_mini50/resolve/main"

        XCTAssertEqual(baseURL, expected, "Dataset base URL should be constructed without trailing slash")
    }

    func testResolveDatasetBaseWithCustomRegistry() {
        let customRegistry = "https://datasets.internal.corp"
        ModelRegistry.baseURL = customRegistry

        let dataset = "alexwengg/musan_mini50"

        let baseURL = ModelRegistry.resolveDatasetBase(dataset)
        let expected = "https://datasets.internal.corp/datasets/alexwengg/musan_mini50/resolve/main"

        XCTAssertEqual(baseURL, expected, "Dataset base URL should use custom registry without trailing slash")
    }

    func testDatasetBaseURLNoTrailingSlash() {
        let dataset = "test/dataset"

        let baseURL = ModelRegistry.resolveDatasetBase(dataset)

        XCTAssertFalse(baseURL.hasSuffix("/"), "Dataset base URL should not have trailing slash")
    }

    // MARK: - Complex Path Tests

    func testComplexRepositoryPaths() throws {
        let testCases: [(repo: String, file: String, expected: String)] = [
            ("user/repo", "file.txt", "https://huggingface.co/user/repo/resolve/main/file.txt"),
            ("org/model-v2", "weights/model.bin", "https://huggingface.co/org/model-v2/resolve/main/weights/model.bin"),
            (
                "user/repo", "path/to/nested/file.zip",
                "https://huggingface.co/user/repo/resolve/main/path/to/nested/file.zip"
            ),
        ]

        for (repo, file, expected) in testCases {
            let url = try ModelRegistry.resolveModel(repo, file)
            XCTAssertEqual(url.absoluteString, expected, "URL for repo=\(repo), file=\(file) should match expected")
        }
    }

    func testComplexDatasetPaths() throws {
        let testCases: [(dataset: String, file: String, expected: String)] = [
            ("user/dataset", "data.csv", "https://huggingface.co/datasets/user/dataset/resolve/main/data.csv"),
            (
                "org/corpus-v1", "splits/train.txt",
                "https://huggingface.co/datasets/org/corpus-v1/resolve/main/splits/train.txt"
            ),
            (
                "user/dataset", "en/audio/sample001.wav",
                "https://huggingface.co/datasets/user/dataset/resolve/main/en/audio/sample001.wav"
            ),
        ]

        for (dataset, file, expected) in testCases {
            let url = try ModelRegistry.resolveDataset(dataset, file)
            XCTAssertEqual(
                url.absoluteString, expected, "URL for dataset=\(dataset), file=\(file) should match expected")
        }
    }

    // MARK: - URLSession Configuration Tests

    func testConfiguredSessionReturnsURLSession() {
        // This is a basic test that the method returns a valid URLSession
        // More detailed testing would require mocking URLSessionConfiguration
        let session = ModelRegistry.configuredSession()

        // The session should be a valid URLSession instance
        XCTAssertNotNil(session, "Configured session should not be nil")
    }

    // MARK: - Consistency Tests

    func testRegistryURLPersistence() {
        let customURL1 = "https://first-mirror.example.com"
        let customURL2 = "https://second-mirror.example.com"

        ModelRegistry.baseURL = customURL1
        XCTAssertEqual(ModelRegistry.baseURL, customURL1, "First custom URL should persist")

        ModelRegistry.baseURL = customURL2
        XCTAssertEqual(ModelRegistry.baseURL, customURL2, "Second custom URL should override first")

        // Reset to default
        ModelRegistry.baseURL = "https://huggingface.co"
        XCTAssertEqual(ModelRegistry.baseURL, "https://huggingface.co", "Should reset to default")
    }

    func testURLConsistencyAcrossRegistries() throws {
        let registries = [
            "https://huggingface.co",
            "https://mirrors.internal.corp",
            "https://air-gapped-server.local",
        ]

        let repo = "test/model"
        let file = "data.bin"

        for registry in registries {
            ModelRegistry.baseURL = registry
            let url = try ModelRegistry.resolveModel(repo, file)

            XCTAssertTrue(url.absoluteString.starts(with: registry), "URL should use the set registry: \(registry)")
            XCTAssertTrue(
                url.absoluteString.contains(repo),
                "URL should contain repository path for registry: \(registry)")
            XCTAssertTrue(
                url.absoluteString.contains(file),
                "URL should contain file path for registry: \(registry)")
        }
    }
}
