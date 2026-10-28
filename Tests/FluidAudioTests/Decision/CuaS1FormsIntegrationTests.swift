import CryptoKit
import Foundation
import XCTest

@testable import FluidAudio

/// Opt-in checks against the actual converted checkpoint and the pinned upstream demo.
/// Fixture setup is in the Mobius CUA-S1-FORMS toolkit under "Swift integration checks".
@MainActor
final class CuaS1FormsIntegrationTests: XCTestCase {
    private struct Decision: Decodable, Sendable {
        let context: String
        let options: [String]
        let label: Int
    }

    private struct Reference: Decodable {
        let modelRevision: String
        let datasetSHA256: String
        let probabilities: [[Float]]

        private enum CodingKeys: String, CodingKey {
            case modelRevision = "model_revision"
            case datasetSHA256 = "dataset_sha256"
            case probabilities
        }
    }

    private func fixtureURL(_ name: String) throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[name], !path.isEmpty else {
            throw XCTSkip("Set \(name) to enable real CUA-S1-FORMS integration tests")
        }
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: path])
        }
        return url
    }

    private func demo() throws -> [Decision] {
        let data = try Data(contentsOf: fixtureURL("FLUIDAUDIO_CUA_DEMO_PATH"))
        return try data.split(separator: 10).map { try JSONDecoder().decode(Decision.self, from: Data($0)) }
    }

    func testAll196DecisionsMatchPyTorch() async throws {
        let modelURL = try fixtureURL("FLUIDAUDIO_CUA_MODEL_PATH")
        let demoURL = try fixtureURL("FLUIDAUDIO_CUA_DEMO_PATH")
        let referenceURL = try fixtureURL("FLUIDAUDIO_CUA_REFERENCE_PATH")
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: referenceURL))
        let hash = SHA256.hash(data: try Data(contentsOf: demoURL)).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(hash, reference.datasetSHA256)
        XCTAssertEqual(hash, "4f43b442e79ba2e2ce731e27e9b8e340c2b5dfcaffc92d8ff564c34f115ff1ca")
        XCTAssertEqual(reference.modelRevision, "f54adbf447f4ca6ec259f529ee3f2e3e09f8cc71")
        let rows = try demo()
        XCTAssertEqual(rows.count, 196)
        XCTAssertEqual(reference.probabilities.count, rows.count)
        guard rows.count == 196, reference.probabilities.count == rows.count else { return }

        let manager = try await CuaS1FormsManager.load(from: modelURL)
        var maximumError: Float = 0
        for (index, row) in rows.enumerated() {
            let result = try await manager.score(context: row.context, options: row.options)
            let expected = reference.probabilities[index]
            XCTAssertEqual(result.selectedIndex, row.label, "Demo row \(index)")
            XCTAssertEqual(result.selectedOption, row.options[row.label])
            XCTAssertEqual(result.probabilities.count, expected.count)
            XCTAssertEqual(result.rawProbabilities.count, expected.count)
            XCTAssertEqual(expected.count, row.options.count)
            let referenceIndex = expected.indices.max { expected[$0] < expected[$1] }
            XCTAssertEqual(result.selectedIndex, referenceIndex)
            for (actual, target) in zip(result.probabilities, expected) {
                maximumError = max(maximumError, abs(actual - target))
                XCTAssertEqual(actual, target, accuracy: 0.005, "Demo row \(index)")
            }
            for (raw, target) in zip(result.rawProbabilities, expected) {
                XCTAssertEqual(raw, target, accuracy: 0.005, "Raw model probability, demo row \(index)")
            }
            XCTAssertEqual(result.probabilities.reduce(0, +), 1, accuracy: 0.001)
            XCTAssertTrue(result.logits.allSatisfy(\.isFinite))
            XCTAssertFalse(result.contextWasTruncated)
            XCTAssertTrue(result.truncatedOptionIndices.isEmpty)
        }
        print("CUA-S1-FORMS Swift: 196 rows; maximum probability error vs PyTorch: \(maximumError)")
    }

    func testReorderedOptionsAndConcurrentCalls() async throws {
        let modelURL = try fixtureURL("FLUIDAUDIO_CUA_MODEL_PATH")
        let rows = try demo()
        let manager = try await CuaS1FormsManager.load(from: modelURL)
        let selectedRows = [0, 31, 63, 95, 127, 195]
        guard rows.count == 196 else { return XCTFail("Expected the pinned 196-row demo") }
        let matches = try await withThrowingTaskGroup(of: Bool.self) { group in
            for index in selectedRows {
                let row = rows[index]
                group.addTask {
                    let original = try await manager.score(context: row.context, options: row.options)
                    let reversed = try await manager.score(context: row.context, options: row.options.reversed())
                    return original.selectedIndex == row.label
                        && original.selectedOption == reversed.selectedOption
                        && zip(original.probabilities, reversed.probabilities.reversed()).allSatisfy {
                            abs($0 - $1) <= 0.005
                        }
                }
            }
            var results: [Bool] = []
            for try await value in group { results.append(value) }
            return results
        }
        XCTAssertEqual(matches.count, selectedRows.count)
        XCTAssertTrue(matches.allSatisfy { $0 })
    }

    func testSharedCacheLoadsRealCompiledModel() async throws {
        let compiled = try fixtureURL("FLUIDAUDIO_CUA_COMPILED_PATH")
        let row = try XCTUnwrap(demo().first)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = root.appendingPathComponent(Repo.cuaS1Forms.folderName)
        try FileManager.default.createDirectory(at: repository, withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: compiled, to: repository.appendingPathComponent(ModelNames.CuaS1Forms.modelFile))
        let manager = try await CuaS1FormsManager.load(cacheDirectory: root)
        let result = try await manager.score(context: row.context, options: row.options)
        XCTAssertEqual(result.selectedIndex, row.label)
    }

    func testLoadsRealPackageThroughCacheSymlinks() async throws {
        let source = try fixtureURL("FLUIDAUDIO_CUA_MODEL_PATH")
        guard source.pathExtension == "mlpackage" else {
            throw XCTSkip("Set FLUIDAUDIO_CUA_MODEL_PATH to the real portable package for the symlink check")
        }
        let row = try XCTUnwrap(demo().first)
        let files = FileManager.default
        let root = files.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? files.removeItem(at: root) }
        let linked = root.appendingPathComponent("linked.mlpackage")
        try files.createDirectory(at: linked, withIntermediateDirectories: true)
        for name in ["Manifest.json", "Data"] {
            try files.createSymbolicLink(
                at: linked.appendingPathComponent(name), withDestinationURL: source.appendingPathComponent(name))
        }
        let prepared = try CuaS1FormsPackage.prepare(linked)
        defer { prepared.cleanup() }
        let weightPath = "Data/com.apple.CoreML/weights/weight.bin"
        let weight = prepared.url.appendingPathComponent(weightPath)
        XCTAssertFalse(try weight.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink ?? true)
        XCTAssertEqual(
            try Data(contentsOf: weight), try Data(contentsOf: source.appendingPathComponent(weightPath)))

        let manager = try await CuaS1FormsManager.load(from: linked)
        let result = try await manager.score(context: row.context, options: row.options)
        XCTAssertEqual(result.selectedIndex, row.label)
        prepared.cleanup()
        XCTAssertFalse(files.fileExists(atPath: prepared.url.path))
        XCTAssertTrue(files.fileExists(atPath: source.path))
    }
}
