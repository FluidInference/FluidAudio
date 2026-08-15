import Foundation
import XCTest

@testable import FluidAudio

/// Coverage for `ModelHub.download`'s file-selection `include` rules: which
/// remote files a repo download actually pulls down. Two representative shapes:
/// a plain repo (required-model dirs + root `.json`/`.txt` allowances) and a
/// subPath repo (prefix stripping, the subPath `.json`/`.model`/`.bin`
/// allowance, and the #649 root-level auxiliary fallback). Driven through the
/// `configuration:` seam with `TreeStubURLProtocol`.
final class ModelHubFileSelectionTests: XCTestCase {

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelHubFileSelection-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        TreeStubURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        TreeStubURLProtocol.reset()
        try? FileManager.default.removeItem(at: workDir)
    }

    private var stubConfiguration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [TreeStubURLProtocol.self]
        return config
    }

    /// Every downloaded file relative to the repo directory, sorted.
    /// Symlinks are resolved on both sides: on macOS `temporaryDirectory` is
    /// `/var/...` while the enumerator returns `/private/var/...`.
    private func downloadedFiles(repoFolder: String) throws -> [String] {
        let repoPath = workDir.appendingPathComponent(repoFolder).resolvingSymlinksInPath()
        let basePrefix = repoPath.path + "/"
        guard
            let enumerator = FileManager.default.enumerator(
                at: repoPath, includingPropertiesForKeys: [.isRegularFileKey])
        else { return [] }
        var files: [String] = []
        for case let url as URL in enumerator {
            if (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
                let path = url.resolvingSymlinksInPath().path
                XCTAssertTrue(path.hasPrefix(basePrefix), "unexpected path \(path)")
                files.append(String(path.dropFirst(basePrefix.count)))
            }
        }
        return files.sorted()
    }

    private func body(_ size: Int) -> Data {
        Data(String(repeating: "x", count: size).utf8)
    }

    // MARK: - Plain repo (patterns + root metadata-extension allowances)

    func testPlainRepoSelectsRequiredModelDirsAndRootJsonTxt() async throws {
        // .vad requires exactly ModelNames.VAD.sileroVadFile
        let model = ModelNames.VAD.sileroVadFile
        TreeStubURLProtocol.trees = [
            "": [
                ["path": model, "type": "directory"],
                ["path": "config.json", "type": "file", "size": 10],
                ["path": "NOTES.txt", "type": "file", "size": 10],
                ["path": "README.md", "type": "file", "size": 10],
                ["path": "stray.bin", "type": "file", "size": 10],
                ["path": "unrelated.mlmodelc", "type": "directory"],
            ],
            model: [
                ["path": "\(model)/coremldata.bin", "type": "file", "size": 10],
                ["path": "\(model)/weights/weight.bin", "type": "file", "size": 10],
            ],
            "unrelated.mlmodelc": [
                ["path": "unrelated.mlmodelc/coremldata.bin", "type": "file", "size": 10]
            ],
        ]
        TreeStubURLProtocol.fileBody = body(10)

        try await ModelHub.download(
            .vad, to: workDir, configuration: stubConfiguration)

        // required-model subtree + root .json/.txt come down;
        // README.md, stray .bin at root, and non-required model dirs do not.
        XCTAssertEqual(
            try downloadedFiles(repoFolder: Repo.vad.folderName),
            [
                "NOTES.txt",
                "config.json",
                "\(model)/coremldata.bin",
                "\(model)/weights/weight.bin",
            ].sorted()
        )
    }

    // MARK: - subPath repo (prefix stripping, subPath metadata rules, #649 fallback)

    func testSubPathRepoStripsPrefixAppliesMetadataRulesAndFallsBackToRootAux() async throws {
        // .parakeetEou160: subPath "160ms", requires 3 .mlmodelc dirs + vocab.json.
        // vocab.json lives at the repo ROOT (the #649 shape), not under 160ms/.
        let sub = "160ms"
        let encoder = ModelNames.ParakeetEOU.encoderFile
        let decoder = ModelNames.ParakeetEOU.decoderFile
        let joint = ModelNames.ParakeetEOU.jointFile
        let vocab = ModelNames.ParakeetEOU.vocab
        TreeStubURLProtocol.trees = [
            sub: [
                ["path": "\(sub)/\(encoder)", "type": "directory"],
                ["path": "\(sub)/\(decoder)", "type": "directory"],
                ["path": "\(sub)/\(joint)", "type": "directory"],
                // Under a subPath the metadata allowance is .json/.model/.bin
                ["path": "\(sub)/config.json", "type": "file", "size": 10],
                ["path": "\(sub)/tokenizer.model", "type": "file", "size": 10],
                ["path": "\(sub)/stats.bin", "type": "file", "size": 10],
                ["path": "\(sub)/README.txt", "type": "file", "size": 10],
            ],
            "\(sub)/\(encoder)": [
                ["path": "\(sub)/\(encoder)/coremldata.bin", "type": "file", "size": 10]
            ],
            "\(sub)/\(decoder)": [
                ["path": "\(sub)/\(decoder)/coremldata.bin", "type": "file", "size": 10]
            ],
            "\(sub)/\(joint)": [
                ["path": "\(sub)/\(joint)/coremldata.bin", "type": "file", "size": 10]
            ],
            // Root listing used by the #649 fallback for required non-bundle files.
            "": [
                ["path": vocab, "type": "file", "size": 10],
                ["path": "160ms", "type": "directory"],
                ["path": "320ms", "type": "directory"],
            ],
        ]
        TreeStubURLProtocol.fileBody = body(10)

        try await ModelHub.download(
            .parakeetEou160, to: workDir, configuration: stubConfiguration)

        // subPath prefix is stripped locally; .json/.model/.bin under the
        // subPath come down, .txt under the subPath does NOT (root-vs-subPath
        // asymmetry); root vocab.json arrives via the #649 fallback.
        XCTAssertEqual(
            try downloadedFiles(repoFolder: Repo.parakeetEou160.folderName),
            [
                "config.json",
                "\(decoder)/coremldata.bin",
                "\(joint)/coremldata.bin",
                "stats.bin",
                "\(encoder)/coremldata.bin",
                "tokenizer.model",
                vocab,
            ].sorted()
        )
    }
}
