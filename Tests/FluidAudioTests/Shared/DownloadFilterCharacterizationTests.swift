import Foundation
import XCTest

@testable import FluidAudio

/// Characterization tests for `ModelHub.download`'s file-selection rules (#765
/// Wave 1). These pin CURRENT behavior — quirks included — so the Wave 3
/// lister extraction can prove the rules moved verbatim. They are not a
/// statement of what the rules *should* be; deliberate changes belong in a
/// later wave with these fixtures edited in the same diff.
///
/// Covered history: the subPath prefix rules, the differing metadata-extension
/// allowances at root (`.json`/`.txt`) vs under a subPath
/// (`.json`/`.model`/`.bin`), the #649 root-level auxiliary fallback, and the
/// #524 `additionalModelNames` union.
final class DownloadFilterCharacterizationTests: XCTestCase {

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("FilterCharacterization-\(UUID().uuidString)", isDirectory: true)
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

        // Pinned: required-model subtree + root .json/.txt come down;
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
                // Pinned: under a subPath the metadata allowance is .json/.model/.bin
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

        // Pinned: subPath prefix is stripped locally; .json/.model/.bin under the
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

    // MARK: - Root fallback matches full names only

    /// The #649 root fallback pulls a root file only when its name equals a
    /// missing required aux file's FULL name. A slash-containing required
    /// path (KokoroAne-zh's `voices/zf_001.bin`) must never match a
    /// same-basename root file — that would download wrong-variant content
    /// to the wrong local path instead of failing loudly with modelNotFound.
    func testRootFallbackNeverMatchesSlashContainingAuxByBasename() async throws {
        let sub = "ANE-zh"
        let voice = ModelNames.KokoroAne.defaultVoiceFileZh  // "voices/zf_001.bin"
        XCTAssertTrue(voice.contains("/"), "fixture requires a slash-containing aux model")
        let decoyName = (voice as NSString).lastPathComponent

        // Serve every required model EXCEPT the voice; plant a same-basename
        // decoy at the repo root.
        var subItems: [[String: Any]] = [
            ["path": "\(sub)/\(ModelNames.KokoroAne.vocab)", "type": "file", "size": 10],
            ["path": "\(sub)/g2pw", "type": "directory"],
        ]
        var trees: [String: [[String: Any]]] = [:]
        for model in ModelNames.KokoroAne.requiredCoreMLModels {
            subItems.append(["path": "\(sub)/\(model)", "type": "directory"])
            trees["\(sub)/\(model)"] = [
                ["path": "\(sub)/\(model)/coremldata.bin", "type": "file", "size": 10]
            ]
        }
        let g2pw = ModelNames.KokoroAne.g2pwModelZh  // "g2pw/g2pw.mlmodelc"
        trees["\(sub)/g2pw"] = [["path": "\(sub)/\(g2pw)", "type": "directory"]]
        trees["\(sub)/\(g2pw)"] = [
            ["path": "\(sub)/\(g2pw)/coremldata.bin", "type": "file", "size": 10]
        ]
        trees[sub] = subItems
        trees[""] = [
            ["path": decoyName, "type": "file", "size": 10],
            ["path": sub, "type": "directory"],
        ]
        TreeStubURLProtocol.trees = trees
        TreeStubURLProtocol.fileBody = body(10)

        do {
            try await ModelHub.download(
                .kokoroAneZh, to: workDir, configuration: stubConfiguration)
            XCTFail("expected modelNotFound for the missing voice")
        } catch DownloadError.modelNotFound(let path) {
            XCTAssertEqual(path, voice)
        }

        // Pinned: the decoy must NOT have been downloaded to the repo root.
        let repoPath = workDir.appendingPathComponent(Repo.kokoroAneZh.folderName)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: repoPath.appendingPathComponent(decoyName).path),
            "a same-basename root file must not satisfy a slash-containing required path")
    }

    // MARK: - additionalModelNames (#524)

    func testAdditionalModelNamesAreUnionedIntoSelection() async throws {
        // .parakeetCtc110m requires MelSpectrogram + AudioEncoder; the TDT-CTC
        // manager additionally requests CtcHead.mlmodelc (#524).
        TreeStubURLProtocol.trees = [
            "": [
                ["path": "MelSpectrogram.mlmodelc", "type": "directory"],
                ["path": "AudioEncoder.mlmodelc", "type": "directory"],
                ["path": "CtcHead.mlmodelc", "type": "directory"],
                ["path": "OtherHead.mlmodelc", "type": "directory"],
            ],
            "MelSpectrogram.mlmodelc": [
                ["path": "MelSpectrogram.mlmodelc/coremldata.bin", "type": "file", "size": 10]
            ],
            "AudioEncoder.mlmodelc": [
                ["path": "AudioEncoder.mlmodelc/coremldata.bin", "type": "file", "size": 10]
            ],
            "CtcHead.mlmodelc": [
                ["path": "CtcHead.mlmodelc/coremldata.bin", "type": "file", "size": 10]
            ],
            "OtherHead.mlmodelc": [
                ["path": "OtherHead.mlmodelc/coremldata.bin", "type": "file", "size": 10]
            ],
        ]
        TreeStubURLProtocol.fileBody = body(10)

        try await ModelHub.download(
            .parakeetCtc110m, to: workDir,
            additionalModelNames: ["CtcHead.mlmodelc"],
            configuration: stubConfiguration)

        // Pinned: the extra model is selected; non-required siblings are not.
        XCTAssertEqual(
            try downloadedFiles(repoFolder: Repo.parakeetCtc110m.folderName),
            [
                "AudioEncoder.mlmodelc/coremldata.bin",
                "CtcHead.mlmodelc/coremldata.bin",
                "MelSpectrogram.mlmodelc/coremldata.bin",
            ].sorted()
        )
    }
}

// MARK: - Tree-serving URLProtocol stub

/// Serves canned HF `tree/main` JSON per path and a fixed body for every
/// `resolve/main` file request. Thread-safe via a lock; keyed on URL shape.
final class TreeStubURLProtocol: URLProtocol {

    private static let lock = NSLock()
    nonisolated(unsafe) private static var _trees: [String: [[String: Any]]] = [:]
    nonisolated(unsafe) private static var _fileBody = Data()

    static var trees: [String: [[String: Any]]] {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _trees
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _trees = newValue
        }
    }

    static var fileBody: Data {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _fileBody
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _fileBody = newValue
        }
    }

    static func reset() {
        trees = [:]
        fileBody = Data()
    }

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        let path = url.path

        let payload: Data
        if let treeRange = path.range(of: "/tree/main") {
            // Listing request: key is everything after "tree/main/" ("" for root).
            var key = String(path[treeRange.upperBound...])
            if key.hasPrefix("/") { key.removeFirst() }
            guard let items = Self.trees[key],
                let json = try? JSONSerialization.data(withJSONObject: items)
            else {
                respond(status: 404, data: Data("[]".utf8))
                return
            }
            payload = json
        } else if path.contains("/resolve/main/") {
            payload = Self.fileBody
        } else {
            respond(status: 404, data: Data())
            return
        }
        respond(status: 200, data: payload)
    }

    override func stopLoading() {}

    private func respond(status: Int, data: Data) {
        let response = HTTPURLResponse(
            url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Length": String(data.count)])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
}
