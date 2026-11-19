import Foundation
import XCTest

@testable import FluidAudio

/// Vocabulary boosting reads `tokenizer.json` from the folder the CTC models were loaded from, not from the
/// default Application Support folder (#1114).
final class CtcModelsDirectoryTests: XCTestCase {

    func testTokenizerDirectoryIsTheLoadedDirectoryElseTheDefault() {
        let custom = URL(fileURLWithPath: "/Models/parakeet-ctc-110m-coreml", isDirectory: true)
        for variant in CtcModelVariant.allCases {
            XCTAssertEqual(CtcModels.tokenizerDirectory(loadedFrom: custom, variant: variant), custom)
            XCTAssertEqual(
                CtcModels.tokenizerDirectory(loadedFrom: nil, variant: variant),
                CtcModels.defaultCacheDirectory(for: variant))
        }
    }

    /// CTC 110M from the cache, or downloaded when `FLUIDAUDIO_RUN_ASR_E2E=1` (the CI E2E step), copied into a
    /// temporary folder.
    func testSessionReadsTokenizerFromTheFolderTheModelsLoadedFrom() async throws {
        let source = CtcModels.defaultCacheDirectory(for: .ctc110m)
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(
            CtcModels.modelsExist(at: source) || allowDownload,
            "CTC 110M not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")
        _ = try await CtcModels.download(to: source)

        // The models and vocab.json in a folder named like the repo, without the tokenizer at first, and a
        // differently named folder holding only vocab.json.
        let fileManager = FileManager.default
        let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let modelFolder = root.appendingPathComponent(CtcModelVariant.ctc110m.repo.folderName, isDirectory: true)
        let renamedFolder = root.appendingPathComponent("my-ctc-models", isDirectory: true)
        try fileManager.createDirectory(at: modelFolder, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: renamedFolder, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        for name in ModelNames.CTC.requiredModels.sorted() + [ModelNames.CTC.vocabularyPath] {
            try fileManager.copyItem(
                at: source.appendingPathComponent(name), to: modelFolder.appendingPathComponent(name))
        }
        try fileManager.copyItem(
            at: source.appendingPathComponent(ModelNames.CTC.vocabularyPath),
            to: renamedFolder.appendingPathComponent(ModelNames.CTC.vocabularyPath))

        // `load(from:)` loads the models from `<parent>/<repo folder>` whatever the given folder is called.
        let direct = try await CtcModels.loadDirect(from: modelFolder)
        let viaHub = try await CtcModels.load(from: modelFolder)
        let viaHubRenamed = try await CtcModels.load(from: renamedFolder)
        let loaded = [
            ("loadDirect(from:)", direct), ("load(from:)", viaHub), ("load(from:) renamed folder", viaHubRenamed),
        ]
        for (label, models) in loaded {
            XCTAssertEqual(models.directory?.path, modelFolder.path, label)
        }
        let unrecorded = CtcModels(
            melSpectrogram: direct.melSpectrogram, encoder: direct.encoder, configuration: direct.configuration,
            vocabulary: direct.vocabulary, variant: direct.variant)
        XCTAssertNil(unrecorded.directory)

        let tokenizer = try await CtcTokenizer.load(from: source)
        let term = "Nyiragongo"
        let untokenized = CustomVocabularyContext(terms: [CustomVocabularyTerm(text: term)])
        let pretokenized = CustomVocabularyContext(terms: [
            CustomVocabularyTerm(text: term, ctcTokenIds: tokenizer.encode(term))
        ])
        for (label, models) in loaded {
            for vocabulary in [untokenized, pretokenized] {
                do {
                    _ = try await VocabularyBoostingSession(vocabulary: vocabulary, ctcModels: models)
                    XCTFail("\(label): session found a tokenizer although the models' folder has none")
                } catch CtcTokenizer.Error.tokenizerNotFound(let directory) {
                    XCTAssertEqual(directory.path, modelFolder.path, label)
                }
            }
        }

        try fileManager.copyItem(
            at: source.appendingPathComponent("tokenizer.json"),
            to: modelFolder.appendingPathComponent("tokenizer.json"))
        for (label, models) in loaded {
            let session = try await VocabularyBoostingSession(vocabulary: untokenized, ctcModels: models)
            XCTAssertEqual(session.vocabulary.terms.first?.ctcTokenIds, tokenizer.encode(term), label)
        }
    }
}
