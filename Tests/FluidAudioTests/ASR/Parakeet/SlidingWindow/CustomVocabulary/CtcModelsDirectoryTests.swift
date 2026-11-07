import Foundation
import XCTest

@testable import FluidAudio

/// Vocabulary boosting reads `tokenizer.json` from the directory the CTC models were loaded from, not from the
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

    /// Needs the CTC 110M models; runs when they are cached or `FLUIDAUDIO_RUN_ASR_E2E=1` allows a download.
    func testSessionReadsTokenizerNextToModelsLoadedFromCustomDirectory() async throws {
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(
            CtcModels.modelsExist(at: CtcModels.defaultCacheDirectory(for: .ctc110m)) || allowDownload,
            "CTC 110M models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")

        let source = try await CtcModels.download()
        let fromDefault = try await CtcModels.load(from: source)
        XCTAssertEqual(fromDefault.directory, source)
        let unrecorded = CtcModels(
            melSpectrogram: fromDefault.melSpectrogram, encoder: fromDefault.encoder,
            configuration: fromDefault.configuration, vocabulary: fromDefault.vocabulary,
            variant: fromDefault.variant)
        XCTAssertNil(unrecorded.directory)

        // The models and vocab.json in a custom folder, without the tokenizer at first.
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let custom = root.appendingPathComponent(CtcModelVariant.ctc110m.repo.folderName, isDirectory: true)
        try FileManager.default.createDirectory(at: custom, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        for name in ModelNames.CTC.requiredModels.sorted() + [ModelNames.CTC.vocabularyPath] {
            try FileManager.default.copyItem(
                at: source.appendingPathComponent(name), to: custom.appendingPathComponent(name))
        }
        let models = try await CtcModels.loadDirect(from: custom)
        XCTAssertEqual(models.directory, custom)

        let tokenizer = try await CtcTokenizer.load(from: source)
        let term = "Nyiragongo"
        let untokenized = CustomVocabularyContext(terms: [CustomVocabularyTerm(text: term)])
        let pretokenized = CustomVocabularyContext(terms: [
            CustomVocabularyTerm(text: term, ctcTokenIds: tokenizer.encode(term))
        ])
        for vocabulary in [untokenized, pretokenized] {
            do {
                _ = try await VocabularyBoostingSession(vocabulary: vocabulary, ctcModels: models)
                XCTFail("Session found a tokenizer although the models' directory has none")
            } catch CtcTokenizer.Error.tokenizerNotFound(let directory) {
                XCTAssertEqual(directory.standardizedFileURL.path, custom.standardizedFileURL.path)
            }
        }

        try FileManager.default.copyItem(
            at: source.appendingPathComponent("tokenizer.json"), to: custom.appendingPathComponent("tokenizer.json"))
        let session = try await VocabularyBoostingSession(vocabulary: untokenized, ctcModels: models)
        XCTAssertEqual(session.vocabulary.terms.first?.ctcTokenIds, tokenizer.encode(term))
    }
}
