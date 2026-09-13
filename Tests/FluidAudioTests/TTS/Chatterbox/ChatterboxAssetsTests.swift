import Foundation
import XCTest

@testable import FluidAudio

/// Aux-asset robustness: the drop-and-refetch recovery must honor
/// `ModelHub.offlineMode` (no purge, no network — rethrow the parse error),
/// and malformed safetensors headers must throw instead of trapping.
final class ChatterboxAssetsTests: XCTestCase {

    private var tmpDir: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ChatterboxAssetsTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        ModelHub.offlineMode = false
        if let tmpDir, FileManager.default.fileExists(atPath: tmpDir.path) {
            try FileManager.default.removeItem(at: tmpDir)
        }
        try super.tearDownWithError()
    }

    private static let logger = AppLogger(category: "ChatterboxAssetsTests")

    // MARK: - loadAuxWithRecovery offline contract

    func testOfflineModePreservesCacheAndRethrowsWithoutRefetch() async throws {
        ModelHub.offlineMode = true
        let auxFile = "tables/tables.safetensors"
        let auxURL = tmpDir.appendingPathComponent(auxFile)
        try FileManager.default.createDirectory(
            at: auxURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("not a safetensors file".utf8).write(to: auxURL)

        var refetched = false
        do {
            let _: Int = try await ChatterboxMLSupport.loadAuxWithRecovery(
                repoDir: tmpDir,
                auxFiles: [auxFile],
                logger: Self.logger,
                refetch: { refetched = true },
                load: { throw ChatterboxError.malformedAsset("corrupt fixture") })
            XCTFail("expected malformedAsset to be rethrown")
        } catch let ChatterboxError.malformedAsset(detail) {
            XCTAssertEqual(detail, "corrupt fixture")
        }

        XCTAssertFalse(refetched, "offline mode must not attempt a re-fetch")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: auxURL.path),
            "offline mode must not purge cached aux files")
    }

    func testOnlineRecoveryDeletesAuxAndRetriesOnce() async throws {
        ModelHub.offlineMode = false
        let auxFile = "tables/tables.safetensors"
        let auxURL = tmpDir.appendingPathComponent(auxFile)
        try FileManager.default.createDirectory(
            at: auxURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("corrupt".utf8).write(to: auxURL)

        var refetched = false
        var loadCalls = 0
        let value: Int = try await ChatterboxMLSupport.loadAuxWithRecovery(
            repoDir: tmpDir,
            auxFiles: [auxFile],
            logger: Self.logger,
            refetch: { refetched = true },
            load: {
                loadCalls += 1
                if loadCalls == 1 {
                    // The corrupt file must be gone before refetch runs.
                    throw ChatterboxError.malformedAsset("first attempt")
                }
                XCTAssertFalse(FileManager.default.fileExists(atPath: auxURL.path))
                return 7
            })

        XCTAssertEqual(value, 7)
        XCTAssertTrue(refetched)
        XCTAssertEqual(loadCalls, 2)
    }

    // MARK: - Dimension validation (unsafe-copy bounds)

    /// `validate` reads only rows/cols, so empty `values` keep fixtures
    /// lightweight.
    private func table(_ rows: Int, _ cols: Int) -> ChatterboxTables.Table {
        ChatterboxTables.Table(rows: rows, cols: cols, values: [])
    }

    private func mtlTables(
        textPosRows: Int = ChatterboxConstants.prefillLength,
        speechPosRows: Int = ChatterboxConstants.maxContext
    ) -> ChatterboxTables {
        let hidden = ChatterboxConstants.hiddenSize
        return ChatterboxTables(
            textEmb: table(ChatterboxConstants.textVocabSize, hidden),
            speechEmb: table(ChatterboxConstants.outputVocabSize, hidden),
            textPos: table(textPosRows, hidden),
            speechPos: table(speechPosRows, hidden))
    }

    private func voice(
        hidden: Int, promptTokens: Int, promptFeatRows: Int
    )
        -> ChatterboxTables.Voice
    {
        ChatterboxTables.Voice(
            condEmb: table(34, hidden),
            promptTokens: [Int32](repeating: 0, count: promptTokens),
            promptFeat: table(promptFeatRows, 80),
            embedding: [Float](repeating: 0, count: 192))
    }

    func testValidDimensionsPassBothVariants() throws {
        try ChatterboxTables.validate(
            mtlTables(),
            voice: voice(
                hidden: ChatterboxConstants.hiddenSize, promptTokens: 157,
                promptFeatRows: 314))
        let nanoHidden = ChatterboxNanoConstants.hiddenSize
        try ChatterboxTables.validate(
            ChatterboxTables.Nano(
                textEmb: table(ChatterboxNanoConstants.textVocabSize, nanoHidden),
                speechEmb: table(ChatterboxNanoConstants.outputVocabSize, nanoHidden)),
            voice: voice(hidden: nanoHidden, promptTokens: 250, promptFeatRows: 500))
    }

    func testOversizedPromptFeatRejected() {
        // A [1001, 80] prompt mel would write 80 floats past the fixed
        // [1, 1000, 80] flow buffer if it reached the pointer copy.
        XCTAssertThrowsError(
            try ChatterboxTables.validate(
                mtlTables(),
                voice: voice(
                    hidden: ChatterboxConstants.hiddenSize, promptTokens: 157,
                    promptFeatRows: 1001)))
        // Rows consistent with tokens but exceeding the mel bucket.
        XCTAssertThrowsError(
            try ChatterboxTables.validate(
                mtlTables(),
                voice: voice(
                    hidden: ChatterboxConstants.hiddenSize, promptTokens: 501,
                    promptFeatRows: 1002)))
    }

    func testPromptFeatTokenMismatchRejected() {
        XCTAssertThrowsError(
            try ChatterboxTables.validate(
                mtlTables(),
                voice: voice(
                    hidden: ChatterboxConstants.hiddenSize, promptTokens: 157,
                    promptFeatRows: 316)))
    }

    func testUndersizedPositionalTablesRejected() {
        let goodVoice = voice(
            hidden: ChatterboxConstants.hiddenSize, promptTokens: 157, promptFeatRows: 314)
        // A one-row text_pos_emb would trap at row(1) on ordinary text.
        XCTAssertThrowsError(
            try ChatterboxTables.validate(mtlTables(textPosRows: 1), voice: goodVoice))
        XCTAssertThrowsError(
            try ChatterboxTables.validate(mtlTables(speechPosRows: 10), voice: goodVoice))
    }

    // MARK: - Safetensors header hardening

    /// Build a syntactically valid safetensors file from a JSON header
    /// string and a payload.
    private func writeSafetensors(header: String, payload: Data) throws -> URL {
        let headerData = Data(header.utf8)
        var file = Data()
        withUnsafeBytes(of: UInt64(headerData.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(headerData)
        file.append(payload)
        let url = tmpDir.appendingPathComponent("tables.safetensors")
        try file.write(to: url)
        return url
    }

    func testShapeProductOverflowThrowsInsteadOfTrapping() throws {
        // shape [Int.max, 2] passes non-negative validation; the rows×cols
        // product must throw, not fatally overflow.
        let header = """
            {"text_emb": {"dtype": "F32", "shape": [9223372036854775807, 2], "data_offsets": [0, 8]},
             "speech_emb": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]}}
            """
        let url = try writeSafetensors(header: header, payload: Data(count: 8))
        XCTAssertThrowsError(try ChatterboxTables.loadNano(tablesURL: url)) { error in
            guard case ChatterboxError.malformedAsset(let detail) = error else {
                return XCTFail("expected malformedAsset, got \(error)")
            }
            XCTAssertTrue(detail.contains("overflow"), detail)
        }
    }

    func testReversedOffsetsThrowInsteadOfTrapping() throws {
        let header = """
            {"text_emb": {"dtype": "F32", "shape": [1, 2], "data_offsets": [8, 0]}}
            """
        let url = try writeSafetensors(header: header, payload: Data(count: 8))
        XCTAssertThrowsError(try ChatterboxTables.loadNano(tablesURL: url))
    }

    func testOutOfPayloadOffsetsThrowInsteadOfTrapping() throws {
        let header = """
            {"text_emb": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 4096]}}
            """
        let url = try writeSafetensors(header: header, payload: Data(count: 8))
        XCTAssertThrowsError(try ChatterboxTables.loadNano(tablesURL: url))
    }

    func testHugeHeaderLengthThrowsInsteadOfTrapping() throws {
        // Header length UInt64.max would overflow Int arithmetic.
        var file = Data()
        withUnsafeBytes(of: UInt64.max.littleEndian) { file.append(contentsOf: $0) }
        file.append(Data("{}".utf8))
        let url = tmpDir.appendingPathComponent("tables.safetensors")
        try file.write(to: url)
        XCTAssertThrowsError(try ChatterboxTables.loadNano(tablesURL: url))
    }

    func testMisalignedTensorBytesThrowInsteadOfSilentTruncation() throws {
        // 7 payload bytes for an F32 tensor is not 4-aligned.
        let header = """
            {"text_emb": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 7]}}
            """
        let url = try writeSafetensors(header: header, payload: Data(count: 7))
        XCTAssertThrowsError(try ChatterboxTables.loadNano(tablesURL: url))
    }
}
