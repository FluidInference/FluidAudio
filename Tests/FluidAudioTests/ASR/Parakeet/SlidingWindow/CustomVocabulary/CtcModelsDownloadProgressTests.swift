import Foundation
import XCTest

@testable import FluidAudio

/// `CtcModels` loads its two bundles with separate `ModelHub.loadModels` calls, and each call reports its own
/// sequence from 0 to 1. The progress handler must still see one sequence from 0 to 1 per `CtcModels` call.
final class CtcModelsDownloadProgressTests: XCTestCase {

    private static let bundles: Set<String> = [ModelNames.CTC.melSpectrogramPath, ModelNames.CTC.audioEncoderPath]

    func testScaledHandlerMapsEachLoadIntoItsShareOfTheRange() {
        let recorder = ProgressStreamRecorder()
        let load: [DownloadProgress] = [
            DownloadProgress(fractionCompleted: 0.0, phase: .listing),
            DownloadProgress(fractionCompleted: 0.5, phase: .compiling(modelName: "Model.mlmodelc")),
            DownloadProgress(fractionCompleted: 1.0, phase: .compiling(modelName: "")),
        ]
        for step in 0..<2 {
            let handler = CtcModels.scaledProgressHandler({ recorder.append($0) }, step: step, of: 2)
            load.forEach { handler?($0) }
        }

        let events = recorder.snapshot()
        XCTAssertEqual(events.map(\.fractionCompleted), [0.0, 0.25, 0.5, 0.75, 1.0])
        let closing = events.filter {
            if case .compiling(let modelName) = $0.phase { return modelName.isEmpty }
            return false
        }
        XCTAssertEqual(closing.map(\.fractionCompleted), [1.0], "only the last load may report the closing event")
        XCTAssertNil(CtcModels.scaledProgressHandler(nil, step: 0, of: 2))
    }

    /// Needs the CTC 110M models; runs when they are cached or `FLUIDAUDIO_RUN_ASR_E2E=1` allows a download.
    func testLoadAndDownloadAndLoadReportOneSequence() async throws {
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        let directory = CtcModels.defaultCacheDirectory(for: .ctc110m)
        try XCTSkipUnless(
            CtcModels.modelsExist(at: directory) || allowDownload,
            "CTC 110M models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")

        let downloadAndLoad = ProgressStreamRecorder()
        _ = try await CtcModels.downloadAndLoad(progressHandler: { downloadAndLoad.append($0) })
        Self.assertOneSequence(downloadAndLoad.snapshot())

        let load = ProgressStreamRecorder()
        _ = try await CtcModels.load(from: directory, progressHandler: { load.append($0) })
        Self.assertOneSequence(load.snapshot())

        let download = ProgressStreamRecorder()
        try await CtcModels.download(progressHandler: { download.append($0) })
        XCTAssertTrue(download.snapshot().isEmpty, "download reported progress for models already present")
    }

    /// Downloads CTC 110M into a temporary directory, so it runs only with `FLUID_NETWORK_TESTS=1`.
    func testFreshDownloadReportsDownloadThenCompileAsOneSequence() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUID_NETWORK_TESTS"] == "1",
            "Downloads CTC 110M; set FLUID_NETWORK_TESTS=1 to run")
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("CtcDownloadProgress-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: workDir) }
        let target = workDir.appendingPathComponent(CtcModelVariant.ctc110m.repo.folderName, isDirectory: true)

        let recorder = ProgressStreamRecorder()
        _ = try await CtcModels.downloadAndLoad(to: target, progressHandler: { recorder.append($0) })

        let events = recorder.snapshot()
        guard case .listing = events.first?.phase else { return XCTFail("first emission should be .listing") }
        XCTAssertEqual(events.first?.fractionCompleted, 0.0)
        let downloadedEveryFile = events.contains {
            if case .downloading(let completedFiles, let totalFiles) = $0.phase {
                return totalFiles > 0 && completedFiles == totalFiles
            }
            return false
        }
        XCTAssertTrue(downloadedEveryFile, "no emission reports completedFiles == totalFiles")
        Self.assertOneSequence(events)
    }

    private static func assertOneSequence(
        _ events: [DownloadProgress], file: StaticString = #filePath, line: UInt = #line
    ) {
        let fractions = events.map(\.fractionCompleted)
        XCTAssertEqual(fractions, fractions.sorted(), "fraction went backwards: \(fractions)", file: file, line: line)
        XCTAssertEqual(fractions.last, 1.0, file: file, line: line)
        XCTAssertEqual(
            fractions.filter { $0 == 1.0 }.count, 1, "reached 1.0 more than once: \(fractions)", file: file,
            line: line)
        let compiled = Set(
            events.compactMap {
                if case .compiling(let modelName) = $0.phase, !modelName.isEmpty { return modelName }
                return nil
            })
        XCTAssertEqual(compiled, bundles, file: file, line: line)
    }
}
