import Foundation
import XCTest

@testable import FluidAudio

/// `CtcModels.download` and `load` forward their progress handler to `ModelHub.loadModels`. Unlike the other CTC
/// tests, which run from the cache, this always downloads the real CTC 110M models into a temporary directory, so it
/// runs only with `FLUIDAUDIO_RUN_ASR_E2E=1`.
final class CtcModelsDownloadProgressTests: XCTestCase {

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("CtcDownloadProgress-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workDir)
    }

    func testDownloadAndLoadReportProgressToTheHandler() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1",
            "Downloads CTC 110M; set FLUIDAUDIO_RUN_ASR_E2E=1 to run")
        let target = workDir.appendingPathComponent(CtcModelVariant.ctc110m.repo.folderName, isDirectory: true)
        let recorder = ProgressStreamRecorder()

        try await CtcModels.download(to: target, variant: .ctc110m, progressHandler: { recorder.append($0) })

        XCTAssertTrue(CtcModels.modelsExist(at: target))
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
        XCTAssertTrue(
            Self.compiledModels(events).isSuperset(of: ModelNames.CTC.requiredModels),
            "compile phase missing a bundle: \(Self.compiledModels(events))")
        XCTAssertTrue(events.allSatisfy { (0...1).contains($0.fractionCompleted) })
        XCTAssertEqual(events.last?.fractionCompleted, 1.0)

        // From the cache now: `load` still reports each bundle's compile.
        let loadRecorder = ProgressStreamRecorder()
        _ = try await CtcModels.load(from: target, variant: .ctc110m, progressHandler: { loadRecorder.append($0) })
        let loadEvents = loadRecorder.snapshot()
        XCTAssertTrue(Self.compiledModels(loadEvents).isSuperset(of: ModelNames.CTC.requiredModels))
        XCTAssertEqual(loadEvents.last?.fractionCompleted, 1.0)
    }

    private static func compiledModels(_ events: [DownloadProgress]) -> Set<String> {
        Set(
            events.compactMap {
                if case .compiling(let modelName) = $0.phase { return modelName }
                return nil
            })
    }
}
