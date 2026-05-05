import Foundation
import XCTest

@testable import FluidAudio

/// Tests for the bundled-voice catalog and the named-voice resolver
/// extension on `StyleTTS2VoiceStyle`. No CoreML / no HF download needed.
final class StyleTTS2VoicePresetsTests: XCTestCase {

    func testCatalogShipsExpected17VoiceCount() {
        // Sourced from yl4579/StyleTTS2-LibriTTS/reference_audio.zip — 17 clips.
        XCTAssertEqual(StyleTTS2VoicePresets.all.count, 17)
        XCTAssertEqual(StyleTTS2VoicePresets.requiredFilenames.count, 17)
    }

    func testEveryFilenameIsRefSBin() {
        for voice in StyleTTS2VoicePresets.all {
            XCTAssertTrue(
                voice.filename.hasPrefix("ref_s_"),
                "filename '\(voice.filename)' missing ref_s_ prefix")
            XCTAssertTrue(
                voice.filename.hasSuffix(".bin"),
                "filename '\(voice.filename)' missing .bin suffix")
        }
    }

    func testIDsAreLowercaseUniqueAndDotFree() {
        let ids = StyleTTS2VoicePresets.all.map { $0.id }
        XCTAssertEqual(Set(ids).count, ids.count, "duplicate voice ids: \(ids)")
        for id in ids {
            XCTAssertEqual(id, id.lowercased(), "id '\(id)' is not lowercase")
            XCTAssertFalse(id.contains("."), "id '\(id)' contains '.'")
            XCTAssertFalse(id.contains("/"), "id '\(id)' contains '/'")
        }
    }

    func testDefaultVoiceIDResolvesToVinay() {
        XCTAssertEqual(StyleTTS2VoicePresets.defaultVoiceID, "vinay")
        XCTAssertNotNil(
            StyleTTS2VoicePresets.voice(forID: StyleTTS2VoicePresets.defaultVoiceID),
            "default voice id must resolve to a catalog entry")
    }

    func testVoiceLookupIsCaseInsensitive() {
        XCTAssertEqual(
            StyleTTS2VoicePresets.voice(forID: "Vinay")?.id, "vinay")
        XCTAssertEqual(
            StyleTTS2VoicePresets.voice(forID: "VINAY")?.id, "vinay")
        XCTAssertNil(
            StyleTTS2VoicePresets.voice(forID: "this-id-doesnt-exist"))
    }

    func testCohortsCoverEveryVoice() {
        let allCohortVoices = StyleTTS2VoicePresets.Cohort.allCases.flatMap {
            cohort in
            StyleTTS2VoicePresets.all.filter { $0.cohort == cohort }
        }
        XCTAssertEqual(
            allCohortVoices.count,
            StyleTTS2VoicePresets.all.count,
            "every voice must belong to a Cohort case")
    }

    func testVoiceURLBuildsExpectedPath() throws {
        let root = URL(fileURLWithPath: "/tmp/fake-styletts2-root", isDirectory: true)
        let url = try XCTUnwrap(
            StyleTTS2VoiceStyle.voiceURL(forID: "vinay", in: root))
        XCTAssertEqual(url.path, "/tmp/fake-styletts2-root/voices/ref_s_Vinay.bin")
    }

    func testVoiceURLReturnsNilForUnknownID() {
        let root = URL(fileURLWithPath: "/tmp/fake-styletts2-root", isDirectory: true)
        XCTAssertNil(StyleTTS2VoiceStyle.voiceURL(forID: "no-such-voice", in: root))
    }

    func testNamedLoadsBlobFromRepoRoot() throws {
        // Build a synthetic bundle containing one blob and resolve it via named().
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("styletts2-named-tests-\(UUID().uuidString)")
        let voicesDir = root.appendingPathComponent("voices", isDirectory: true)
        try FileManager.default.createDirectory(
            at: voicesDir, withIntermediateDirectories: true)

        // Build a deterministic 256-fp32 blob: index 0..255 ramp (acoustic 1.0,
        // prosody -1.0 split would lose ordering info).
        var floats: [Float] = (0..<256).map { Float($0) }
        let blobURL = voicesDir.appendingPathComponent("ref_s_Vinay.bin")
        let data = floats.withUnsafeMutableBufferPointer { Data(buffer: $0) }
        try data.write(to: blobURL)

        let style = try StyleTTS2VoiceStyle.named("vinay", in: root)
        XCTAssertEqual(style.concatenated.count, 256)
        XCTAssertEqual(style.concatenated.first, 0)
        XCTAssertEqual(style.concatenated.last, 255)

        // Cleanup
        try? FileManager.default.removeItem(at: root)
    }

    func testNamedThrowsForUnknownID() {
        let root = URL(fileURLWithPath: "/tmp/fake-styletts2-root", isDirectory: true)
        XCTAssertThrowsError(try StyleTTS2VoiceStyle.named("no-such-voice", in: root)) {
            error in
            guard case StyleTTS2Error.modelNotFound = error else {
                XCTFail("expected modelNotFound, got \(error)")
                return
            }
        }
    }

    func testRequiredModelsIncludesVoicesDir() {
        XCTAssertTrue(
            ModelNames.StyleTTS2.requiredModels.contains(ModelNames.StyleTTS2.voicesDir),
            "StyleTTS2 requiredModels must include the voices/ directory entry "
                + "or the downloader will skip it")
    }
}
