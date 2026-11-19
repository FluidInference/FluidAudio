import LuxTtsG2pResources
import XCTest

@testable import FluidAudio

/// The LuxTTS G2P tables live in their own SwiftPM target so the `TTS` trait
/// can prune the bundle (#990). Pin that the bundle still carries both files
/// at its root, which is what `LuxTtsG2p.init` looks up.
final class LuxTtsG2pResourcesTests: XCTestCase {

    func testBundleContainsLexiconAndAuxTablesAtRoot() throws {
        let bundle = LuxTtsG2pResources.bundle

        let lexicon = try XCTUnwrap(
            bundle.url(forResource: "luxtts_en_us_lexicon.tsv", withExtension: "zz"),
            "compressed lexicon missing from LuxTtsG2pResources bundle")
        let aux = try XCTUnwrap(
            bundle.url(forResource: "luxtts_en_us_g2p_aux", withExtension: "json"),
            "G2P aux tables missing from LuxTtsG2pResources bundle")

        // `resourceURL` is the bundle root on iOS and `Contents/Resources` on macOS.
        let root = try XCTUnwrap(bundle.resourceURL).standardizedFileURL.path
        XCTAssertEqual(lexicon.deletingLastPathComponent().standardizedFileURL.path, root)
        XCTAssertEqual(aux.deletingLastPathComponent().standardizedFileURL.path, root)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: root + "/Resources"),
            "`.process` must flatten the Resources directory (a top-level Resources dir breaks iOS code signing)")
    }

    func testBundledFilesAreNonEmpty() throws {
        let bundle = LuxTtsG2pResources.bundle
        for (name, ext) in [("luxtts_en_us_lexicon.tsv", "zz"), ("luxtts_en_us_g2p_aux", "json")] {
            let url = try XCTUnwrap(bundle.url(forResource: name, withExtension: ext))
            let size = try XCTUnwrap(
                FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)
            XCTAssertGreaterThan(size.intValue, 0, "\(name).\(ext) is empty")
        }
    }
}
