import XCTest

@testable import FluidAudio

final class OfflineDiarizerOsAdvisoryTests: XCTestCase {

    private func version(_ major: Int, _ minor: Int, _ patch: Int = 0) -> OperatingSystemVersion {
        OperatingSystemVersion(majorVersion: major, minorVersion: minor, patchVersion: patch)
    }

    // #878: the entire macOS 14 line crashes in libBNNS regardless of
    // patch level (1200/1200 on 14.8.7 CI runners).
    func testMacOS14LineIsFlagged() {
        XCTAssertTrue(OfflineDiarizerManager.isBnnsCrashProneOS(version(14, 0), onMacOS: true))
        XCTAssertTrue(OfflineDiarizerManager.isBnnsCrashProneOS(version(14, 6, 1), onMacOS: true))
        XCTAssertTrue(OfflineDiarizerManager.isBnnsCrashProneOS(version(14, 8, 7), onMacOS: true))
    }

    func testFixedAndUnaffectedMacOSLinesAreNotFlagged() {
        XCTAssertFalse(OfflineDiarizerManager.isBnnsCrashProneOS(version(13, 6), onMacOS: true))
        XCTAssertFalse(OfflineDiarizerManager.isBnnsCrashProneOS(version(15, 0), onMacOS: true))
        XCTAssertFalse(OfflineDiarizerManager.isBnnsCrashProneOS(version(15, 7, 7), onMacOS: true))
        XCTAssertFalse(OfflineDiarizerManager.isBnnsCrashProneOS(version(26, 5, 2), onMacOS: true))
    }

    // No reproduction reported on iOS; only macOS is flagged.
    func testIOSIsNotFlagged() {
        XCTAssertFalse(OfflineDiarizerManager.isBnnsCrashProneOS(version(14, 0), onMacOS: false))
        XCTAssertFalse(OfflineDiarizerManager.isBnnsCrashProneOS(version(17, 0), onMacOS: false))
        XCTAssertFalse(OfflineDiarizerManager.isBnnsCrashProneOS(version(18, 0), onMacOS: false))
    }
}
