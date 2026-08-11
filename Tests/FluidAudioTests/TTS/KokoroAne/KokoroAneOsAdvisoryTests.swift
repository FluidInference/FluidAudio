import XCTest

@testable import FluidAudio

final class KokoroAneOsAdvisoryTests: XCTestCase {

    private func version(_ major: Int, _ minor: Int, _ patch: Int = 0) -> OperatingSystemVersion {
        OperatingSystemVersion(majorVersion: major, minorVersion: minor, patchVersion: patch)
    }

    func testAffectedLineIsFlaggedOnAllPlatforms() {
        for onMacOS in [true, false] {
            XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 4), onMacOS: onMacOS))
            XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 4, 2), onMacOS: onMacOS))
            XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 5), onMacOS: onMacOS))
            XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 5, 2), onMacOS: onMacOS))
        }
    }

    func testMacOSFixedAndUnaffectedLinesAreNotFlagged() {
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(26, 3, 1), onMacOS: true))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(26, 6), onMacOS: true))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(26, 7), onMacOS: true))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(27, 0), onMacOS: true))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(25, 5), onMacOS: true))
    }

    // #844: iOS 26.6 still reproduces the libBNNS SIGSEGV that macOS 26.6
    // fixed, so on non-macOS the whole 26.4+ line stays flagged.
    func testIOSStaysFlaggedThrough26Line() {
        XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 6), onMacOS: false))
        XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 6, 1), onMacOS: false))
        XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 7), onMacOS: false))
    }

    func testIOSUnaffectedLinesAreNotFlagged() {
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(26, 3, 1), onMacOS: false))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(27, 0), onMacOS: false))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(25, 5), onMacOS: false))
    }
}
