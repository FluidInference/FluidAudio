import XCTest

@testable import FluidAudio

final class KokoroAneOsAdvisoryTests: XCTestCase {

    private func version(_ major: Int, _ minor: Int, _ patch: Int = 0) -> OperatingSystemVersion {
        OperatingSystemVersion(majorVersion: major, minorVersion: minor, patchVersion: patch)
    }

    func testAffectedLineIsFlagged() {
        XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 4)))
        XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 4, 2)))
        XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 5)))
        XCTAssertTrue(KokoroAneManager.isBnnsCrashProneOS(version(26, 5, 2)))
    }

    func testFixedAndUnaffectedLinesAreNotFlagged() {
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(26, 3, 1)))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(26, 6)))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(26, 7)))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(27, 0)))
        XCTAssertFalse(KokoroAneManager.isBnnsCrashProneOS(version(25, 5)))
    }
}
