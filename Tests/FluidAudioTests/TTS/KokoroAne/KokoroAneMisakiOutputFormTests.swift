import XCTest

@testable import FluidAudio

/// misaki's `ɾ` → `T` flap step on the English frontend output.
final class KokoroAneMisakiOutputFormTests: XCTestCase {

    func testFlapBecomesT() {
        XCTAssertEqual(KokoroAneManager.misakiOutputForm("mˈɛɾᵊl wˈɔɾəɹ"), "mˈɛTᵊl wˈɔTəɹ")
    }

    func testGlottalStopIsKept() {
        XCTAssertEqual(KokoroAneManager.misakiOutputForm("bˈʌʔn"), "bˈʌʔn")
    }
}
