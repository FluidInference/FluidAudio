import XCTest

@testable import FluidAudio

final class KokoroAneComputeUnitsTests: XCTestCase {

    private func version(_ major: Int, _ minor: Int, _ patch: Int = 0) -> OperatingSystemVersion {
        OperatingSystemVersion(majorVersion: major, minorVersion: minor, patchVersion: patch)
    }

    func testDefaultOn26LineIsAneTailGpu() {
        XCTAssertEqual(KokoroAneComputeUnits.defaultUnits(for: version(26, 5)), .aneTailGpu)
        XCTAssertEqual(KokoroAneComputeUnits.defaultUnits(for: version(26, 6)), .aneTailGpu)
        XCTAssertEqual(KokoroAneComputeUnits.defaultUnits(for: version(25, 0)), .aneTailGpu)
    }

    // #843: on the OS 27 line the GPU stages abort in MPSGraph under CoreML,
    // so the default routes noise + tail to CPU.
    func testDefaultOn27LineIsAneTailCpu() {
        XCTAssertEqual(KokoroAneComputeUnits.defaultUnits(for: version(27, 0)), .aneTailCpu)
        XCTAssertEqual(KokoroAneComputeUnits.defaultUnits(for: version(27, 1)), .aneTailCpu)
        XCTAssertEqual(KokoroAneComputeUnits.defaultUnits(for: version(28, 0)), .aneTailCpu)
    }

    func testAneTailCpuNeverTouchesGpu() {
        let units = KokoroAneComputeUnits.aneTailCpu
        let stages = [
            units.albert, units.postAlbert, units.alignment, units.prosody,
            units.noise, units.vocoder, units.tail,
        ]
        for stage in stages {
            XCTAssertNotEqual(stage, .cpuAndGPU)
            XCTAssertNotEqual(stage, .all)
        }
        XCTAssertEqual(units.noise, .cpuOnly)
        XCTAssertEqual(units.tail, .cpuOnly)
    }
}
