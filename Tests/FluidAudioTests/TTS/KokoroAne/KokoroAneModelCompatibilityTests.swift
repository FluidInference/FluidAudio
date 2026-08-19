import Foundation
import XCTest

@testable import FluidAudio

final class KokoroAneModelCompatibilityTests: XCTestCase {

    func testRecognizesFlexibleShapeInformationOnDynamicMilFunction() {
        // Exact structure emitted by the current Core ML compiler for the
        // published KokoroPostAlbert bundle, reduced to the function header.
        let mil = Data(
            """
            program(1.0) {
                func main<ios17>(tensor<int32, [1, ?]> input_ids)
                    [FlexibleShapeInformation = tuple<tuple<tensor<string, []>>>()] {
                }
            }
            """.utf8)

        XCTAssertTrue(
            KokoroAneModelCompatibility.milProgramHasFlexibleShapeInformation(mil))
    }

    func testRejectsLegacyDynamicMilFunctionWithoutFlexibleShapeInformation() {
        // This is the model shape that produces the E5 warning in #738: the
        // external input is dynamic, but the function has no shape attribute.
        let legacyMil = Data(
            """
            program(1.0) {
                func main<ios17>(tensor<int32, [1, ?]> input_ids) {
                }
            }
            """.utf8)

        XCTAssertFalse(
            KokoroAneModelCompatibility.milProgramHasFlexibleShapeInformation(
                legacyMil))
    }

    func testRejectsEmptyMilProgram() {
        XCTAssertFalse(
            KokoroAneModelCompatibility.milProgramHasFlexibleShapeInformation(Data()))
    }
}
