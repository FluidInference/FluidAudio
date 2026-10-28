import XCTest

@testable import FluidAudio

/// Number reading used when the NeMo engine is not linked. Expected strings
/// are NeMo's own readings, so output does not change with the trait.
final class RomanceNumberNormalizerTests: XCTestCase {

    func testSpanishCardinalsMatchNeMo() {
        let cases: [(Int64, String)] = [
            (0, "cero"), (1, "un"), (16, "dieciséis"), (21, "veintiún"), (31, "treinta y un"),
            (100, "cien"), (101, "ciento un"), (201, "doscientos un"), (1000, "mil"), (1001, "mil un"),
            (2024, "dos mil veinticuatro"), (21000, "veintiún mil"), (1_000_000, "un millón"),
            (2_000_000, "dos millones"), (1_000_000_000, "mil millones"),
        ]
        for (n, expected) in cases {
            XCTAssertEqual(RomanceNumberNormalizer.cardinal(n, .spanish), expected, "\(n)")
        }
    }

    func testFrenchCardinalsMatchNeMo() {
        let cases: [(Int64, String)] = [
            (0, "zéro"), (21, "vingt et un"), (22, "vingt-deux"), (71, "soixante et onze"), (80, "quatre-vingts"),
            (81, "quatre-vingt-un"), (91, "quatre-vingt-onze"), (100, "cent"), (200, "deux cents"),
            (201, "deux cent un"), (1001, "mille et un"), (2024, "deux mille vingt-quatre"),
            (21000, "vingt et un mille"), (80000, "quatre-vingts mille"), (1_000_000, "un million"),
            (2_000_000, "deux millions"), (1_000_000_000, "un milliard"),
        ]
        for (n, expected) in cases {
            XCTAssertEqual(RomanceNumberNormalizer.cardinal(n, .french), expected, "\(n)")
        }
    }

    func testInlineNumbersDecimalsAndPercent() {
        XCTAssertEqual(
            RomanceNumberNormalizer.normalize("El 2024 fue un año difícil.", language: .spanish),
            "El dos mil veinticuatro fue un año difícil.")
        XCTAssertEqual(
            RomanceNumberNormalizer.normalize("3,5 y 50%", language: .spanish), "tres coma cinco y cincuenta por ciento"
        )
        XCTAssertEqual(RomanceNumberNormalizer.normalize("1.000 casas", language: .spanish), "mil casas")
        XCTAssertEqual(RomanceNumberNormalizer.normalize("1 000 ans", language: .french), "mille ans")
        XCTAssertEqual(
            RomanceNumberNormalizer.normalize("3,5 et 50 %", language: .french),
            "trois virgule cinq et cinquante pour cent")
    }
}
