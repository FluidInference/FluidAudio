import Foundation
import XCTest

@testable import FluidAudio

/// Network-free tests for `MandarinPolyphoneCatalog` — the parser for
/// `POLYPHONIC_CHARS.txt` and the diacritic → digit conversion that
/// adapts upstream g2pW labels to the v1.1-zh vocab style.
final class MandarinPolyphoneCatalogTests: XCTestCase {

    private static let sample = """
        # comment lines and blanks are skipped
        行\tㄒㄧㄥˊ
        行\tㄏㄤˊ
        行\tㄒㄧㄥˋ
        长\tㄔㄤˊ
        长\tㄓㄤˇ
        都\tㄉㄡ
        都\tㄉㄨ
        """

    func testParsesCharsInOrder() throws {
        let cat = try MandarinPolyphoneCatalog(text: Self.sample)
        XCTAssertEqual(cat.chars, ["行", "长", "都"])
        // Reverse map matches the order.
        XCTAssertEqual(cat.charIndex["行"], 0)
        XCTAssertEqual(cat.charIndex["长"], 1)
        XCTAssertEqual(cat.charIndex["都"], 2)
    }

    func testLabelsAreSortedUnique() throws {
        let cat = try MandarinPolyphoneCatalog(text: Self.sample)
        // 7 entries, but two of them (`ㄉㄡ` `ㄉㄨ`) and the two
        // `行` rows that share `ㄒㄧㄥ` differ — distinct. So total
        // unique label count is 7.
        XCTAssertEqual(cat.labels.count, 7)
        XCTAssertEqual(cat.labels, cat.labels.sorted())
    }

    func testCandidatesPerChar() throws {
        let cat = try MandarinPolyphoneCatalog(text: Self.sample)
        let xing = cat.candidates(for: "行")
        XCTAssertNotNil(xing)
        XCTAssertEqual(xing?.count, 3)
        let zhang = cat.candidates(for: "长")
        XCTAssertEqual(zhang?.count, 2)
        // Non-polyphonic char lookup returns nil.
        XCTAssertNil(cat.candidates(for: "好"))
    }

    func testBopomofoReverseLookup() throws {
        let cat = try MandarinPolyphoneCatalog(text: Self.sample)
        let labels = Set(cat.labels)
        XCTAssertTrue(labels.contains("ㄒㄧㄥˊ"))
        XCTAssertTrue(labels.contains("ㄏㄤˊ"))
        // Round-trip via the digit form.
        let xingCandidates = try XCTUnwrap(cat.candidates(for: "行"))
        let digitForms = xingCandidates.compactMap { cat.bopomofoDigitForm(forLabel: $0) }
        XCTAssertEqual(Set(digitForms), Set(["ㄒㄧㄥ2", "ㄏㄤ2", "ㄒㄧㄥ4"]))
    }

    func testToneDigitConversion() {
        // Each diacritic maps to its tone digit.
        XCTAssertEqual(MandarinPolyphoneCatalog.toneDigitForm("ㄒㄧㄥˊ"), "ㄒㄧㄥ2")
        XCTAssertEqual(MandarinPolyphoneCatalog.toneDigitForm("ㄏㄠˇ"), "ㄏㄠ3")
        XCTAssertEqual(MandarinPolyphoneCatalog.toneDigitForm("ㄕˋ"), "ㄕ4")
        XCTAssertEqual(MandarinPolyphoneCatalog.toneDigitForm("ㄉㄜ˙"), "ㄉㄜ5")
        // Missing diacritic ⇒ implicit tone 1.
        XCTAssertEqual(MandarinPolyphoneCatalog.toneDigitForm("ㄉㄡ"), "ㄉㄡ1")
        // Empty input returns empty.
        XCTAssertEqual(MandarinPolyphoneCatalog.toneDigitForm(""), "")
    }

    func testRejectsMalformedRow() {
        let bad = "行\n长\tㄔㄤˊ\n"
        XCTAssertThrowsError(try MandarinPolyphoneCatalog(text: bad)) { err in
            guard let e = err as? MandarinPolyphoneCatalog.LoadError,
                case .malformed = e
            else {
                XCTFail("Expected malformed error, got \(err)")
                return
            }
        }
    }

    func testRejectsMultiHanziKey() {
        let bad = "行人\tㄒㄧㄥˊ\n"
        XCTAssertThrowsError(try MandarinPolyphoneCatalog(text: bad))
    }

    func testHandlesCRLFAndBlanks() throws {
        let crlf = "行\tㄒㄧㄥˊ\r\n\r\n# comment\r\n长\tㄓㄤˇ\r\n"
        let cat = try MandarinPolyphoneCatalog(text: crlf)
        XCTAssertEqual(cat.chars, ["行", "长"])
    }

    func testDeduplicatesRepeatedRows() throws {
        let dup = "行\tㄒㄧㄥˊ\n行\tㄒㄧㄥˊ\n行\tㄏㄤˊ\n"
        let cat = try MandarinPolyphoneCatalog(text: dup)
        // Only two distinct labels for 行.
        XCTAssertEqual(cat.candidates(for: "行")?.count, 2)
    }

    func testInvalidLabelIndexReturnsNil() throws {
        let cat = try MandarinPolyphoneCatalog(text: Self.sample)
        XCTAssertNil(cat.bopomofo(forLabel: -1))
        XCTAssertNil(cat.bopomofo(forLabel: cat.labels.count + 100))
    }
}
