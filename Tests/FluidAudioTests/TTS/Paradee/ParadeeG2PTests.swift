import XCTest

@testable import FluidAudio

/// The text side of Paradee's eSpeak-free G2P, against upstream
/// `paradee/g2p.py` (package 1.3.1). Expected values come from running the
/// upstream functions. Number tables map each piece to its own key, so the
/// expected lists show which pieces a number is put together from.
final class ParadeeG2PTests: XCTestCase {

    private struct Keys {
        var overrides: [String: String] = [:]

        /// A table holding every piece the composition functions can ask for,
        /// each mapped to its key unless overridden.
        var table: [String: String] {
            var t: [String: String] = [:]
            for n in 0..<100 { t[String(n)] = String(n) }
            for h in stride(from: 100, through: 900, by: 100) { t[String(h)] = String(h) }
            for k in [
                "thousand", "lakh", "crore", "million", "millions", "milliard", "milliards", "millón", "millones",
                "100+",
            ] {
                t[k] = k
            }
            return t.merging(overrides) { $1 }
        }
    }

    // MARK: - Numbers

    func testHindiNumberPieces() {
        let t = Keys().table
        XCTAssertEqual(ParadeeG2PNumbers.hindi(7, t), ["7"])
        XCTAssertEqual(ParadeeG2PNumbers.hindi(100, t), ["100"])
        XCTAssertEqual(ParadeeG2PNumbers.hindi(125_000, t), ["1", "lakh", "25", "thousand"])
        XCTAssertEqual(
            ParadeeG2PNumbers.hindi(12_345_678, t), ["1", "crore", "23", "lakh", "45", "thousand", "600", "78"])
        XCTAssertEqual(
            ParadeeG2PNumbers.hindi(999_999_999, t), ["99", "crore", "99", "lakh", "99", "thousand", "900", "99"])
    }

    func testFrenchNumberPiecesStressOnlyTheLastWord() {
        let t = Keys(overrides: [
            "1": "ˈœ̃", "2": "dˈø", "thousand": "mˈil", "million": "miljˈɔ̃", "millions": "miljˈɔ̃",
        ]).table
        XCTAssertEqual(ParadeeG2PNumbers.french(0, t), ["0"])
        XCTAssertEqual(ParadeeG2PNumbers.french(1001, t), ["mil", "ˈœ̃"])
        XCTAssertEqual(ParadeeG2PNumbers.french(2000, t), ["dø", "mˈil"])
        XCTAssertEqual(ParadeeG2PNumbers.french(1250, t), ["mil", "200", "50"])
        XCTAssertEqual(ParadeeG2PNumbers.french(1_000_000, t), ["œ̃", "miljˈɔ̃"])
        XCTAssertEqual(ParadeeG2PNumbers.french(2_000_000, t), ["dø", "miljˈɔ̃"])
        XCTAssertEqual(ParadeeG2PNumbers.french(21000, t), ["21", "mˈil"])
    }

    func testSpanishNumberPieces() {
        let t = Keys(overrides: ["1": "uno", "21": "beintiuno", "31": "tɾeintajuno", "3": "tres"]).table
        XCTAssertEqual(ParadeeG2PNumbers.spanish(0, t), ["0"])
        XCTAssertEqual(ParadeeG2PNumbers.spanish(100, t), ["100"])
        XCTAssertEqual(ParadeeG2PNumbers.spanish(101, t), ["100+", "uno"])
        XCTAssertEqual(ParadeeG2PNumbers.spanish(1000, t), ["thousand"])
        XCTAssertEqual(ParadeeG2PNumbers.spanish(3500, t), ["tres", "thousand", "500"])
        // veintiuno mil -> veintiún mil, but not for 11
        XCTAssertEqual(ParadeeG2PNumbers.spanish(21000, t), ["beintiun", "thousand"])
        XCTAssertEqual(ParadeeG2PNumbers.spanish(31000, t), ["tɾeintajun", "thousand"])
        XCTAssertEqual(ParadeeG2PNumbers.spanish(11000, t), ["11", "thousand"])
        XCTAssertEqual(ParadeeG2PNumbers.spanish(1_000_000, t), ["millón"])
        XCTAssertEqual(ParadeeG2PNumbers.spanish(2_500_000, t), ["2", "millones", "500", "thousand"])
    }

    func testSpanishSoftening() {
        XCTAssertEqual(ParadeeG2PNumbers.soften("dˈos", before: "a"), "ðˈos")
        XCTAssertEqual(ParadeeG2PNumbers.soften("dˈos", before: "#"), "dˈos")
        XCTAssertEqual(ParadeeG2PNumbers.soften("dˈos", before: "n"), "dˈos")
        XCTAssertEqual(ParadeeG2PNumbers.soften("dˈos", before: "l"), "dˈos")
        XCTAssertEqual(ParadeeG2PNumbers.soften("bˈeinte", before: "l"), "βˈeinte")
        XCTAssertEqual(ParadeeG2PNumbers.soften("ɡato", before: "s"), "ɣato")
        XCTAssertEqual(ParadeeG2PNumbers.soften("ˈuno", before: "a"), "ˈuno")
        XCTAssertEqual(ParadeeG2PNumbers.soften("dˈos", before: ""), "dˈos")
    }

    func testSpanishNumberSoftensAcrossPieces() {
        // Each piece is softened by the last sound of the piece before it.
        let t = Keys(overrides: ["2": "dˈos", "thousand": "mˈil", "200": "doθjˈentos"]).table
        XCTAssertEqual(ParadeeG2PNumbers.read(2200, language: .spanish, table: t, before: "a"), "ðˈos mˈil doθjˈentos")
        XCTAssertEqual(ParadeeG2PNumbers.read(2200, language: .spanish, table: t, before: "n"), "dˈos mˈil doθjˈentos")
    }

    // MARK: - Text

    private func scalars(_ s: String) -> [Unicode.Scalar] { Array(s.unicodeScalars) }

    func testPrepare() {
        func prepared(_ text: String, _ language: ParadeeLanguage) -> String {
            ParadeeG2P.string(
                ParadeeG2P.prepare(text, language: language, percent: ParadeeG2P.percentWords[language] ?? ""))
        }
        // Devanagari digits, danda followed by text, percent
        XCTAssertEqual(prepared("३ लोग।कल 50% सस्ता", .hindi), "3 लोग। कल 50 प्रतिशत सस्ता")
        // guillemets, zero-width space, French digit groups (space, no-break and narrow no-break), " %"
        XCTAssertEqual(
            prepared("« Bon\u{200B}jour » 1 250 et 1\u{00A0}000\u{202F}000, soit 15 %", .french),
            "“ Bonjour ” 1250 et 1000000, soit 15 pour cent")
        // groups only join before exactly three digits; only French joins them
        XCTAssertEqual(prepared("12 3456 et 7 89", .french), "12 3456 et 7 89")
        XCTAssertEqual(prepared("1 250", .spanish), "1 250")
    }

    func testWordsAndContexts() {
        // upstream: _words("les amis, «bonjour» — ok") and _contexts of it
        let pieces = ParadeeG2P.splitWhitespace(scalars("les amis, «bonjour» — ok"))
        let words = pieces.map(ParadeeG2P.word)
        XCTAssertEqual(words.map { $0.map { ParadeeG2P.string($0.core) } }, ["les", "amis", "bonjour", nil, "ok"])
        XCTAssertEqual(words.map { $0.map(\.lead) }, [false, false, true, nil, false])
        XCTAssertEqual(words.map { $0.map(\.trail) }, [false, true, true, nil, false])
        let contexts = ParadeeG2P.contexts(words)
        XCTAssertEqual(contexts.map { $0?.before }, ["#", "s", "#", nil, "#"])
        XCTAssertEqual(contexts.map { $0?.after }, ["a", "#", "#", nil, "#"])
    }

    func testContextLettersAreLowercased() {
        let words = ParadeeG2P.splitWhitespace(scalars("Les Amis")).map(ParadeeG2P.word)
        XCTAssertEqual(ParadeeG2P.contexts(words).map { $0?.after }, ["a", "#"])
    }

    func testDigitRunsAndValues() {
        let runs = ParadeeG2P.digitRuns(scalars("n°81ème"))
        XCTAssertEqual(runs.map { ParadeeG2P.string($0.scalars) }, ["n°", "81", "ème"])
        XCTAssertEqual(runs.map(\.isDigits), [false, true, false])
        XCTAssertEqual(ParadeeG2P.value(scalars("1250")), 1250)
        // Devanagari digits are decimal digits too
        XCTAssertEqual(ParadeeG2P.value(scalars("२५")), 25)
    }

    func testCharacterClassesFollowPythonStr() {
        XCTAssertTrue(ParadeeG2P.isEdge("«"))
        XCTAssertTrue(ParadeeG2P.isEdge("€"))
        XCTAssertTrue(ParadeeG2P.isEdge("\u{200D}"))
        XCTAssertTrue(ParadeeG2P.isEdge("'"))
        XCTAssertFalse(ParadeeG2P.isEdge("é"))
        XCTAssertTrue(ParadeeG2P.isEdge("।"))
        // A Devanagari vowel sign is not alphanumeric for Python, the consonant is.
        XCTAssertFalse(ParadeeG2P.isAlphanumeric("ा"))
        XCTAssertTrue(ParadeeG2P.isAlphanumeric("क"))
        XCTAssertTrue(ParadeeG2P.isAlphanumeric("½"))
        XCTAssertFalse(ParadeeG2P.isDecimal("½"))
        XCTAssertTrue(ParadeeG2P.isSpace("\u{202F}"))
        XCTAssertTrue(ParadeeG2P.isSpace("\u{1F}"))
    }

    // MARK: - Model rows and CTC

    private let data = ParadeeG2PData(
        inputs: ["a": 9, "b": 10, "é": 11],
        specials: [
            "<pad>": 0, "<s1>": 1, "<s2>": 2, "<hi>": 3, "<fr>": 4, "<es>": 5, "<en-us>": 6, "<en-gb>": 7, "<unk>": 8,
        ],
        outputs: ["<blank>", "ˈ", "a", "b"], numbers: [:], lexicon: [:], englishLowercased: [:])

    func testRowLayout() {
        // <lang> before <s1> lowercased letters <s2> after; unknown symbols are <unk>
        XCTAssertEqual(data.row(.french, "#", "AbÉx", "b"), [4, 8, 1, 9, 10, 11, 8, 2, 10])
        XCTAssertEqual(data.row(.spanish, "a", "ab", "#"), [5, 9, 1, 9, 10, 2, 8])
        // a context that lowercases to more than one scalar is <unk>
        XCTAssertEqual(data.row(.hindi, "i\u{307}", "a", "#").prefix(2), [3, 8])
        let long = String(repeating: "a", count: 70)
        XCTAssertEqual(data.row(.french, "#", long, "#").count, ParadeeG2P.maxWordLength + 5)
    }

    func testCtcDecodeMergesRepeatsAndDropsBlanks() {
        XCTAssertEqual(data.decode([0, 3, 3, 0, 3, 1, 1, 2, 0]), "bbˈa")
        XCTAssertEqual(data.decode([0, 0, 0]), "")
    }
}
