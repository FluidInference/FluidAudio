import XCTest

@testable import FluidAudio

/// Spanish frontend for the Kokoro ANE `.spanish` variant (#926). Expected
/// strings are espeak-ng `es` output after Misaki's `EspeakG2P`
/// post-processing, which is what the Spanish voices were trained on.
final class SpanishG2PTests: XCTestCase {

    func testSentencesMatchEspeak() {
        let cases: [(String, String)] = [
            (
                "Por ello, cuando se inventó, el lápiz ganó muchos amigos.",
                "poɾ ˈeʎo, kwˌando se ˌimbentˈo, el lˈapiθ ɣanˈo mˈuʧos amˈiɣos."
            ),
            (
                "Estas parejas podrían elegir planificar la adopción de un bebé.",
                "ˈestas paɾˈexas poðɾˈian ˌelexˈiɾ plˌanifikˈaɾ la ˌaðopθjˈon de ˈum beβˈe."
            ),
            (
                "¿Dónde está el capítulo? Llegó al río con mucho tiempo.",
                "¿dˈonde estˈa el kapˈitulo? ʎeɣˈo al rˈio kon mˈuʧo tjˈempo."
            ),
            ("Hoy es muy fácil, generalmente.", "ˈoɪ ˈes mˈuj fˈaθil, xˌeneɾˈalmˈente."),
        ]
        for (text, expected) in cases {
            XCTAssertEqual(SpanishG2P.phonemize(text), expected, text)
        }
    }

    func testStressFollowsSpellingRules() {
        XCTAssertEqual(SpanishG2P.phonemizeWord("casa", stress: .primary), "kˈasa")  // vowel-final: penult
        XCTAssertEqual(SpanishG2P.phonemizeWord("papel", stress: .primary), "papˈel")  // consonant-final: final
        XCTAssertEqual(SpanishG2P.phonemizeWord("árbol", stress: .primary), "ˈaɾbol")  // written accent wins
        // Stress sits before the vowel, after onset consonants and glides.
        XCTAssertEqual(SpanishG2P.phonemizeWord("tiempo", stress: .primary), "tjˈempo")
    }

    func testDiphthongLigaturesAndHiatus() {
        XCTAssertEqual(SpanishG2P.phonemizeWord("causa", stress: .primary), "kˈWsa")
        XCTAssertEqual(SpanishG2P.phonemizeWord("rey", stress: .primary), "rˈA")
        XCTAssertEqual(SpanishG2P.phonemizeWord("cliente", stress: .primary), "kliˈɛnte")
        XCTAssertEqual(SpanishG2P.phonemizeWord("prohibido", stress: .primary), "pɾˌoibˈido")
    }

    func testLenitionDependsOnPhraseContext() {
        // b/d/g are stops after a pause or nasal and lenite elsewhere.
        XCTAssertEqual(SpanishG2P.phonemize("la vida de los gatos"), "la βˈiða ðe los ɣˈatos")
        XCTAssertEqual(SpanishG2P.phonemize("un vaso, con dos gatos"), "ˈum bˈaso, kon dˈos ɣˈatos")
    }

    func testFunctionWordRegainsStressBeforePause() {
        XCTAssertEqual(SpanishG2P.phonemize("tengo que."), "tˈɛŋɡo kˈe.")
    }

    func testAcronymsAreSpelled() {
        XCTAssertEqual(SpanishG2P.phonemize("la ONU"), "la ˌoˌeneˈu")
    }

    func testDigitsAreDroppedNotSpelled() {
        // Numbers are verbalized by NeMo TN upstream; stray digits vanish.
        XCTAssertEqual(SpanishG2P.phonemize("casa 12"), "kˈasa")
    }
}
