import XCTest

@testable import FluidAudio

/// French frontend for the Kokoro ANE `.french` variant (#926). Lexicon rows
/// are verbatim ipa-dict `fr_FR` entries; expected strings are espeak-ng
/// `fr-fr` output after Misaki's `EspeakG2P` post-processing.
final class FrenchG2PTests: XCTestCase {

    private let lexicon = FrenchLexicon(
        tsv: [
            "arrivés\taʁive", "au\to", "autres\totʁ", "aux\to", "comme\tkɔm", "continents\tkɔ̃tinɑ̃", "de\tdə",
            "des\tde", "elle\tɛl", "est\tɛst", "et\te", "fine\tfin", "hommage\tɔmaʒ", "héros\tʼeʁo",
            "itinéraire\titineʁɛʁ", "les\tle", "luna\tlyna", "lutteurs\tlytœʁ", "mers\tmɛʁ", "niveau\tnivo",
            "ont\tɔ̃", "pensez\tpɑ̃se", "plus\tply", "randonnée\tʁɑ̃dɔne", "rendu\tʁɑ̃dy", "similaire\tsimilɛʁ",
            "ski\tski", "sont\tsɔ̃", "sous\tsu", "un\tœ̃", "unis\tyni", "à\ta", "également\tegalmɑ̃",
            "épaisse\tepɛs", "états\teta", "états-unis\tetazyni",
        ].joined(separator: "\n"))

    private func phonemize(_ text: String) -> String {
        FrenchPhonology.phonemize(text, isLexiconEntry: lexicon.contains) {
            FrenchPhonology.lookup($0, lexicon: lexicon)
        }
    }

    func testSentencesMatchEspeak() {
        let cases: [(String, String)] = [
            (
                "Les autres lutteurs ont également rendu hommage à Luna.",
                "lez otʁ lytˈœʁz ˈɔ̃t eɡalmˈɑ̃ ʁɑ̃dˈy ɔmˈaʒ a lynˈa."
            ),
            (
                "Pensez à l'itinéraire de ski comme à un itinéraire de randonnée similaire.",
                "pɑ̃sˈez a litineʁˈɛʁ də skˈi kɔm a œ̃n itineʁˈɛʁ də ʁɑ̃dɔnˈe similˈɛʁ."
            ),
            (
                "Elle est plus fine au niveau des mers et plus épaisse sous les continents.",
                "ɛl ɛ ply fˈin o nivˈo de mˈɛʁz e plyz epˈɛs su le kɔ̃tinˈɑ̃."
            ),
        ]
        for (text, expected) in cases {
            XCTAssertEqual(phonemize(text), expected, text)
        }
    }

    func testHAspireBlocksLiaisonAndCompoundsStressEachPart() {
        let result = phonemize("Les héros sont arrivés aux États-Unis.")
        XCTAssertTrue(result.hasPrefix("le eʁˈo "), result)  // no z before h aspiré
        XCTAssertTrue(result.hasSuffix("oz etˈazynˈi."), result)
    }

    func testEspeakVowelConventions() {
        // ɔ closes to o in open non-final syllables; ɥ → y; ɲ → nj; g → ɡ.
        XCTAssertEqual(FrenchPhonology.mapToEspeak("pʁɔʒɛ", word: "projet"), "pʁoʒɛ")
        XCTAssertEqual(FrenchPhonology.mapToEspeak("lɥi", word: "lui"), "lyi")
        XCTAssertEqual(FrenchPhonology.mapToEspeak("mɔ̃taɲ", word: "montagnes"), "mɔ̃tanj")
        XCTAssertEqual(FrenchPhonology.mapToEspeak("ɡʁɑ̃d", word: "grande"), "ɡʁɑ̃d")
        // Schwa between single consonants after a vowel drops (devenu).
        XCTAssertEqual(FrenchPhonology.mapToEspeak("dəvəny", word: "devenu"), "dəvny")
    }

    func testStressSkipsFinalSchwaAndKeepsNasalVowelWhole() {
        XCTAssertEqual(FrenchPhonology.addStress("pʁɛskə", .primary), "pʁˈɛskə")
        XCTAssertEqual(FrenchPhonology.addStress("ʁɑ̃dy", .primary), "ʁɑ̃dˈy")
        XCTAssertEqual(FrenchPhonology.addStress("mɔ̃", .primary), "mˈɔ̃")
    }

    func testAcronymsWithoutVowelsAreSpelled() {
        XCTAssertTrue(FrenchPhonology.isSpelledAcronym("SNCF"))
        XCTAssertFalse(FrenchPhonology.isSpelledAcronym("ONU"))
        XCTAssertEqual(FrenchPhonology.spell("sncf"), "ˌɛsˌɛnsˌeˈɛf")
    }

    func testLexiconBinarySearchHandlesUnsortedInput() {
        let lex = FrenchLexicon(tsv: "zèbre\tzɛbʁ\nabricot\tabʁiko\nélan\telɑ̃\nbateau\tbato")
        XCTAssertEqual(lex.count, 4)
        XCTAssertEqual(lex.lookup("abricot"), "abʁiko")
        XCTAssertEqual(lex.lookup("élan"), "elɑ̃")
        XCTAssertEqual(lex.lookup("zèbre"), "zɛbʁ")
        XCTAssertNil(lex.lookup("bate"))
        XCTAssertNil(lex.lookup(""))
    }

    func testFallbackRunsOncePerUnknownWord() async throws {
        let calls = CallCounter()
        let g2p = FrenchG2P(lexicon: lexicon) { word in
            await calls.record(word)
            return word == "zorglub" ? "zɔʁɡlyb" : nil
        }
        let first = try await g2p.phonemize("Luna et zorglub.")
        let second = try await g2p.phonemize("Zorglub est là.")
        XCTAssertEqual(first, "lynˈa e zɔʁɡlˈyb.")
        XCTAssertTrue(second.hasPrefix("zɔʁɡlˈyb"), second)
        let recorded = await calls.words
        XCTAssertEqual(recorded.filter { $0 == "zorglub" }.count, 1)
    }
}

private actor CallCounter {
    private(set) var words: [String] = []
    func record(_ word: String) { words.append(word) }
}
