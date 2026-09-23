import XCTest

@testable import FluidAudio

/// French frontend for the Kokoro ANE `.french` variant (#926). Lexicon rows
/// are verbatim `fr_lexicon_cache.json` entries (espeak-ng `fr-fr` forms);
/// expected strings are espeak-ng output after Misaki's `EspeakG2P`
/// post-processing.
final class FrenchG2PTests: XCTestCase {

    private let lexicon = KokoroAneLexicon(
        entries: [
            "arrivés": "aʁivˈe", "au": "o", "autres": "ˈotʁ", "aux": "o", "comme": "kˈɔm",
            "continents": "kɔ̃tinˈɑ̃", "de": "də", "des": "de", "elle": "ˈɛl", "est": "ˈɛ", "et": "ˈe",
            "fine": "fˈin", "hommage": "ɔmˈaʒ", "héros": "eʁˈo", "itinéraire": "itineʁˈɛʁ", "les": "lˈe",
            "luna": "lynˈa", "lutteurs": "lytˈœʁ", "mers": "mˈɛʁ", "niveau": "nivˈo", "ont": "ˈɔ̃",
            "pensez": "pɑ̃sˈe", "plus": "plˈy", "randonnée": "ʁɑ̃dɔnˈe", "rendu": "ʁɑ̃dˈy",
            "similaire": "similˈɛʁ", "ski": "skˈi", "sont": "sˈɔ̃", "sous": "sˈu", "un": "ˈœ̃", "unis": "ynˈi",
            "à": "ˈa", "également": "eɡalmˈɑ̃", "épaisse": "epˈɛs", "états": "etˈa",
            "ans": "ˈɑ̃", "sept": "sˈɛt", "demain": "dəmˈɛ̃", "île": "ˈil",
        ],
        hAspire: ["héros"])

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

    func testLiaisonDoesNotDoubleASpokenConsonant() {
        XCTAssertEqual(phonemize("six ans"), "sˈiz ˈɑ̃")  // voiced, not sisz
        XCTAssertEqual(phonemize("sept ans"), "sˈɛt ˈɑ̃")  // not sɛtt
        XCTAssertEqual(phonemize("les ans"), "lez ˈɑ̃")  // plain liaison unchanged
        // espeak still links plurals in -es and -ent verbs after a spoken consonant.
        XCTAssertEqual(FrenchPhonology.applyLiaison("classes", "klˈas"), "klˈasz")
        XCTAssertEqual(FrenchPhonology.applyLiaison("mettent", "mˈɛt"), "mˈɛtt")
        XCTAssertEqual(FrenchPhonology.applyLiaison("bus", "bˈys"), "bˈys")
        XCTAssertEqual(FrenchPhonology.applyLiaison("neuf", "nˈœf"), "nˈœv")
    }

    func testLoneAccentedCapitalIsAWord() {
        XCTAssertFalse(FrenchPhonology.isSpelledAcronym("À"))
        XCTAssertEqual(phonemize("À demain."), "a dəmˈɛ̃.")
    }

    func testApostropheCompoundResolvesThroughParts() async throws {
        let g2p = FrenchG2P(lexicon: lexicon) { word in word == "presqu" ? "pʁɛsk" : nil }
        let result = try await g2p.phonemize("une presqu'île")
        XCTAssertTrue(result.hasSuffix("pʁɛskˈil"), result)
    }

    func testAcronymsWithoutVowelsAreSpelled() {
        XCTAssertTrue(FrenchPhonology.isSpelledAcronym("SNCF"))
        XCTAssertFalse(FrenchPhonology.isSpelledAcronym("ONU"))
        XCTAssertEqual(FrenchPhonology.spell("sncf"), "ˌɛsˌɛnsˌeˈɛf")
    }

    func testLexiconCacheSchemaLoads() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("fr_lexicon_cache_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let json =
            #"{"lower":{"bɔ̃":["x"],"héros":["e","ʁ","ˈ","o"]},"caseSensitive":{"Paris":["p","a","ʁ","ˈ","i"]},"hAspire":["héros"]}"#
        try json.write(to: url, atomically: true, encoding: .utf8)
        let lex = try KokoroAneLexicon(contentsOf: url)
        XCTAssertEqual(lex.lookup("héros"), "eʁˈo")
        XCTAssertEqual(lex.lookup("Héros"), "eʁˈo")  // falls back to lowercase
        XCTAssertEqual(lex.lookup("Paris"), "paʁˈi")
        XCTAssertTrue(lex.isHAspire("héros"))
        XCTAssertFalse(lex.isHAspire("bɔ̃"))
        XCTAssertNil(lex.lookup("absent"))
    }

    func testLexiconStressFollowsWordClass() {
        let lookup = { (w: String) in FrenchPhonology.lookup(w, lexicon: self.lexicon) }
        XCTAssertEqual(FrenchPhonology.wordPhonemes("elle", stress: .primary, lookup: lookup).phonemes, "ˈɛl")
        XCTAssertEqual(FrenchPhonology.wordPhonemes("elle", stress: .none, lookup: lookup).phonemes, "ɛl")
        XCTAssertEqual(FrenchPhonology.wordPhonemes("plus", stress: .secondary, lookup: lookup).phonemes, "plˌy")
        // Citation overrides beat the lexicon (running-text "est" is ɛ).
        XCTAssertEqual(FrenchPhonology.lookup("est", lexicon: lexicon), .raw("ɛ"))
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
