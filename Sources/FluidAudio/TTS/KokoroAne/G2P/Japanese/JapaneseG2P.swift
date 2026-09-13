import Foundation

#if canImport(voicevox_core)
import voicevox_core
#endif

/// Context-aware Japanese text frontend for the Kokoro ANE Japanese variant.
///
/// OpenJTalk resolves kana readings for kanji and compounds. The resolved
/// moras are then converted to the IPA alphabet used to train Kokoro's
/// Japanese voices. The runtime and dictionary are loaded lazily; callers who
/// use another Kokoro variant do not download the dictionary.
actor JapaneseG2P {
    private struct Mora: Decodable {
        let text: String
    }

    private struct AccentPhrase: Decodable {
        let moras: [Mora]
    }

    #if canImport(voicevox_core)
    /// Stored as an address-sized integer so the actor's nonisolated deinit
    /// only reads Sendable state; all live-handle use remains actor-isolated.
    private let handleAddress: UInt
    #endif

    static var isAvailable: Bool {
        #if canImport(voicevox_core)
        true
        #else
        false
        #endif
    }

    init(dictionaryURL: URL) throws {
        #if canImport(voicevox_core)
        var pendingHandle: OpaquePointer?
        let result = dictionaryURL.path.withCString {
            voicevox_open_jtalk_rc_new($0, &pendingHandle)
        }
        guard result == 0, let pendingHandle else {
            throw KokoroAneError.inputProcessingFailed(
                "OpenJTalk could not load its dictionary at \(dictionaryURL.path) "
                    + "(VOICEVOX result \(result)).")
        }
        handleAddress = UInt(bitPattern: pendingHandle)
        #else
        throw KokoroAneError.inputProcessingFailed(
            "Japanese text processing is unavailable because the "
                + "JapaneseTextProcessing package trait is disabled.")
        #endif
    }

    deinit {
        #if canImport(voicevox_core)
        voicevox_open_jtalk_rc_delete(OpaquePointer(bitPattern: handleAddress))
        #endif
    }

    func phonemize(_ text: String) throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("Japanese G2P received empty text.")
        }

        var output = ""
        var spokenText = ""

        func flushSpokenText() throws {
            guard !spokenText.isEmpty else { return }
            let spoken = try analyze(spokenText)
            guard !spoken.isEmpty else {
                throw KokoroAneError.inputProcessingFailed(
                    "OpenJTalk produced no pronunciation for '\(spokenText)'.")
            }
            appendSpoken(spoken, to: &output)
            spokenText.removeAll(keepingCapacity: true)
        }

        for character in trimmed {
            if character.isWhitespace {
                try flushSpokenText()
                appendSpace(to: &output)
                continue
            }
            if let punctuation = Self.punctuationMap[character] {
                try flushSpokenText()
                appendPunctuation(punctuation, to: &output)
                continue
            }
            // The object particle を is a word boundary the analysis cannot
            // express: VOICEVOX renders it as the mora オ, and after an
            // o-ending word that is indistinguishable from a long vowel
            // (日本語を → ɲihoŋɡoː instead of ɲihoŋɡo o). を is not part of
            // any modern word, so analyzing it with what follows is safe.
            if character == "を" {
                try flushSpokenText()
            }
            spokenText.append(character)
        }
        try flushSpokenText()

        let result = output.trimmingCharacters(in: .whitespaces)
        guard !result.isEmpty else {
            throw KokoroAneError.inputProcessingFailed(
                "Japanese G2P produced no phonemes for '\(text)'.")
        }
        return result
    }

    private func analyze(_ text: String) throws -> String {
        #if canImport(voicevox_core)
        guard let handle = OpaquePointer(bitPattern: handleAddress) else {
            throw KokoroAneError.inputProcessingFailed("OpenJTalk is not initialized.")
        }

        var jsonPointer: UnsafeMutablePointer<CChar>?
        let result = text.withCString {
            voicevox_open_jtalk_rc_analyze(handle, $0, &jsonPointer)
        }
        guard result == 0, let jsonPointer else {
            throw KokoroAneError.inputProcessingFailed(
                "OpenJTalk analysis failed (VOICEVOX result \(result)).")
        }
        defer { voicevox_json_free(jsonPointer) }

        guard let data = String(validatingCString: jsonPointer)?.data(using: .utf8) else {
            throw KokoroAneError.inputProcessingFailed(
                "OpenJTalk returned invalid UTF-8 analysis data.")
        }
        let phrases: [AccentPhrase]
        do {
            phrases = try JSONDecoder().decode([AccentPhrase].self, from: data)
        } catch {
            throw KokoroAneError.inputProcessingFailed(
                "OpenJTalk returned malformed analysis data: \(error.localizedDescription)")
        }

        var converted: [String] = []
        converted.reserveCapacity(phrases.count)
        for phrase in phrases where !phrase.moras.isEmpty {
            let phonemes = try JapaneseMoraMapper.phonemize(phrase.moras.map(\.text))
            if !phonemes.isEmpty { converted.append(phonemes) }
        }
        return converted.joined(separator: " ")
        #else
        throw KokoroAneError.inputProcessingFailed(
            "Japanese text processing is unavailable because the "
                + "JapaneseTextProcessing package trait is disabled.")
        #endif
    }

    private func appendSpoken(_ spoken: String, to output: inout String) {
        if let last = output.last, !last.isWhitespace, !Self.openingPunctuation.contains(last) {
            output.append(" ")
        }
        output.append(spoken)
    }

    private func appendSpace(to output: inout String) {
        guard !output.isEmpty, output.last?.isWhitespace != true else { return }
        output.append(" ")
    }

    private func appendPunctuation(_ punctuation: Character, to output: inout String) {
        if Self.openingPunctuation.contains(punctuation) {
            appendSpace(to: &output)
            output.append(punctuation)
            return
        }
        while output.last?.isWhitespace == true { output.removeLast() }
        output.append(punctuation)
        if Self.stoppingPunctuation.contains(punctuation) {
            output.append(" ")
        }
    }

    private static let openingPunctuation: Set<Character> = ["(", "“"]
    private static let stoppingPunctuation: Set<Character> = ["!", ")", ",", ".", ":", ";", "?", "”"]

    private static let punctuationMap: [Character: Character] = [
        "!": "!", "\"": "”", "(": "(", ")": ")", ",": ",", ".": ".", ":": ":", ";": ";", "?": "?",
        "—": "—", "“": "“", "”": "”", "…": "…", "、": ",", "。": ".", "〈": "“", "〉": "”", "《": "“",
        "》": "”", "「": "“", "」": "”", "『": "“", "』": "”", "【": "“", "】": "”", "！": "!", "（": "(",
        "）": ")", "：": ":", "；": ";", "？": "?", "〜": "—", "～": "—", "・": " ",
    ]
}
