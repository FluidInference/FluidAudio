import Foundation

/// Catalog of Kokoro voice options with metadata and identifier normalization helpers.
@available(macOS 13.0, iOS 16.0, *)
public enum KokoroVoiceCatalog {

    /// Describes a single Kokoro voice option.
    public struct VoiceOption: Hashable, Sendable {
        public enum Gender: String, Sendable {
            case female
            case male
            case neutral
        }

        public let id: String
        public let label: String
        public let languageCode: String
        public let languageName: String
        public let gender: Gender

        public init(id: String, label: String, languageCode: String, languageName: String, gender: Gender) {
            self.id = id
            self.label = label
            self.languageCode = languageCode
            self.languageName = languageName
            self.gender = gender
        }
    }

    /// Default Kokoro voice identifier.
    public static let defaultVoiceId = "af_heart"

    /// Ordered list of all supported voice options.
    public static let allVoices: [VoiceOption] = [
        VoiceOption(
            id: "af_alloy", label: "Alloy", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "af_aoede", label: "Aoede", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "af_bella", label: "Bella", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "af_heart", label: "Heart", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "af_jessica", label: "Jessica", languageCode: "en-US", languageName: "American English", gender: .female
        ),
        VoiceOption(
            id: "af_kore", label: "Kore", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "af_nicole", label: "Nicole", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "af_nova", label: "Nova", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "af_river", label: "River", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "af_sarah", label: "Sarah", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "af_sky", label: "Sky", languageCode: "en-US", languageName: "American English", gender: .female),
        VoiceOption(
            id: "am_adam", label: "Adam", languageCode: "en-US", languageName: "American English", gender: .male),
        VoiceOption(
            id: "am_echo", label: "Echo", languageCode: "en-US", languageName: "American English", gender: .male),
        VoiceOption(
            id: "am_eric", label: "Eric", languageCode: "en-US", languageName: "American English", gender: .male),
        VoiceOption(
            id: "am_fenrir", label: "Fenrir", languageCode: "en-US", languageName: "American English", gender: .male),
        VoiceOption(
            id: "am_liam", label: "Liam", languageCode: "en-US", languageName: "American English", gender: .male),
        VoiceOption(
            id: "am_michael", label: "Michael", languageCode: "en-US", languageName: "American English", gender: .male),
        VoiceOption(
            id: "am_onyx", label: "Onyx", languageCode: "en-US", languageName: "American English", gender: .male),
        VoiceOption(
            id: "am_puck", label: "Puck", languageCode: "en-US", languageName: "American English", gender: .male),
        VoiceOption(
            id: "am_santa", label: "Santa", languageCode: "en-US", languageName: "American English", gender: .male),
        VoiceOption(
            id: "bf_alice", label: "Alice", languageCode: "en-GB", languageName: "British English", gender: .female),
        VoiceOption(
            id: "bf_emma", label: "Emma", languageCode: "en-GB", languageName: "British English", gender: .female),
        VoiceOption(
            id: "bf_isabella", label: "Isabella", languageCode: "en-GB", languageName: "British English",
            gender: .female),
        VoiceOption(
            id: "bf_lily", label: "Lily", languageCode: "en-GB", languageName: "British English", gender: .female),
        VoiceOption(
            id: "bm_daniel", label: "Daniel", languageCode: "en-GB", languageName: "British English", gender: .male),
        VoiceOption(
            id: "bm_fable", label: "Fable", languageCode: "en-GB", languageName: "British English", gender: .male),
        VoiceOption(
            id: "bm_george", label: "George", languageCode: "en-GB", languageName: "British English", gender: .male),
        VoiceOption(
            id: "bm_lewis", label: "Lewis", languageCode: "en-GB", languageName: "British English", gender: .male),
        VoiceOption(id: "ef_dora", label: "Dora", languageCode: "es-ES", languageName: "Spanish", gender: .female),
        VoiceOption(id: "em_alex", label: "Alex", languageCode: "es-ES", languageName: "Spanish", gender: .male),
        VoiceOption(id: "em_santa", label: "Santa", languageCode: "es-ES", languageName: "Spanish", gender: .male),
        VoiceOption(id: "ff_siwis", label: "Siwis", languageCode: "fr-FR", languageName: "French", gender: .female),
        VoiceOption(id: "hf_alpha", label: "Alpha", languageCode: "hi-IN", languageName: "Hindi", gender: .female),
        VoiceOption(id: "hf_beta", label: "Beta", languageCode: "hi-IN", languageName: "Hindi", gender: .female),
        VoiceOption(id: "hm_omega", label: "Omega", languageCode: "hi-IN", languageName: "Hindi", gender: .male),
        VoiceOption(id: "hm_psi", label: "Psi", languageCode: "hi-IN", languageName: "Hindi", gender: .male),
        VoiceOption(id: "if_sara", label: "Sara", languageCode: "it-IT", languageName: "Italian", gender: .female),
        VoiceOption(id: "im_nicola", label: "Nicola", languageCode: "it-IT", languageName: "Italian", gender: .male),
        VoiceOption(id: "jf_alpha", label: "Alpha", languageCode: "ja-JP", languageName: "Japanese", gender: .female),
        VoiceOption(
            id: "jf_gongitsune", label: "Gongitsune", languageCode: "ja-JP", languageName: "Japanese", gender: .female),
        VoiceOption(id: "jf_nezumi", label: "Nezumi", languageCode: "ja-JP", languageName: "Japanese", gender: .female),
        VoiceOption(
            id: "jf_tebukuro", label: "Tebukuro", languageCode: "ja-JP", languageName: "Japanese", gender: .female),
        VoiceOption(id: "jm_kumo", label: "Kumo", languageCode: "ja-JP", languageName: "Japanese", gender: .male),
        VoiceOption(
            id: "pf_dora", label: "Dora", languageCode: "pt-BR", languageName: "Portuguese (Brazil)", gender: .female),
        VoiceOption(
            id: "pm_alex", label: "Alex", languageCode: "pt-BR", languageName: "Portuguese (Brazil)", gender: .male),
        VoiceOption(
            id: "pm_santa", label: "Santa", languageCode: "pt-BR", languageName: "Portuguese (Brazil)", gender: .male),
        VoiceOption(
            id: "zf_xiaobei", label: "Xiaobei", languageCode: "zh-CN", languageName: "Mandarin Chinese", gender: .female
        ),
        VoiceOption(
            id: "zf_xiaoni", label: "Xiaoni", languageCode: "zh-CN", languageName: "Mandarin Chinese", gender: .female),
        VoiceOption(
            id: "zf_xiaoxiao", label: "Xiaoxiao", languageCode: "zh-CN", languageName: "Mandarin Chinese",
            gender: .female),
        VoiceOption(
            id: "zf_xiaoyi", label: "Xiaoyi", languageCode: "zh-CN", languageName: "Mandarin Chinese", gender: .female),
        VoiceOption(
            id: "zm_yunjian", label: "Yunjian", languageCode: "zh-CN", languageName: "Mandarin Chinese", gender: .male),
        VoiceOption(
            id: "zm_yunxi", label: "Yunxi", languageCode: "zh-CN", languageName: "Mandarin Chinese", gender: .male),
        VoiceOption(
            id: "zm_yunxia", label: "Yunxia", languageCode: "zh-CN", languageName: "Mandarin Chinese", gender: .male),
        VoiceOption(
            id: "zm_yunyang", label: "Yunyang", languageCode: "zh-CN", languageName: "Mandarin Chinese", gender: .male),
    ]

    /// Lookup of voices by canonical identifier.
    private static let voicesById: [String: VoiceOption] = {
        var dict: [String: VoiceOption] = [:]
        for voice in allVoices {
            dict[voice.id] = voice
        }
        return dict
    }()

    /// Additional alias overrides for backwards compatibility.
    private static let aliasOverrides: [String: String] = [
        "hf_omega": "hm_omega",
        "hfpsi": "hm_psi",
        "hf_psi": "hm_psi",
    ]

    /// Normalizes a user-specified voice identifier to the canonical snake_case form.
    private static func normalizedKey(for identifier: String) -> String {
        let trimmed = identifier.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return trimmed }

        var working = trimmed
        if !working.contains("_") {
            working = camelCaseToSnake(working)
        }
        working =
            working
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: " ", with: "_")
        working = working.lowercased()
        return working
    }

    /// Converts camelCase identifiers to snake_case.
    private static func camelCaseToSnake(_ value: String) -> String {
        guard !value.isEmpty else { return value }
        var scalars: [Character] = []
        scalars.reserveCapacity(value.count * 2)
        for character in value {
            if character.isUppercase {
                if !scalars.isEmpty && scalars.last != "_" {
                    scalars.append("_")
                }
                scalars.append(Character(character.lowercased()))
            } else {
                scalars.append(character)
            }
        }
        return String(scalars)
    }

    /// Builds an index of known aliases to canonical identifiers.
    private static let aliasIndex: [String: String] = {
        var map: [String: String] = [:]
        for option in allVoices {
            let id = option.id
            map[id] = id
            map[id.replacingOccurrences(of: "_", with: "")] = id
            map[id.replacingOccurrences(of: "_", with: "-")] = id
            map[id.replacingOccurrences(of: "_", with: " ")] = id
            let camel = snakeToCamel(id)
            map[camel] = id
            map[camel.lowercased()] = id
        }
        for (alias, canonical) in aliasOverrides {
            map[alias] = canonical
        }
        return map
    }()

    /// Converts snake_case voice identifiers to lowerCamelCase.
    private static func snakeToCamel(_ value: String) -> String {
        guard !value.isEmpty else { return value }
        var result = ""
        var makeUpper = false
        for character in value {
            if character == "_" {
                makeUpper = true
                continue
            }
            if makeUpper {
                result.append(Character(String(character).uppercased()))
                makeUpper = false
            } else {
                result.append(character)
            }
        }
        return result
    }

    /// Returns the canonical voice identifier for a supplied name or alias.
    public static func canonicalVoiceId(for identifier: String) -> String? {
        let normalized = normalizedKey(for: identifier)
        guard !normalized.isEmpty else { return nil }
        if let override = aliasOverrides[normalized] { return override }
        if let exact = voicesById[normalized]?.id { return exact }
        if let alias = aliasIndex[normalized] { return alias }
        return nil
    }

    /// Retrieves metadata for the supplied voice identifier.
    public static func voice(for identifier: String) -> VoiceOption? {
        guard let canonical = canonicalVoiceId(for: identifier) else { return nil }
        return voicesById[canonical]
    }

    /// Group voices by BCP-47 language code.
    public static func voices(forLanguage languageCode: String) -> [VoiceOption] {
        let normalized = languageCode.lowercased()
        return allVoices.filter { $0.languageCode.lowercased() == normalized }
    }

    /// All supported language codes.
    public static var supportedLanguageCodes: [String] {
        Array(Set(allVoices.map { $0.languageCode })).sorted()
    }
}
