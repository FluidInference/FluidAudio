#if TTS
import Foundation

/// Pinyin syllable → Bopomofo + tone digit string, matching
/// `kokoro-82m-coreml/ANE-zh/vocab.json`.
///
/// Direct port of `misaki/zh_frontend.py:ZH_MAP`. Each pinyin syllable
/// `<initial><final><tone>` (tone in 1…5, optional) decomposes into:
///
///   1. **Initial** — one of the 21 pinyin initials (b, p, m, f, d, t,
///      n, l, g, k, h, zh, ch, sh, r, z, c, s, j, q, x). May be empty
///      for finals like `er`, `an`, `ou` etc.
///   2. **Final** — one of the canonical compound finals listed in
///      `finalMap`. The empty-initial variants (`yi` → `i`, `wu` → `u`,
///      `yu` → `v`, `yue` → `ve` …) are normalised first by
///      `normalizePinyinForLookup`.
///   3. **Tone digit** — `1`-`5`, appended verbatim. Tone `5` (neutral)
///      remains as `5` since the v1.1-zh vocab includes it explicitly.
///
/// Special pinyin spelling rules `pypinyin` does *not* normalize for us
/// but the kokoro vocab requires (mirrors misaki):
///
///   * `zi/ci/si`     → final = `ii`  (vocab: `ㄭ`)
///   * `zhi/chi/shi/ri` → final = `iii` (vocab: `十`)
///
/// Punctuation falls through unchanged — the kokoro vocab keeps ASCII
/// `.,!?;:` so they tokenise directly.
public enum MandarinBopomofoMap {

    /// Initials (multi-char first so `zh`/`ch`/`sh` win over `z`/`c`/`s`/
    /// `h` during longest-prefix matching).
    static let initials: [String] = [
        "zh", "ch", "sh",
        "b", "p", "m", "f",
        "d", "t", "n", "l",
        "g", "k", "h",
        "j", "q", "x",
        "r", "z", "c", "s",
    ]

    /// Initial → bopomofo character.
    static let initialMap: [String: String] = [
        "b": "ㄅ", "p": "ㄆ", "m": "ㄇ", "f": "ㄈ",
        "d": "ㄉ", "t": "ㄊ", "n": "ㄋ", "l": "ㄌ",
        "g": "ㄍ", "k": "ㄎ", "h": "ㄏ",
        "j": "ㄐ", "q": "ㄑ", "x": "ㄒ",
        "zh": "ㄓ", "ch": "ㄔ", "sh": "ㄕ", "r": "ㄖ",
        "z": "ㄗ", "c": "ㄘ", "s": "ㄙ",
    ]

    /// Final → bopomofo (or special hanzi token from the v1.1-zh vocab).
    static let finalMap: [String: String] = [
        "a": "ㄚ", "o": "ㄛ", "e": "ㄜ", "ie": "ㄝ",
        "ai": "ㄞ", "ei": "ㄟ", "ao": "ㄠ", "ou": "ㄡ",
        "an": "ㄢ", "en": "ㄣ", "ang": "ㄤ", "eng": "ㄥ",
        "er": "ㄦ", "i": "ㄧ", "u": "ㄨ", "v": "ㄩ",
        // Special "i"-after-sibilant variants.
        "ii": "ㄭ",
        "iii": "十",
        // Compound finals (these are encoded as Hanzi tokens in the
        // v1.1-zh vocab — the model was trained on this exact set).
        "ve": "月", "ia": "压", "ian": "言", "iang": "阳",
        "iao": "要", "in": "阴", "ing": "应", "iong": "用",
        "iou": "又", "ong": "中", "ua": "穵", "uai": "外",
        "uan": "万", "uang": "王", "uei": "为", "uen": "文",
        "ueng": "瓮", "uo": "我", "van": "元", "vn": "云",
    ]

    /// Punctuation passthrough (matches `ZH_MAP[p] = p` in misaki).
    /// Anything not in this set is dropped so the bopomofo string never
    /// contains characters the vocab can't encode.
    public static let allowedPunctuation: Set<Character> = [
        ";", ":", ",", ".", "!", "?", "/", "—", "…", "\"",
        "(", ")", "“", "”", " ",
    ]

    /// Strip pypinyin's "empty initial" surface forms so a downstream
    /// initial/final split sees the underlying canonical syllable. Mirrors
    /// the spelling rules at the top of `pypinyin/style/_constants.py`:
    ///
    /// * `yi` / `ya` / `yo` / … → drops the leading `y`, replaces with
    ///   `i` when the next vowel is not already `i` / `v`.
    /// * `wu` / `wa` / … → drops the leading `w`, replaces with `u`
    ///   when the next vowel is not already `u`.
    /// * `yu` / `yue` / `yuan` / `yun` → `v` / `ve` / `van` / `vn`.
    static func normalizePinyinForLookup(_ syllable: String) -> String {
        canonicalFinalForStandalone[syllable] ?? syllable
    }

    /// Inverse of `standaloneFinalSpelling` (`yi` → `i`, `wu` → `u`,
    /// `yue` → `ve`, …).
    static let canonicalFinalForStandalone: [String: String] = {
        var out: [String: String] = [:]
        for (canonical, spelled) in standaloneFinalSpelling { out[spelled] = canonical }
        return out
    }()

    /// Apply the `i → ii / iii` rule for sibilant initials, matching
    /// `_get_initials_finals` in misaki. Mutates `final` in-place.
    static func sibilantIFix(initial: String, final: inout String) {
        guard final == "i" else { return }
        switch initial {
        case "z", "c", "s":
            final = "ii"
        case "zh", "ch", "sh", "r":
            final = "iii"
        default:
            break
        }
    }

    /// Convert one toned syllable (e.g. `("hao", 3)`) into the bopomofo
    /// + digit string the v1.1-zh vocab expects (`"ㄏㄠ3"`). Returns
    /// `nil` when the syllable cannot be parsed (the caller logs and
    /// drops it — kokoro's own behaviour for OOV tokens is identical).
    ///
    /// Pass `erhua: true` to append a trailing `ㄦ` between the final
    /// and the tone digit (`小孩儿` → `ㄒㄧㄠ3ㄏㄞㄦ2`). Only the
    /// `儿` suffix is encoded — phonetic compaction (`-n`/`-ng` drop)
    /// is left to the acoustic model, matching misaki's style.
    public static func encode(syllable base: String, tone: Int, erhua: Bool = false) -> String? {
        guard !base.isEmpty else { return nil }
        let normalized = normalizePinyinForLookup(base)
        guard let (initial, finalRaw) = splitInitialFinal(normalized) else {
            return nil
        }
        var final = finalRaw
        sibilantIFix(initial: initial, final: &final)
        // j/q/x + u → v (ü). The umlaut is implicit in pinyin orthography
        // after these initials — e.g. `qu` is `qü` (`ㄑㄩ`), not `q + u`.
        // Without this rewrite the kokoro model speaks `cu` (ㄘㄨ) for `qù`,
        // which the zh-CN ASR mishears as 醋 instead of 去.
        if (initial == "j" || initial == "q" || initial == "x")
            && final.hasPrefix("u")
        {
            final = "v" + final.dropFirst()
        }

        // Expand standard pinyin orthographic contractions. In written
        // pinyin "ui"/"un"/"iu" are shorthands for "uei"/"uen"/"iou"
        // when preceded by a consonant initial. The finalMap only carries
        // the full forms (matching misaki's _get_initials_finals), so
        // syllables like gui/dui/hui/liu/jiu/dun would silently drop
        // without this expansion.
        if !initial.isEmpty {
            switch final {
            case "ui": final = "uei"
            case "un": final = "uen"
            case "iu": final = "iou"
            default: break
            }
        }

        var out = ""
        if !initial.isEmpty {
            guard let bo = initialMap[initial] else { return nil }
            out.append(bo)
        }
        if !final.isEmpty {
            guard let bo = finalMap[final] else { return nil }
            out.append(bo)
        }
        // Erhua suffix sits between the final and the tone digit so the
        // model sees `ㄒㄧㄠㄦ3` (a single toned r-coloured syllable)
        // rather than `ㄒㄧㄠ3ㄦ` (two tokens).
        if erhua, let bo = finalMap["er"] {
            out.append(bo)
        }
        // Tone digit — the v1.1-zh vocab carries 1…5 verbatim.
        if (1...5).contains(tone) {
            out.append(String(tone))
        }
        return out.isEmpty ? nil : out
    }

    /// Bopomofo initial → pinyin initial (inverse of `initialMap`).
    static let bopomofoInitialMap: [Character: String] = {
        var out: [Character: String] = [:]
        for (pinyin, bo) in initialMap { out[Character(bo)] = pinyin }
        return out
    }()

    /// Full bopomofo final → canonical pinyin final. Unlike `finalMap`
    /// (which targets the v1.1-zh vocab's compound tokens) these are the
    /// plain Zhuyin spellings g2pW labels use, so `ㄧㄥ` is `ing`, not
    /// `应`. `ㄨㄥ` is `ueng` here; `decode` rewrites it to `ong` after
    /// an initial. `ㄝ` (ê), `ㄧㄞ` (yai) and `ㄩㄤ` are left out on
    /// purpose: the zh vocab has no token for them, so `decode` returns
    /// `nil` and the caller keeps the dict reading instead of emitting a
    /// syllable `encode` would drop.
    static let bopomofoFinalMap: [String: String] = [
        "ㄚ": "a", "ㄛ": "o", "ㄜ": "e", "ㄞ": "ai", "ㄟ": "ei",
        "ㄠ": "ao", "ㄡ": "ou", "ㄢ": "an", "ㄣ": "en",
        "ㄤ": "ang", "ㄥ": "eng", "ㄦ": "er",
        "ㄧ": "i", "ㄧㄚ": "ia", "ㄧㄝ": "ie", "ㄧㄠ": "iao",
        "ㄧㄡ": "iou", "ㄧㄢ": "ian", "ㄧㄣ": "in", "ㄧㄤ": "iang",
        "ㄧㄥ": "ing",
        "ㄨ": "u", "ㄨㄚ": "ua", "ㄨㄛ": "uo", "ㄨㄞ": "uai",
        "ㄨㄟ": "uei", "ㄨㄢ": "uan", "ㄨㄣ": "uen", "ㄨㄤ": "uang",
        "ㄨㄥ": "ueng",
        "ㄩ": "v", "ㄩㄝ": "ve", "ㄩㄢ": "van", "ㄩㄣ": "vn",
        "ㄩㄥ": "iong",
    ]

    /// Standalone-final spelling (no initial), as pypinyin writes it.
    static let standaloneFinalSpelling: [String: String] = [
        "i": "yi", "ia": "ya", "ie": "ye", "iao": "yao", "iou": "you",
        "ian": "yan", "in": "yin", "iang": "yang", "ing": "ying",
        "iong": "yong",
        "u": "wu", "ua": "wa", "uo": "wo", "uai": "wai", "uei": "wei",
        "uan": "wan", "uen": "wen", "uang": "wang", "ueng": "weng",
        "v": "yu", "ve": "yue", "van": "yuan", "vn": "yun",
    ]

    /// Parse a g2pW label (`<bopomofo><tone digit>`, e.g. `ㄏㄤ2`) into
    /// the same pinyin `Syllable` the dict path produces (`hang`, 2),
    /// so a disambiguated reading flows through sandhi, erhua and
    /// `encode` exactly like a dict reading. The spelling follows
    /// pypinyin's TONE3 output (`xing`, `ju`, `liu`, `gong`, `weng`,
    /// `zhi`, `lv`), which is what `encode` and the sandhi rules key
    /// on. Returns `nil` for labels outside the standard inventory.
    public static func decode(bopomofo label: String) -> MandarinPinyinNormalizer.Syllable? {
        let digitForm = MandarinPolyphoneCatalog.toneDigitForm(label)
        guard let last = digitForm.last, let tone = Int(String(last)), (1...5).contains(tone)
        else { return nil }
        var body = Substring(digitForm.dropLast())
        guard let first = body.first else { return nil }

        var initial = ""
        if let ini = bopomofoInitialMap[first] {
            initial = ini
            body = body.dropFirst()
        }

        var final: String
        if body.isEmpty {
            // Bare sibilant (`ㄓ`, `ㄗ`, …): pinyin spells the buzzing
            // vowel as `i`. A bare non-sibilant initial is malformed.
            switch initial {
            case "zh", "ch", "sh", "r", "z", "c", "s":
                final = "i"
            default:
                return nil
            }
        } else {
            guard let mapped = bopomofoFinalMap[String(body)] else { return nil }
            final = mapped
        }

        if initial.isEmpty {
            final = standaloneFinalSpelling[final] ?? final
        } else {
            switch final {
            case "ueng": final = "ong"
            case "iou": final = "iu"
            case "uei": final = "ui"
            case "uen": final = "un"
            default: break
            }
            // j/q/x drop the umlaut in pinyin orthography (`ju`, `que`,
            // `xuan`, `jun`); `encode` restores it.
            if initial == "j" || initial == "q" || initial == "x", final.hasPrefix("v") {
                final = "u" + final.dropFirst()
            }
        }
        return MandarinPinyinNormalizer.Syllable(base: initial + final, tone: tone)
    }

    /// Longest-prefix initial split. The empty initial is allowed.
    static func splitInitialFinal(_ syllable: String) -> (String, String)? {
        for ini in initials {
            if syllable.hasPrefix(ini) {
                return (ini, String(syllable.dropFirst(ini.count)))
            }
        }
        // No matching initial → empty initial, full string is the final.
        return ("", syllable)
    }
}
#endif
