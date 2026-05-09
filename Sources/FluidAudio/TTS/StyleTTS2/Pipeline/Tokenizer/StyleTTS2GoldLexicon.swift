import Foundation

/// Misaki gold/silver English lexicon for StyleTTS2.
///
/// Mirrors the Kokoro pipeline: the per-word phoneme map is sourced from
/// `us_gold.json` (high-confidence, hand-curated) and `us_silver.json`
/// (lower-confidence) — the same JSON files Kokoro consumes for its
/// preprocessed `us_lexicon_cache.json`. Per-word lookup happens in this
/// order:
///
///   1. case-sensitive match in gold (preserves proper-noun casing such as
///      `"AI"` → `ˈAˌI`)
///   2. case-sensitive match in silver
///   3. lowercased + apostrophe-normalized match in gold
///   4. lowercased + apostrophe-normalized match in silver
///
/// On a hit the returned IPA string is rewritten through
/// `MisakiNormalizer` so Misaki's compact uppercase diphthong alphabet
/// (`A I O W Y`, `R`, `ᵊ`, …) lowers into the espeak-style IPA character
/// set that StyleTTS2's `text_encoder` was trained on. Without this step
/// `StyleTTS2TextCleaner.encode(_:)` would silently drop ~30 % of vowel
/// tokens (those characters aren't in StyleTTS2's vocab).
///
/// On a miss the caller falls back to CharsiuG2P (`MultilingualG2PModel`).
public actor StyleTTS2GoldLexicon {

    public enum LoadError: Error, LocalizedError {
        case fileMissing(URL)
        case decodeFailed(String)

        public var errorDescription: String? {
            switch self {
            case .fileMissing(let url):
                return "Misaki lexicon file missing at \(url.path)"
            case .decodeFailed(let detail):
                return "Misaki lexicon decode failed: \(detail)"
            }
        }
    }

    private let logger = AppLogger(category: "StyleTTS2GoldLexicon")

    /// Lower-cased word → IPA string (Misaki notation, pre-normalization).
    private var lowerGold: [String: String] = [:]
    private var lowerSilver: [String: String] = [:]

    /// Original-case word → IPA string. Misaki ships proper-noun and
    /// abbreviation entries in their canonical case (`"AI"`, `"NATO"`,
    /// `"iPhone"`); preserving them lets us hit the right pronunciation
    /// before falling back to the lowercased map.
    private var caseSensitiveGold: [String: String] = [:]
    private var caseSensitiveSilver: [String: String] = [:]

    private var loaded = false

    public init() {}

    /// Idempotently load both lexicons from `directory` (typically the
    /// Kokoro cache dir at `~/.cache/fluidaudio/Models/kokoro`). The silver
    /// file is optional — its absence only widens the CharsiuG2P fallback
    /// surface.
    public func load(directory: URL) throws {
        if loaded { return }

        let goldURL = directory.appendingPathComponent("us_gold.json")
        let silverURL = directory.appendingPathComponent("us_silver.json")

        guard FileManager.default.fileExists(atPath: goldURL.path) else {
            throw LoadError.fileMissing(goldURL)
        }

        let (lowerGoldMap, caseGoldMap) = try Self.parse(url: goldURL)
        lowerGold = lowerGoldMap
        caseSensitiveGold = caseGoldMap
        logger.info(
            "Loaded gold lexicon: \(lowerGoldMap.count) lower / \(caseGoldMap.count) cased entries")

        if FileManager.default.fileExists(atPath: silverURL.path) {
            do {
                let (lowerSilverMap, caseSilverMap) = try Self.parse(url: silverURL)
                lowerSilver = lowerSilverMap
                caseSensitiveSilver = caseSilverMap
                logger.info(
                    "Loaded silver lexicon: \(lowerSilverMap.count) lower / \(caseSilverMap.count) cased entries"
                )
            } catch {
                logger.warning("Silver lexicon unreadable at \(silverURL.path): \(error)")
            }
        } else {
            logger.notice("Silver lexicon not present at \(silverURL.path); skipping")
        }

        loaded = true
    }

    /// Returns the espeak-normalized IPA for `word`, or `nil` if neither
    /// lexicon has an entry. The lookup respects case for proper nouns and
    /// abbreviations before falling back to the lower-cased map.
    public func phonemes(for word: String) -> String? {
        let normalized = Self.normalizeKey(word)
        guard !normalized.isEmpty else { return nil }

        if let raw = caseSensitiveGold[word]
            ?? caseSensitiveSilver[word]
            ?? caseSensitiveGold[normalized]
            ?? caseSensitiveSilver[normalized]
            ?? lowerGold[normalized]
            ?? lowerSilver[normalized]
        {
            return MisakiNormalizer.normalize(raw)
        }
        return nil
    }

    /// Visible for tests: total entry counts after load.
    public func metrics() -> (lowerGold: Int, lowerSilver: Int, casedGold: Int, casedSilver: Int) {
        (lowerGold.count, lowerSilver.count, caseSensitiveGold.count, caseSensitiveSilver.count)
    }

    // MARK: - Parsing

    /// Each top-level entry in `us_gold.json` / `us_silver.json` is either:
    /// - a string (the phoneme transcription), or
    /// - a dict mapping POS tags ("DEFAULT", "NOUN", "VERB", …) to either a
    ///   string or `null`. Without sentence-level POS tagging we always
    ///   resolve to the `"DEFAULT"` value (and fall back to any non-null
    ///   string if `DEFAULT` itself is null/missing).
    static func parse(url: URL) throws -> (lower: [String: String], cased: [String: String]) {
        let data = try Data(contentsOf: url)
        guard
            let root = try JSONSerialization.jsonObject(with: data, options: [])
                as? [String: Any]
        else {
            throw LoadError.decodeFailed("top-level JSON is not an object")
        }

        var lower: [String: String] = [:]
        var cased: [String: String] = [:]
        lower.reserveCapacity(root.count)
        cased.reserveCapacity(root.count / 8)

        for (key, value) in root {
            guard let resolved = resolveEntry(value), !resolved.isEmpty else { continue }

            let normalized = normalizeKey(key)
            if normalized.isEmpty { continue }

            // Misaki keeps proper nouns / abbreviations in their original
            // case ("AI", "NATO", "iPhone"). Anything that survives
            // round-tripping through the lower-case key on its own is kept
            // case-sensitive too so we hit it first for inputs that match
            // the canonical case exactly.
            if key != normalized {
                cased[key] = resolved
            }
            // Last write wins on collision — gold's lowercase entries are
            // unique by construction; for silver this matches what Kokoro
            // does (`mapValues`).
            lower[normalized] = resolved
        }

        return (lower, cased)
    }

    private static func resolveEntry(_ value: Any) -> String? {
        if let s = value as? String { return s }
        if let dict = value as? [String: Any] {
            if let def = dict["DEFAULT"] as? String { return def }
            // Pick any non-null string fallback so we don't drop the entry
            // entirely when DEFAULT is null but a POS variant is set.
            for (_, v) in dict {
                if let s = v as? String, !s.isEmpty { return s }
            }
        }
        return nil
    }

    /// Strip surrounding whitespace, lowercase, and replace any of the
    /// curly-apostrophe variants Misaki sometimes uses with the ASCII
    /// `'`. Keeps internal hyphens (Misaki has entries like `-able`).
    static func normalizeKey(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "" }
        var out = String()
        out.reserveCapacity(trimmed.count)
        for ch in trimmed.lowercased() {
            switch ch {
            case "\u{2019}", "\u{02BC}", "\u{2018}", "\u{201B}", "\u{2032}":
                out.append("'")
            default:
                out.append(ch)
            }
        }
        return out
    }
}

/// Convert Misaki's compact IPA shorthand into the espeak-style IPA
/// character set StyleTTS2 was trained on.
///
/// Misaki collapses common American-English diphthongs and rhotics into
/// single uppercase ASCII letters. StyleTTS2's `TextCleaner` doesn't
/// know those characters, so without this rewrite the diphthong tokens
/// would be silently dropped by `StyleTTS2TextCleaner.encode(_:)`.
///
/// Mappings (US English; Misaki's `british=False` mode):
///   - `A` → `eɪ`   (FACE)
///   - `I` → `aɪ`   (PRICE)
///   - `O` → `oʊ`   (GOAT)
///   - `W` → `aʊ`   (MOUTH)
///   - `Y` → `ɔɪ`   (CHOICE)
///   - `Q` → `oʊ`   (variant goat used in some entries — collapsed to `O`)
///   - `R` → `ɹ`    (American rhotic)
///   - `ᵊ` → `ə`    (superscript schwa Misaki uses for reduced syllables)
///
/// Untouched characters (`ˈˌːᵻəɛɪɹʊʌ…`) already live in StyleTTS2's
/// vocabulary so they pass through verbatim.
public enum MisakiNormalizer {

    public static func normalize(_ ipa: String) -> String {
        var out = String()
        out.reserveCapacity(ipa.count + 4)
        for ch in ipa {
            switch ch {
            case "A":
                out.append("e")
                out.append("ɪ")
            case "I":
                out.append("a")
                out.append("ɪ")
            case "O":
                out.append("o")
                out.append("ʊ")
            case "W":
                out.append("a")
                out.append("ʊ")
            case "Y":
                out.append("ɔ")
                out.append("ɪ")
            case "Q":
                out.append("o")
                out.append("ʊ")
            case "R":
                out.append("ɹ")
            case "\u{1D4A}":  // ᵊ superscript schwa
                out.append("ə")
            default:
                out.append(ch)
            }
        }
        return out
    }
}
