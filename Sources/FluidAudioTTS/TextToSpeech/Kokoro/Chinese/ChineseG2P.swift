import Foundation

/// Chinese Grapheme-to-Phoneme converter for Kokoro TTS.
/// Converts Chinese text to Bopomofo (注音符号) phoneme sequences.
///
/// Pipeline:
/// 1. Text normalization (punctuation, numbers)
/// 2. Word segmentation (jieba-style)
/// 3. Pinyin conversion with POS tagging
/// 4. Tone sandhi application
/// 5. Bopomofo mapping
public final class ChineseG2P {

    // MARK: - Components

    private let tokenizer = ChineseTokenizer.shared
    private let pinyinConverter = PinyinConverter.shared
    private let toneSandhi = ToneSandhi()

    /// Unknown token placeholder
    private let unknownToken = "❓"

    /// Shared instance
    public static let shared = ChineseG2P()

    private var isInitialized = false

    private init() {}

    // MARK: - Initialization

    /// Load all required dictionaries from data
    public func initialize(
        jiebaData: Data,
        pinyinSingleData: Data,
        pinyinPhrasesData: Data? = nil
    ) throws {
        try tokenizer.loadDictionary(data: jiebaData, isCompressed: true)
        try pinyinConverter.loadSinglePinyin(data: pinyinSingleData, isCompressed: true)
        if let phrasesData = pinyinPhrasesData {
            try pinyinConverter.loadPhrasePinyin(data: phrasesData, isCompressed: true)
        }
        isInitialized = true
    }

    /// Load dictionaries from URLs
    public func initialize(
        jiebaURL: URL,
        pinyinSingleURL: URL,
        pinyinPhrasesURL: URL? = nil
    ) throws {
        try tokenizer.loadDictionary(from: jiebaURL)
        try pinyinConverter.loadSinglePinyin(from: pinyinSingleURL)
        if let phrasesURL = pinyinPhrasesURL {
            try pinyinConverter.loadPhrasePinyin(from: phrasesURL)
        }
        isInitialized = true
    }

    // MARK: - Public API

    /// Convert Chinese text to Bopomofo phoneme string
    /// - Parameter text: Chinese text input
    /// - Returns: Bopomofo phoneme string suitable for Kokoro TTS
    public func convert(_ text: String) throws -> String {
        guard isInitialized || tokenizer.isLoaded else {
            throw ChineseG2PError.notInitialized
        }

        // 1. Normalize text
        let normalized = normalizeText(text)
        guard !normalized.isEmpty else { return "" }

        // 2. Segment into words
        let words = segmentText(normalized)

        // 3. Convert each word to bopomofo
        var result: [String] = []

        for word in words {
            let bopomofo = convertWord(word)
            result.append(bopomofo)
        }

        // 4. Join without separator (matching Python misaki format)
        return result.joined(separator: "")
    }

    /// Convert Chinese text to array of phoneme tokens
    public func convertToTokens(_ text: String) throws -> [String] {
        let phonemeString = try convert(text)
        return tokenize(phonemeString)
    }

    // MARK: - Text Processing

    /// Normalize Chinese text (punctuation, whitespace)
    private func normalizeText(_ text: String) -> String {
        var result = text

        // Map Chinese punctuation to ASCII equivalents
        let punctuationMap: [Character: Character] = [
            "、": ",", "，": ",", "。": ".", "．": ".",
            "！": "!", "：": ":", "；": ";", "？": "?",
            "«": "\"", "»": "\"", "《": "\"", "》": "\"",
            "「": "\"", "」": "\"", "【": "\"", "】": "\"",
            "（": "(", "）": ")",
        ]

        result = String(result.map { punctuationMap[$0] ?? $0 })

        // Collapse whitespace
        result = result.components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
            .joined(separator: " ")

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Segment text into words
    private func segmentText(_ text: String) -> [String] {
        // Split by non-Chinese characters first
        var segments: [String] = []
        var currentSegment = ""
        var isChineseSegment = false

        for char in text {
            let isChinese = isChineseCharacter(char)

            if isChinese != isChineseSegment && !currentSegment.isEmpty {
                segments.append(currentSegment)
                currentSegment = ""
            }

            currentSegment.append(char)
            isChineseSegment = isChinese
        }

        if !currentSegment.isEmpty {
            segments.append(currentSegment)
        }

        // Segment Chinese portions
        var result: [String] = []
        for segment in segments {
            if let first = segment.first, isChineseCharacter(first) {
                let words = tokenizer.segment(segment)
                result.append(contentsOf: words)
            } else {
                result.append(segment)
            }
        }

        return result
    }

    /// Convert a single word to bopomofo
    private func convertWord(_ word: String) -> String {
        // Check if it's punctuation or non-Chinese
        if word.count == 1 {
            let char = word.first!
            if !isChineseCharacter(char) {
                return BopomofoMapper.convert(word)
            }
        }

        // Check for non-Chinese word
        if let first = word.first, !isChineseCharacter(first) {
            // Keep non-Chinese text as-is (could be English mixed in)
            return unknownToken
        }

        // Get pinyin for each character
        let pinyins = pinyinConverter.toPinyin(word, style: .tone3)

        // Apply tone sandhi
        // Note: Simplified - full implementation needs POS tagging
        let modifiedPinyins = toneSandhi.modifiedTone(word: word, pos: "n", finals: pinyins)

        // Convert to bopomofo
        let bopomofo = modifiedPinyins.map { BopomofoMapper.convert($0) }

        return bopomofo.joined()
    }

    /// Check if a character is Chinese
    private func isChineseCharacter(_ char: Character) -> Bool {
        guard let scalar = char.unicodeScalars.first else { return false }
        // CJK Unified Ideographs range
        return scalar.value >= 0x4E00 && scalar.value <= 0x9FFF
    }

    /// Tokenize bopomofo string into individual tokens
    private func tokenize(_ bopomofo: String) -> [String] {
        // Split on slashes (word boundaries)
        let words = bopomofo.split(separator: "/")

        var tokens: [String] = []
        for word in words {
            // Each character/symbol is a token
            for char in word {
                tokens.append(String(char))
            }
            // Add word boundary marker if needed
        }

        return tokens
    }

    // MARK: - Vocabulary

    /// Get the Bopomofo vocabulary for Kokoro
    public var vocabulary: Set<String> {
        BopomofoMapper.vocabulary
    }
}

// MARK: - Errors

public enum ChineseG2PError: Error, LocalizedError {
    case notInitialized
    case conversionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "ChineseG2P not initialized. Call initialize() with dictionary data first."
        case .conversionFailed(let message):
            return "G2P conversion failed: \(message)"
        }
    }
}
