import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "ForcedAlignerTokenizer")

/// Minimal BPE tokenizer for the Qwen3-ForcedAligner.
///
/// Implements byte-pair encoding compatible with the Qwen3 tokenizer family.
/// Loads vocab.json (token -> id mapping) and merges.txt (BPE merge rules)
/// to tokenize input text for the forced alignment pipeline.
///
/// The forced aligner formats input as:
/// `<|audio_start|><|audio_pad|><|audio_end|><timestamp><timestamp>word1<timestamp><timestamp>word2<timestamp><timestamp>`
struct ForcedAlignerTokenizer {

    /// Token string to ID mapping
    private let vocab: [String: Int]

    /// BPE merge pairs in priority order
    private let merges: [(String, String)]

    /// Merge pair to rank mapping for fast lookup
    private let mergeRanks: [String: Int]

    /// Byte-to-unicode mapping (GPT-2 style)
    private static let byteToUnicode: [UInt8: Character] = {
        var printable = [Int]()
        printable.append(contentsOf: 33...126)
        printable.append(contentsOf: 161...172)
        printable.append(contentsOf: 174...255)
        let printableSet = Set(printable)

        var mapping = [UInt8: Character]()
        var extra: UInt32 = 256
        for b in 0...255 {
            if printableSet.contains(b) {
                mapping[UInt8(b)] = Character(Unicode.Scalar(b)!)
            } else {
                mapping[UInt8(b)] = Character(Unicode.Scalar(extra)!)
                extra += 1
            }
        }
        return mapping
    }()

    /// Initialize from vocab.json and merges.txt file contents.
    init(vocabData: Data, mergesData: Data) throws {
        guard let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            throw ForcedAlignerError.tokenizerFailed("Invalid vocab.json format")
        }
        self.vocab = vocabDict

        let mergesText = String(data: mergesData, encoding: .utf8) ?? ""
        let lines = mergesText.components(separatedBy: "\n")

        var mergePairs: [(String, String)] = []
        var ranks: [String: Int] = [:]

        for (idx, line) in lines.enumerated() {
            // Skip header and empty lines
            guard !line.hasPrefix("#") && !line.isEmpty else { continue }
            let parts = line.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let pair = (String(parts[0]), String(parts[1]))
            mergePairs.append(pair)
            let key = "\(pair.0) \(pair.1)"
            ranks[key] = idx
        }

        self.merges = mergePairs
        self.mergeRanks = ranks
    }

    /// Initialize from vocab.json and merges.txt file URLs.
    init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        let mergesData = try Data(contentsOf: mergesURL)
        try self.init(vocabData: vocabData, mergesData: mergesData)
    }

    /// Tokenize text into input_ids for the forced aligner.
    ///
    /// Formats the input as required by the aligner:
    /// `<|audio_start|><|audio_pad|>...<|audio_end|><timestamp><timestamp>word1<timestamp>...`
    ///
    /// - Parameters:
    ///   - text: Input text to align.
    ///   - numAudioFrames: Number of audio frames (determines audio_pad count).
    /// - Returns: Tuple of (wordList, inputIds) where inputIds includes all special tokens.
    func tokenize(text: String, numAudioFrames: Int) -> (words: [String], inputIds: [Int]) {
        // Split text into words
        let words = text.split(separator: " ").compactMap { segment -> String? in
            let cleaned = segment.filter { $0.isLetter || $0.isNumber || $0 == "'" }
            return cleaned.isEmpty ? nil : cleaned
        }

        // Build input_ids
        var inputIds: [Int] = []

        // Audio tokens
        inputIds.append(ForcedAlignerConfig.audioStartTokenId)
        for _ in 0..<numAudioFrames {
            inputIds.append(ForcedAlignerConfig.audioPadTokenId)
        }
        inputIds.append(ForcedAlignerConfig.audioEndTokenId)

        // Text tokens with timestamp delimiters
        // Format matches Python: word1<ts><ts>word2<ts><ts>...wordN<ts><ts>
        // (NO leading timestamps before the first word)
        for (i, word) in words.enumerated() {
            // <timestamp><timestamp> between words (not before first word)
            if i > 0 {
                inputIds.append(ForcedAlignerConfig.timestampTokenId)
                inputIds.append(ForcedAlignerConfig.timestampTokenId)
            }

            // Tokenize the word using BPE
            let wordTokenIds = encode(word)
            inputIds.append(contentsOf: wordTokenIds)
        }

        // Trailing <timestamp><timestamp>
        inputIds.append(ForcedAlignerConfig.timestampTokenId)
        inputIds.append(ForcedAlignerConfig.timestampTokenId)

        return (words: words, inputIds: inputIds)
    }

    /// Encode a single word/text into BPE token IDs.
    func encode(_ text: String) -> [Int] {
        // Convert text bytes to unicode characters (GPT-2 style)
        let unicodeChars = text.utf8.map { byte -> String in
            guard let ch = Self.byteToUnicode[byte] else { return "" }
            return String(ch)
        }

        // Start with individual characters as initial tokens
        var tokens = unicodeChars

        // Apply BPE merges iteratively
        while tokens.count > 1 {
            // Find the merge pair with highest priority (lowest rank)
            var bestRank = Int.max
            var bestIdx = -1

            for i in 0..<(tokens.count - 1) {
                let pair = "\(tokens[i]) \(tokens[i + 1])"
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }

            guard bestIdx >= 0 else { break }

            // Apply the merge
            let merged = tokens[bestIdx] + tokens[bestIdx + 1]
            tokens[bestIdx] = merged
            tokens.remove(at: bestIdx + 1)
        }

        // Look up token IDs
        return tokens.compactMap { token in
            vocab[token]
        }
    }

    /// Download tokenizer files from HuggingFace.
    ///
    /// Downloads vocab.json and merges.txt from the upstream Qwen3 model.
    static func download(to directory: URL) async throws -> (vocabURL: URL, mergesURL: URL) {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        let vocabURL = directory.appendingPathComponent("vocab.json")
        let mergesURL = directory.appendingPathComponent("merges.txt")

        if fm.fileExists(atPath: vocabURL.path) && fm.fileExists(atPath: mergesURL.path) {
            logger.info("Tokenizer files already present")
            return (vocabURL: vocabURL, mergesURL: mergesURL)
        }

        logger.info("Downloading tokenizer files from HuggingFace...")

        // Download from the upstream Qwen3-ForcedAligner model
        let baseURL = "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/resolve/main"

        for (filename, localURL) in [("vocab.json", vocabURL), ("merges.txt", mergesURL)] {
            guard !fm.fileExists(atPath: localURL.path) else { continue }
            guard let url = URL(string: "\(baseURL)/\(filename)") else {
                throw ForcedAlignerError.tokenizerFailed("Invalid URL for \(filename)")
            }

            let data = try await DownloadUtils.fetchHuggingFaceFile(
                from: url,
                description: "tokenizer/\(filename)"
            )
            try data.write(to: localURL)
            logger.info("Downloaded \(filename) (\(data.count) bytes)")
        }

        return (vocabURL: vocabURL, mergesURL: mergesURL)
    }
}
