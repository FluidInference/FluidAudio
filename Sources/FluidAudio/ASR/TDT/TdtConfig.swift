public struct TdtConfig: Sendable {
    public let includeTokenDuration: Bool
    public let maxSymbolsPerStep: Int
    public let durationBins: [Int]
    public let blankId: Int
    public let boundarySearchFrames: Int
    public let maxTokensPerChunk: Int
    public let consecutiveBlankLimit: Int

    public static let `default` = TdtConfig()

    public init(
        includeTokenDuration: Bool = true,
        maxSymbolsPerStep: Int = 10,
        durationBins: [Int] = [0, 1, 2, 3, 4],
        // Parakeet-TDT-0.6b-v3 uses 8192 regular tokens + blank token at index 8192
        blankId: Int = 8192,
        // Number of frames to search at chunk boundaries for token alignment (NeMo tdt_search_boundary)
        // Increased to 20 frames (1.6s) to match the context overlap in ChunkProcessor
        boundarySearchFrames: Int = 20,
        // Maximum tokens to process per chunk (prevents runaway decoding)
        // Increased to 150 for 11.2s center chunks (~40 words * 2-3 tokens/word + buffer)
        maxTokensPerChunk: Int = 150,
        // Number of consecutive blanks to trigger early termination in last chunk
        consecutiveBlankLimit: Int = 5
    ) {
        self.includeTokenDuration = includeTokenDuration
        self.maxSymbolsPerStep = maxSymbolsPerStep
        self.durationBins = durationBins
        self.blankId = blankId
        self.boundarySearchFrames = boundarySearchFrames
        self.maxTokensPerChunk = maxTokensPerChunk
        self.consecutiveBlankLimit = consecutiveBlankLimit
    }
}
