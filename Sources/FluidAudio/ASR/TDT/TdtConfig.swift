public struct TdtConfig: Sendable {
    public let includeTokenDuration: Bool
    public let maxSymbolsPerStep: Int
    public let durationBins: [Int]
    public let blankId: Int

    public static let `default` = TdtConfig()

    public init(
        includeTokenDuration: Bool = true,
        maxSymbolsPerStep: Int = 10,
        durationBins: [Int] = [0, 1, 2, 3, 4],
        // Parakeet-TDT-0.6b-v3 uses 8192 regular tokens + blank token at index 8192
        blankId: Int = 8192
    ) {
        self.includeTokenDuration = includeTokenDuration
        self.maxSymbolsPerStep = maxSymbolsPerStep
        self.durationBins = durationBins
        self.blankId = blankId
    }
}
