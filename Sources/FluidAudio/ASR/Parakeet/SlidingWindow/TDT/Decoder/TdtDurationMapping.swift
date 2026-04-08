import Foundation

/// Utilities for mapping TDT duration bins to encoder frame advances.
enum TdtDurationMapping {

    /// Map a duration bin index to actual encoder frames to advance.
    ///
    /// Parakeet-TDT models use a discrete duration head with bins that map to frame advances.
    /// - v3 models: 5 bins [1, 2, 3, 4, 5] (direct 1:1 mapping)
    /// - v2 models: May have different bin configurations
    ///
    /// - Parameters:
    ///   - binIndex: The duration bin index from the model output
    ///   - durationBins: Array mapping bin indices to frame advances
    /// - Returns: Number of encoder frames to advance
    /// - Throws: `ASRError.invalidDurationBin` if binIndex is out of range
    static func mapDurationBin(_ binIndex: Int, durationBins: [Int]) throws -> Int {
        guard binIndex >= 0 && binIndex < durationBins.count else {
            throw ASRError.processingFailed("Duration bin index out of range: \(binIndex)")
        }
        return durationBins[binIndex]
    }

    /// Clamp probability to valid range [0, 1] to handle edge cases.
    ///
    /// - Parameter value: Raw probability value (may be slightly outside [0,1] due to float precision or NaN)
    /// - Returns: Clamped probability in [0, 1], or 0 if value is not finite
    static func clampProbability(_ value: Float) -> Float {
        guard value.isFinite else { return 0 }
        return max(0.0, min(1.0, value))
    }
}
