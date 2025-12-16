import Foundation
import OSLog

/// Resolves and validates speaker count constraints for clustering.
@available(macOS 14.0, iOS 17.0, *)
struct SpeakerCountConstraints: Sendable {

    let numSpeakers: Int?
    let minSpeakers: Int
    let maxSpeakers: Int

    private static let logger = AppLogger(category: "SpeakerCountConstraints")

    /// Resolves speaker count constraints, clamping to valid ranges.
    ///
    /// - Parameters:
    ///   - numEmbeddings: Total number of embeddings available for clustering.
    ///   - numSpeakers: Exact speaker count (overrides min/max if set). Values <= 0 are clamped to 1.
    ///   - minSpeakers: Minimum speakers (defaults to 1). Values <= 0 are clamped to 1.
    ///   - maxSpeakers: Maximum speakers (defaults to numEmbeddings).
    /// - Returns: Resolved constraints with validated bounds.
    ///
    /// - Note: If `minSpeakers > maxSpeakers`, `minSpeakers` is silently clamped to `maxSpeakers`.
    ///   This prevents crashes but may not reflect user intent.
    static func resolve(
        numEmbeddings: Int,
        numSpeakers: Int?,
        minSpeakers: Int?,
        maxSpeakers: Int?
    ) -> SpeakerCountConstraints {
        var resolvedMin = numSpeakers ?? minSpeakers ?? 1
        resolvedMin = max(1, min(numEmbeddings, resolvedMin))

        var resolvedMax = numSpeakers ?? maxSpeakers ?? numEmbeddings
        resolvedMax = max(1, min(numEmbeddings, resolvedMax))

        if resolvedMin > resolvedMax {
            logger.warning(
                "minSpeakers (\(resolvedMin)) > maxSpeakers (\(resolvedMax)); clamping minSpeakers to \(resolvedMax)"
            )
            resolvedMin = resolvedMax
        }

        let resolvedNum: Int?
        if resolvedMin == resolvedMax {
            resolvedNum = resolvedMin
        } else {
            resolvedNum = numSpeakers
        }

        logger.debug(
            "Resolved constraints: numSpeakers=\(resolvedNum.map(String.init) ?? "nil"), min=\(resolvedMin), max=\(resolvedMax)"
        )

        return SpeakerCountConstraints(
            numSpeakers: resolvedNum,
            minSpeakers: resolvedMin,
            maxSpeakers: resolvedMax
        )
    }

    /// Checks if the detected speaker count needs adjustment.
    func needsAdjustment(detectedCount: Int) -> Bool {
        detectedCount < minSpeakers || detectedCount > maxSpeakers
    }

    /// Returns the target speaker count to use.
    func targetCount(detectedCount: Int) -> Int {
        if detectedCount < minSpeakers {
            return minSpeakers
        }
        if detectedCount > maxSpeakers {
            return maxSpeakers
        }
        return detectedCount
    }
}
