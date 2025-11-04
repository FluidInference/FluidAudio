import CoreML
import Foundation
import OSLog

/// Streaming-specific LocalAgreement processor implementing the Whisper-Streaming LocalAgreement-2 policy.
///
/// This processor validates streaming transcription by comparing consecutive chunk outputs.
/// The algorithm:
/// 1. Process chunk N with fresh decoder state → Output N
/// 2. Process chunk N+1 with fresh decoder state → Output N+1
/// 3. Find longest common prefix between outputs
/// 4. Confirm matching tokens, hold remainder as provisional
/// 5. As chunk N+2 arrives, previous provisional tokens may be confirmed
///
/// This temporal comparison approach validates output stability across new audio context,
/// matching the original Whisper-Streaming algorithm.
internal actor LocalAgreementStreamingProcessor {
    private let logger = AppLogger(category: "LAStreaming")
    private let config: ASRConfig
    private let decoder: TdtDecoderV3

    // Stateless decoding state - reset per chunk
    private var chunkDecoderState = TdtDecoderState.make()

    // Previous chunk output for temporal comparison
    private var previousChunkOutput: TdtHypothesis?

    // Accumulated confirmed tokens across all chunks
    private var confirmedTokens: [Int] = []
    private var confirmedTimestamps: [Int] = []
    private var confirmedConfidences: [Float] = []

    // Provisional tokens held from previous chunk
    private var provisionalTokens: [Int] = []
    private var provisionalTimestamps: [Int] = []
    private var provisionalConfidences: [Float] = []

    private var processedChunks: Int = 0

    /// Initialize the streaming processor
    init(decoder: TdtDecoderV3, config: ASRConfig) {
        self.decoder = decoder
        self.config = config
    }

    /// Reset all state for a new streaming session
    func reset() {
        chunkDecoderState = TdtDecoderState.make()
        previousChunkOutput = nil
        confirmedTokens.removeAll()
        confirmedTimestamps.removeAll()
        confirmedConfidences.removeAll()
        provisionalTokens.removeAll()
        provisionalTimestamps.removeAll()
        provisionalConfidences.removeAll()
        processedChunks = 0

        logger.debug("Streaming processor reset")
    }

    /// Process a single chunk with LocalAgreement-2 validation.
    ///
    /// Follows the stateless pattern from ChunkProcessor:
    /// - Reset decoder state before processing
    /// - Perform single decoding pass
    /// - Compare output with previous chunk for temporal validation
    /// - Return confirmed and provisional tokens
    ///
    /// - Parameters:
    ///   - audioSamples: Audio samples for this chunk
    ///   - asrManager: ASRManager for model access and utilities
    ///   - globalFrameOffset: Timestamp offset for alignment
    ///
    /// - Returns: Tuple of (confirmedTokens, provisionalTokens, allTimestamps, allConfidences)
    func processChunk(
        audioSamples: [Float],
        asrManager: AsrManager,
        globalFrameOffset: Int
    ) async throws -> (
        confirmed: [Int],
        provisional: [Int],
        allTimestamps: [Int],
        allConfidences: [Float]
    ) {
        // Reset decoder state for fresh processing (stateless like ChunkProcessor)
        chunkDecoderState.reset()

        // Transcribe current chunk with fresh state
        let currentOutput = try await transcribeChunkWithFreshState(
            audioSamples,
            asrManager: asrManager,
            globalFrameOffset: globalFrameOffset
        )

        // Temporal comparison with previous chunk
        let newConfirmed: [Int]
        let newProvisional: [Int]
        var newConfirmedTimestamps: [Int] = []
        var newConfirmedConfidences: [Float] = []
        var newProvisionalTimestamps: [Int] = []
        var newProvisionalConfidences: [Float] = []

        if let previous = previousChunkOutput {
            // Find common prefix between previous and current output
            newConfirmed = findCommonPrefix(
                previousTokens: previous.ySequence,
                currentTokens: currentOutput.ySequence,
                previousConfidences: previous.tokenConfidences,
                currentConfidences: currentOutput.tokenConfidences,
                threshold: config.localAgreementConfig.confidenceThreshold
            )

            let confirmedCount = newConfirmed.count

            // Extract timestamps and confidences for confirmed portion
            if confirmedCount > 0 {
                newConfirmedTimestamps = Array(currentOutput.timestamps.prefix(confirmedCount))
                newConfirmedConfidences = Array(currentOutput.tokenConfidences.prefix(confirmedCount))
            }

            // Remainder is provisional
            newProvisional = Array(currentOutput.ySequence.dropFirst(confirmedCount))
            if newProvisional.count > 0 {
                newProvisionalTimestamps = Array(currentOutput.timestamps.dropFirst(confirmedCount))
                newProvisionalConfidences = Array(currentOutput.tokenConfidences.dropFirst(confirmedCount))
            }

            logger.debug(
                "Chunk \(self.processedChunks): "
                    + "PrevOutput=\(previous.tokenCount) CurrentOutput=\(currentOutput.tokenCount) "
                    + "Confirmed=\(confirmedCount) Provisional=\(newProvisional.count)"
            )
        } else {
            // First chunk - everything is provisional (nothing to confirm yet)
            newConfirmed = []
            newProvisional = currentOutput.ySequence
            newProvisionalTimestamps = currentOutput.timestamps
            newProvisionalConfidences = currentOutput.tokenConfidences

            logger.debug(
                "Chunk \(self.processedChunks): "
                    + "FirstChunk - all \(currentOutput.tokenCount) tokens held provisional"
            )
        }

        // Store current output for next iteration's comparison
        previousChunkOutput = currentOutput

        // Accumulate confirmed tokens
        confirmedTokens.append(contentsOf: newConfirmed)
        confirmedTimestamps.append(contentsOf: newConfirmedTimestamps)
        confirmedConfidences.append(contentsOf: newConfirmedConfidences)

        // Update provisional tokens for next chunk
        provisionalTokens = newProvisional
        provisionalTimestamps = newProvisionalTimestamps
        provisionalConfidences = newProvisionalConfidences

        // Enforce max provisional frames
        if provisionalTokens.count > config.localAgreementConfig.maxProvisionalFrames {
            let forceConfirmCount = provisionalTokens.count - config.localAgreementConfig.maxProvisionalFrames
            confirmedTokens.append(contentsOf: provisionalTokens.prefix(forceConfirmCount))
            confirmedTimestamps.append(contentsOf: provisionalTimestamps.prefix(forceConfirmCount))
            confirmedConfidences.append(contentsOf: provisionalConfidences.prefix(forceConfirmCount))

            provisionalTokens = Array(provisionalTokens.dropFirst(forceConfirmCount))
            provisionalTimestamps = Array(provisionalTimestamps.dropFirst(forceConfirmCount))
            provisionalConfidences = Array(provisionalConfidences.dropFirst(forceConfirmCount))

            logger.debug("Forced confirmation of \(forceConfirmCount) provisional tokens to prevent latency growth")
        }

        processedChunks += 1

        return (
            confirmed: newConfirmed,
            provisional: newProvisional,
            allTimestamps: newConfirmedTimestamps + newProvisionalTimestamps,
            allConfidences: newConfirmedConfidences + newProvisionalConfidences
        )
    }

    /// Finalize streaming and promote all remaining provisional tokens to confirmed.
    func finalize() -> (tokens: [Int], timestamps: [Int], confidences: [Float]) {
        // On finalization, promote all remaining provisional tokens
        confirmedTokens.append(contentsOf: provisionalTokens)
        confirmedTimestamps.append(contentsOf: provisionalTimestamps)
        confirmedConfidences.append(contentsOf: provisionalConfidences)

        logger.debug(
            "Streaming finalized with \(self.confirmedTokens.count) confirmed tokens, "
                + "\(self.processedChunks) chunks processed"
        )

        return (
            tokens: confirmedTokens,
            timestamps: confirmedTimestamps,
            confidences: confirmedConfidences
        )
    }

    /// Get current statistics
    var confirmedCount: Int { confirmedTokens.count }
    var provisionalCount: Int { provisionalTokens.count }
    var chunkCount: Int { processedChunks }

    // MARK: - Private Helpers

    /// Transcribe a chunk with fresh decoder state (stateless pattern).
    private func transcribeChunkWithFreshState(
        _ samples: [Float],
        asrManager: AsrManager,
        globalFrameOffset: Int
    ) async throws -> TdtHypothesis {
        let paddedChunk = asrManager.padAudioIfNeeded(samples, targetLength: 240_000)
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: samples.count)

        // Must copy state before passing to async function (can't pass actor-isolated inout)
        var stateCopy = chunkDecoderState
        let (hypothesis, _) = try await asrManager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: samples.count,
            actualAudioFrames: actualFrameCount,
            decoderState: &stateCopy,
            contextFrameAdjustment: 0,
            isLastChunk: false,
            globalFrameOffset: globalFrameOffset
        )

        // Update the actor-isolated state with the copy
        chunkDecoderState = stateCopy

        return hypothesis
    }

    /// Find longest common prefix between two token sequences.
    ///
    /// Compares tokens and verifies confidence agreement. A match requires:
    /// - Token IDs match
    /// - Both tokens meet confidence threshold
    /// - Stops at first disagreement
    ///
    /// - Parameters:
    ///   - previousTokens: Tokens from previous chunk's output
    ///   - currentTokens: Tokens from current chunk's output
    ///   - previousConfidences: Confidences from previous output
    ///   - currentConfidences: Confidences from current output
    ///   - threshold: Minimum confidence for token agreement
    ///
    /// - Returns: Array of confirmed tokens (longest common prefix)
    private func findCommonPrefix(
        previousTokens: [Int],
        currentTokens: [Int],
        previousConfidences: [Float],
        currentConfidences: [Float],
        threshold: Float
    ) -> [Int] {
        guard !previousTokens.isEmpty && !currentTokens.isEmpty else {
            return []
        }

        var matchLength = 0
        let minLength = min(previousTokens.count, currentTokens.count)

        for i in 0..<minLength {
            let prevToken = previousTokens[i]
            let currToken = currentTokens[i]
            let prevConf = previousConfidences[i]
            let currConf = currentConfidences[i]

            // Tokens must match and both must meet confidence threshold
            let tokensMatch = prevToken == currToken
            let bothAboveThreshold = prevConf >= threshold && currConf >= threshold

            if tokensMatch && bothAboveThreshold {
                matchLength += 1
            } else {
                // Log first mismatch for debugging
                if matchLength == 0 {
                    logger.debug(
                        "FirstMismatch@0: prev=\(prevToken)(conf=\(prevConf)) vs curr=\(currToken)(conf=\(currConf)) tokMatch=\(tokensMatch) confOK=\(bothAboveThreshold)"
                    )
                }
                // Stop at first disagreement
                break
            }
        }

        return Array(previousTokens.prefix(matchLength))
    }
}
