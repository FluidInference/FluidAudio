import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]

    private let logger = AppLogger(category: "ChunkProcessor")
    private typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float)
    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let start: Double
        let end: Double
    }

    // Stateless chunking aligned with CoreML reference:
    // - process ~14.96s of audio per window (frame-aligned) to stay under encoder limit
    // - 2.0s overlap (frame-aligned) to give the decoder slack when merging windows
    private let sampleRate: Int = 16000
    private let overlapSeconds: Double = 2.0

    private var maxModelSamples: Int { 240_000 }  // CoreML encoder capacity (15 seconds)
    private var chunkSamples: Int {
        // Match CoreML reference chunk length (239,840 samples ≈ 14.99s)
        let raw = max(maxModelSamples - ASRConstants.melHopSize, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }
    private var overlapSamples: Int {
        let requested = Int(overlapSeconds * Double(sampleRate))
        let capped = min(requested, chunkSamples / 2)
        return capped / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }
    private var strideSamples: Int {
        let raw = max(chunkSamples - overlapSamples, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    func process(
        using manager: AsrManager,
        startTime: Date,
        progressHandler: ((Double) async -> Void)? = nil
    ) async throws -> ASRResult {
        var chunkOutputs: [[TokenWindow]] = []

        var chunkStart = 0
        var chunkIndex = 0
        var chunkDecoderState = TdtDecoderState.make()

        while chunkStart < audioSamples.count {
            let candidateEnd = chunkStart + chunkSamples
            let isLastChunk = candidateEnd >= audioSamples.count
            let chunkEnd = isLastChunk ? audioSamples.count : candidateEnd

            if chunkEnd <= chunkStart {
                break
            }

            chunkDecoderState.reset()

            let chunkRange = chunkStart..<chunkEnd
            let chunkSamplesSlice = Array(audioSamples[chunkRange])

            let (windowTokens, windowTimestamps, windowConfidences) = try await transcribeChunk(
                samples: chunkSamplesSlice,
                chunkStart: chunkStart,
                isLastChunk: isLastChunk,
                using: manager,
                decoderState: &chunkDecoderState
            )

            // Combine tokens, timestamps, and confidences into aligned tuples
            guard windowTokens.count == windowTimestamps.count && windowTokens.count == windowConfidences.count else {
                throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
            }

            let windowData: [TokenWindow] = zip(zip(windowTokens, windowTimestamps), windowConfidences).map {
                (token: $0.0.0, timestamp: $0.0.1, confidence: $0.1)
            }
            chunkOutputs.append(windowData)

            chunkIndex += 1

            if isLastChunk {
                break
            }

            if let progressHandler {
                let progress = min(
                    1.0, max(0.0, Double(chunkEnd) / Double(audioSamples.count)))
                await progressHandler(progress)
            }

            chunkStart += strideSamples
        }

        guard var mergedTokens = chunkOutputs.first else {
            return manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if chunkOutputs.count > 1 {
            for chunk in chunkOutputs.dropFirst() {
                mergedTokens = mergeChunks(mergedTokens, chunk)
            }
        }

        if mergedTokens.count > 1 {
            mergedTokens.sort { $0.timestamp < $1.timestamp }
        }

        let allTokens = mergedTokens.map { $0.token }
        let allTimestamps = mergedTokens.map { $0.timestamp }
        let allConfidences = mergedTokens.map { $0.confidence }

        return manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            confidences: allConfidences,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: audioSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    private func transcribeChunk(
        samples: [Float],
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float]) {
        guard !samples.isEmpty else { return ([], [], []) }

        let paddedChunk = manager.padAudioIfNeeded(samples, targetLength: maxModelSamples)
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: samples.count)
        let globalFrameOffset = chunkStart / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: samples.count,
            actualAudioFrames: actualFrameCount,
            decoderState: &decoderState,
            contextFrameAdjustment: 0,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        if hypothesis.isEmpty || encoderSequenceLength == 0 {
            return ([], [], [])
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences)
    }

    private func mergeChunks(
        _ left: [TokenWindow],
        _ right: [TokenWindow]
    ) -> [TokenWindow] {
        if left.isEmpty { return right }
        if right.isEmpty { return left }

        let frameDuration = Double(ASRConstants.samplesPerEncoderFrame) / Double(sampleRate)
        let overlapDuration = overlapSeconds
        let halfOverlapWindow = overlapDuration / 2

        func startTime(of token: TokenWindow) -> Double {
            Double(token.timestamp) * frameDuration
        }

        func endTime(of token: TokenWindow) -> Double {
            startTime(of: token) + frameDuration
        }

        let leftEndTime = endTime(of: left.last!)
        let rightStartTime = startTime(of: right.first!)

        if leftEndTime <= rightStartTime {
            return left + right
        }

        let overlapLeft: [IndexedToken] = left.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            let end = start + frameDuration
            guard end > rightStartTime - overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: end)
        }

        let overlapRight: [IndexedToken] = right.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            guard start < leftEndTime + overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: start + frameDuration)
        }

        guard overlapLeft.count >= 2 && overlapRight.count >= 2 else {
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration)
        }

        let minimumPairs = max(overlapLeft.count / 2, 1)

        let contiguousPairs = findBestContiguousPairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        if contiguousPairs.count >= minimumPairs {
            return mergeUsingMatches(
                matches: contiguousPairs,
                overlapLeft: overlapLeft,
                overlapRight: overlapRight,
                left: left,
                right: right
            )
        }

        let lcsPairs = findLongestCommonSubsequencePairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        guard !lcsPairs.isEmpty else {
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration)
        }

        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right
        )
    }

    private func findBestContiguousPairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        var best: [(Int, Int)] = []

        for i in 0..<overlapLeft.count {
            for j in 0..<overlapRight.count {
                let leftToken = overlapLeft[i]
                let rightToken = overlapRight[j]

                if tokensMatch(leftToken, rightToken, tolerance: tolerance) {
                    var current: [(Int, Int)] = []
                    var k = i
                    var l = j

                    while k < overlapLeft.count && l < overlapRight.count {
                        let nextLeft = overlapLeft[k]
                        let nextRight = overlapRight[l]

                        if tokensMatch(nextLeft, nextRight, tolerance: tolerance) {
                            current.append((k, l))
                            k += 1
                            l += 1
                        } else {
                            break
                        }
                    }

                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }

        return best
    }

    private func findLongestCommonSubsequencePairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        let leftCount = overlapLeft.count
        let rightCount = overlapRight.count

        var dp = Array(repeating: Array(repeating: 0, count: rightCount + 1), count: leftCount + 1)

        for i in 1...leftCount {
            for j in 1...rightCount {
                if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var pairs: [(Int, Int)] = []
        var i = leftCount
        var j = rightCount

        while i > 0 && j > 0 {
            if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        return pairs.reversed()
    }

    private func tokensMatch(_ left: IndexedToken, _ right: IndexedToken, tolerance: Double) -> Bool {
        guard left.token.token == right.token.token else { return false }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    private func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow]
    ) -> [TokenWindow] {
        let leftIndices = matches.map { overlapLeft[$0.0].index }
        let rightIndices = matches.map { overlapRight[$0.1].index }

        var result: [TokenWindow] = []

        if let firstLeft = leftIndices.first, firstLeft > 0 {
            result.append(contentsOf: left[..<firstLeft])
        }

        for idx in 0..<matches.count {
            let leftIndex = leftIndices[idx]
            let rightIndex = rightIndices[idx]

            result.append(left[leftIndex])

            guard idx < matches.count - 1 else { continue }

            let nextLeftIndex = leftIndices[idx + 1]
            let nextRightIndex = rightIndices[idx + 1]

            let gapLeft = nextLeftIndex > leftIndex + 1 ? Array(left[(leftIndex + 1)..<nextLeftIndex]) : []
            let gapRight = nextRightIndex > rightIndex + 1 ? Array(right[(rightIndex + 1)..<nextRightIndex]) : []

            if gapRight.count > gapLeft.count {
                result.append(contentsOf: gapRight)
            } else {
                result.append(contentsOf: gapLeft)
            }
        }

        if let lastRight = rightIndices.last, lastRight + 1 < right.count {
            result.append(contentsOf: right[(lastRight + 1)...])
        }

        return result
    }

    private func mergeByMidpoint(
        left: [TokenWindow],
        right: [TokenWindow],
        leftEndTime: Double,
        rightStartTime: Double,
        frameDuration: Double
    ) -> [TokenWindow] {
        let cutoff = (leftEndTime + rightStartTime) / 2
        let trimmedLeft = left.filter { Double($0.timestamp) * frameDuration <= cutoff }
        let trimmedRight = right.filter { Double($0.timestamp) * frameDuration >= cutoff }
        return trimmedLeft + trimmedRight
    }
}
