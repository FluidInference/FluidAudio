import Accelerate
import Foundation

struct OfflineReconstruction {
    private let config: OfflineDiarizerConfig
    private let logger = AppLogger(category: "OfflineReconstruction")

    private struct Accumulator {
        var start: Double
        var end: Double
        var scoreSum: Double
        var frameCount: Int
    }

    init(config: OfflineDiarizerConfig) {
        self.config = config
    }

    func buildSegments(
        segmentation: SegmentationOutput,
        hardClusters: [[Int]],
        centroids: [[Double]]
    ) -> [TimedSpeakerSegment] {
        guard segmentation.numChunks > 0, segmentation.numFrames > 0 else { return [] }

        let frameDuration = segmentation.frameDuration
        guard frameDuration > 0 else { return [] }

        let clusterCount = max(centroids.count, 1)
        let gapThreshold = max(config.minGapDuration, config.segmentationMinDurationOff)

        var maxTime = 0.0
        for chunkIndex in 0..<segmentation.numChunks {
            let offset = chunkStartTime(for: chunkIndex, segmentation: segmentation)
            let end = offset + Double(segmentation.numFrames) * frameDuration
            if end > maxTime {
                maxTime = end
            }
        }

        let totalFrames = max(1, Int(ceil(maxTime / frameDuration)))
        var activationSums = Array(
            repeating: Array(repeating: 0.0, count: clusterCount),
            count: totalFrames
        )
        var activationCounts = Array(
            repeating: Array(repeating: 0.0, count: clusterCount),
            count: totalFrames
        )
        var expectedCountSums = [Double](repeating: 0, count: totalFrames)
        var expectedCountWeights = [Double](repeating: 0, count: totalFrames)

        for chunkIndex in 0..<segmentation.numChunks {
            guard chunkIndex < segmentation.speakerWeights.count else { continue }
            let chunkWeights = segmentation.speakerWeights[chunkIndex]
            guard !chunkWeights.isEmpty else { continue }

            let chunkOffset = chunkStartTime(for: chunkIndex, segmentation: segmentation)
            let chunkAssignments =
                chunkIndex < hardClusters.count
                ? hardClusters[chunkIndex] : Array(repeating: -2, count: segmentation.numSpeakers)

            for frameIndex in 0..<chunkWeights.count {
                let frameStart = chunkOffset + Double(frameIndex) * frameDuration
                var globalFrame = Int((frameStart / frameDuration).rounded())
                if globalFrame < 0 {
                    globalFrame = 0
                } else if globalFrame >= totalFrames {
                    globalFrame = totalFrames - 1
                }

                let weights = chunkWeights[frameIndex]
                var frameActivations = [Double](repeating: 0, count: clusterCount)

                for speakerIndex in 0..<min(weights.count, chunkAssignments.count) {
                    let cluster = chunkAssignments[speakerIndex]
                    guard cluster >= 0, cluster < clusterCount else { continue }
                    let value = Double(weights[speakerIndex])
                    if value > frameActivations[cluster] {
                        frameActivations[cluster] = value
                    }
                }

                let expectedCount = weights.reduce(0.0) { partialSum, value in
                    partialSum + Double(value)
                }
                expectedCountSums[globalFrame] += expectedCount
                expectedCountWeights[globalFrame] += 1

                for cluster in 0..<clusterCount {
                    let value = frameActivations[cluster]
                    if value > 0 {
                        activationSums[globalFrame][cluster] += value
                        activationCounts[globalFrame][cluster] += 1
                    }
                }
            }
        }

        var activationAverages = Array(
            repeating: Array(repeating: 0.0, count: clusterCount),
            count: totalFrames
        )
        for frame in 0..<totalFrames {
            let sums = activationSums[frame]
            let counts = activationCounts[frame]
            var averages = [Double](repeating: 0, count: clusterCount)

            // Vectorized division: averages = sums / counts (where counts > 0)
            sums.withUnsafeBufferPointer { sumsPtr in
                counts.withUnsafeBufferPointer { countsPtr in
                    averages.withUnsafeMutableBufferPointer { averagesPtr in
                        guard let sumsBase = sumsPtr.baseAddress,
                            let countsBase = countsPtr.baseAddress,
                            let averagesBase = averagesPtr.baseAddress
                        else { return }

                        vDSP_vdivD(
                            countsBase,
                            1,
                            sumsBase,
                            1,
                            averagesBase,
                            1,
                            vDSP_Length(clusterCount)
                        )
                    }
                }
            }

            // Zero out results where count was 0 (to avoid division by zero artifacts)
            for cluster in 0..<clusterCount where counts[cluster] == 0 {
                averages[cluster] = 0
            }

            activationAverages[frame] = averages
        }

        var speakerCountPerFrame = [Int](repeating: 0, count: totalFrames)
        var speakerCountHistogram: [Int: Int] = [:]
        let maxAllowedSpeakers = min(clusterCount, segmentation.numSpeakers)
        for frame in 0..<totalFrames {
            let weight = expectedCountWeights[frame]
            guard weight > 0 else { continue }
            var rounded = Int((expectedCountSums[frame] / weight).rounded(.toNearestOrEven))
            if rounded < 0 { rounded = 0 }
            if rounded > maxAllowedSpeakers { rounded = maxAllowedSpeakers }
            speakerCountPerFrame[frame] = rounded
            speakerCountHistogram[rounded, default: 0] += 1
        }

        if !speakerCountHistogram.isEmpty {
            let histogramString =
                speakerCountHistogram
                .sorted { $0.key < $1.key }
                .map { "\($0.key):\($0.value)" }
                .joined(separator: ", ")
            logger.debug("Speaker-count histogram \(histogramString)")
        }

        var perFrameClusters = [[Int]](repeating: [], count: totalFrames)
        for frame in 0..<totalFrames {
            let required = speakerCountPerFrame[frame]
            guard required > 0 else { continue }
            let ranked = activationSums[frame].enumerated().sorted { $0.element > $1.element }
            let selected = ranked.prefix(required).map { $0.offset }
            perFrameClusters[frame] = selected
        }

        var activeSegments: [Int: Accumulator] = [:]
        var rawSegments: [TimedSpeakerSegment] = []

        for frameIndex in 0..<totalFrames {
            let frameStart = Double(frameIndex) * frameDuration
            let frameEnd = frameStart + frameDuration
            let activeClusters = Set(perFrameClusters[frameIndex])
            let averageScores = activationAverages[frameIndex]

            for (cluster, accumulator) in activeSegments where !activeClusters.contains(cluster) {
                appendSegment(
                    cluster: cluster,
                    accumulator: accumulator,
                    endTime: frameStart,
                    centroids: centroids,
                    output: &rawSegments
                )
            }
            activeSegments = activeSegments.filter { activeClusters.contains($0.key) }

            for cluster in activeClusters {
                let score = averageScores.indices.contains(cluster) ? averageScores[cluster] : 0
                if var existing = activeSegments[cluster] {
                    existing.end = frameEnd
                    existing.scoreSum += score
                    existing.frameCount += 1
                    activeSegments[cluster] = existing
                } else {
                    activeSegments[cluster] = Accumulator(
                        start: frameStart,
                        end: frameEnd,
                        scoreSum: score,
                        frameCount: 1
                    )
                }
            }
        }

        for (cluster, accumulator) in activeSegments {
            appendSegment(
                cluster: cluster,
                accumulator: accumulator,
                endTime: accumulator.end,
                centroids: centroids,
                output: &rawSegments
            )
        }

        let merged = mergeSegments(rawSegments, gapThreshold: gapThreshold)
        return sanitize(segments: merged)
    }

    func buildSpeakerDatabase(
        segments: [TimedSpeakerSegment]
    ) -> [String: [Float]] {
        var sums: [String: [Float]] = [:]
        var counts: [String: Int] = [:]

        for segment in segments {
            if var current = sums[segment.speakerId] {
                let embedding = segment.embedding
                precondition(
                    embedding.count == current.count,
                    "Embedding dimensionality mismatch while accumulating speaker database"
                )
                embedding.withUnsafeBufferPointer { sourcePointer in
                    current.withUnsafeMutableBufferPointer { destinationPointer in
                        guard
                            let sourceBase = sourcePointer.baseAddress,
                            let destinationBase = destinationPointer.baseAddress
                        else { return }
                        cblas_saxpy(
                            Int32(embedding.count),
                            1.0,
                            sourceBase,
                            1,
                            destinationBase,
                            1
                        )
                    }
                }
                sums[segment.speakerId] = current
            } else {
                sums[segment.speakerId] = segment.embedding
            }
            counts[segment.speakerId, default: 0] += 1
        }

        var database: [String: [Float]] = [:]
        for (speaker, sum) in sums {
            guard let count = counts[speaker], count > 0 else { continue }
            var averaged = sum
            var scale = 1 / Float(count)
            let length = averaged.count
            averaged.withUnsafeMutableBufferPointer { pointer in
                guard let baseAddress = pointer.baseAddress else { return }
                vDSP_vsmul(
                    baseAddress,
                    1,
                    &scale,
                    baseAddress,
                    1,
                    vDSP_Length(length)
                )
            }
            database[speaker] = averaged
        }

        return database
    }

    private func excludeOverlaps(in segments: [TimedSpeakerSegment]) -> [TimedSpeakerSegment] {
        guard !segments.isEmpty else { return [] }

        var sanitized: [TimedSpeakerSegment] = []

        for segment in segments {
            var adjustedStart = segment.startTimeSeconds
            let adjustedEnd = segment.endTimeSeconds

            if let previous = sanitized.last {
                if adjustedStart < previous.endTimeSeconds {
                    adjustedStart = previous.endTimeSeconds
                }
            }

            if adjustedStart >= adjustedEnd {
                continue
            }

            let duration = adjustedEnd - adjustedStart
            if duration < Float(config.minSegmentDuration) {
                continue
            }

            let originalDuration = segment.endTimeSeconds - segment.startTimeSeconds
            let qualityScale = originalDuration > 0 ? duration / originalDuration : 1
            let adjustedQuality = max(0, min(1, segment.qualityScore * qualityScale))

            let trimmed = TimedSpeakerSegment(
                speakerId: segment.speakerId,
                embedding: segment.embedding,
                startTimeSeconds: adjustedStart,
                endTimeSeconds: adjustedEnd,
                qualityScore: adjustedQuality
            )
            sanitized.append(trimmed)
        }

        return sanitized
    }

    private func appendSegment(
        cluster: Int,
        accumulator: Accumulator,
        endTime: Double,
        centroids: [[Double]],
        output: inout [TimedSpeakerSegment]
    ) {
        guard endTime > accumulator.start else { return }
        let averageScore: Double
        if accumulator.frameCount > 0 {
            averageScore = accumulator.scoreSum / Double(accumulator.frameCount)
        } else {
            averageScore = accumulator.scoreSum
        }
        let quality = Float(min(max(averageScore, 0), 1))
        let centroidDouble =
            centroids.indices.contains(cluster)
            ? centroids[cluster]
            : Array(repeating: 0, count: centroids.first?.count ?? 0)
        let centroid = centroidDouble.map { Float($0) }

        let segment = TimedSpeakerSegment(
            speakerId: "S\(cluster + 1)",
            embedding: centroid,
            startTimeSeconds: Float(accumulator.start),
            endTimeSeconds: Float(endTime),
            qualityScore: quality
        )
        output.append(segment)
    }

    private func mergeSegments(
        _ segments: [TimedSpeakerSegment],
        gapThreshold: Double
    ) -> [TimedSpeakerSegment] {
        guard !segments.isEmpty else { return [] }

        let sorted = segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
        var merged: [TimedSpeakerSegment] = []
        var current = sorted[0]

        for segment in sorted.dropFirst() {
            if segment.speakerId == current.speakerId {
                let gap = Double(segment.startTimeSeconds) - Double(current.endTimeSeconds)
                if gap <= gapThreshold {
                    let blended = blendedQuality(current, segment)
                    current = TimedSpeakerSegment(
                        speakerId: current.speakerId,
                        embedding: current.embedding,
                        startTimeSeconds: current.startTimeSeconds,
                        endTimeSeconds: max(current.endTimeSeconds, segment.endTimeSeconds),
                        qualityScore: blended
                    )
                    continue
                }
            }

            merged.append(current)
            current = segment
        }

        merged.append(current)
        return merged
    }

    private func blendedQuality(_ lhs: TimedSpeakerSegment, _ rhs: TimedSpeakerSegment) -> Float {
        let lhsDuration = Double(lhs.durationSeconds)
        let rhsDuration = Double(rhs.durationSeconds)
        let totalDuration = lhsDuration + rhsDuration

        guard totalDuration > 0 else {
            return min(max((lhs.qualityScore + rhs.qualityScore) / 2, 0), 1)
        }

        let weighted =
            Double(lhs.qualityScore) * lhsDuration
            + Double(rhs.qualityScore) * rhsDuration

        return Float(min(max(weighted / totalDuration, 0), 1))
    }

    private func sanitize(segments: [TimedSpeakerSegment]) -> [TimedSpeakerSegment] {
        var ordered = segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
        let minimumDuration = max(
            Float(config.minSegmentDuration),
            Float(config.segmentationMinDurationOn)
        )
        ordered = ordered.filter {
            ($0.endTimeSeconds - $0.startTimeSeconds) >= minimumDuration
        }

        if config.embeddingExcludeOverlap {
            ordered = excludeOverlaps(in: ordered)
        }

        return ordered
    }

    private func chunkStartTime(
        for chunkIndex: Int,
        segmentation: SegmentationOutput
    ) -> Double {
        if segmentation.chunkOffsets.indices.contains(chunkIndex) {
            return segmentation.chunkOffsets[chunkIndex]
        } else {
            return Double(chunkIndex) * config.windowDuration
        }
    }
}
