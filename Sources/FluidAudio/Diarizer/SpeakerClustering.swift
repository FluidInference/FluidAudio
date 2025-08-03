import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
internal struct SpeakerClustering {

    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "Clustering")
    private let config: DiarizerConfig

    init(config: DiarizerConfig) {
        self.config = config
    }

    func calculateSpeakerActivities(_ binarizedSegments: [[[Float]]]) -> [Float] {
        let numSpeakers = binarizedSegments[0][0].count
        let numFrames = binarizedSegments[0].count
        var activities: [Float] = Array(repeating: 0.0, count: numSpeakers)

        for speakerIndex in 0..<numSpeakers {
            for frameIndex in 0..<numFrames {
                activities[speakerIndex] += binarizedSegments[0][frameIndex][speakerIndex]
            }
        }

        return activities
    }

    func assignSpeaker(embedding: [Float], speakerDB: inout [String: [Float]]) -> String {
        if speakerDB.isEmpty {
            let speakerId = "Speaker 1"
            speakerDB[speakerId] = embedding
            return speakerId
        }

        var minDistance: Float = Float.greatestFiniteMagnitude
        var identifiedSpeaker: String? = nil
        var allDistances: [(String, Float)] = []

        for (speakerId, refEmbedding) in speakerDB {
            let distance = cosineDistance(embedding, refEmbedding)
            allDistances.append((speakerId, distance))
            if distance < minDistance {
                minDistance = distance
                identifiedSpeaker = speakerId
            }
        }

        if let bestSpeaker = identifiedSpeaker {
            if minDistance > config.clusteringThreshold {
                let newSpeakerId = "Speaker \(speakerDB.count + 1)"
                speakerDB[newSpeakerId] = embedding
                return newSpeakerId
            } else {
                updateSpeakerEmbedding(bestSpeaker, embedding, speakerDB: &speakerDB)
                return bestSpeaker
            }
        }

        return "Unknown"
    }

    func updateSpeakerEmbedding(
        _ speakerId: String, _ newEmbedding: [Float], speakerDB: inout [String: [Float]],
        alpha: Float = 0.9
    ) {
        guard var oldEmbedding = speakerDB[speakerId] else { return }

        for i in 0..<oldEmbedding.count {
            oldEmbedding[i] = alpha * oldEmbedding[i] + (1 - alpha) * newEmbedding[i]
        }
        speakerDB[speakerId] = oldEmbedding
    }

    func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else {
            if config.debugMode {
                logger.debug(
                    "🔍 CLUSTERING DEBUG: Invalid embeddings for distance calculation - a.count: \(a.count), b.count: \(b.count)"
                )
            }
            return Float.infinity
        }

        var dotProduct: Float = 0
        var magnitudeA: Float = 0
        var magnitudeB: Float = 0

        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            magnitudeA += a[i] * a[i]
            magnitudeB += b[i] * b[i]
        }

        magnitudeA = sqrt(magnitudeA)
        magnitudeB = sqrt(magnitudeB)

        guard magnitudeA > 0 && magnitudeB > 0 else {
            logger.warning(
                "🔍 CLUSTERING DEBUG: Zero magnitude embedding detected - magnitudeA: \(magnitudeA), magnitudeB: \(magnitudeB)"
            )
            return Float.infinity
        }

        let similarity = dotProduct / (magnitudeA * magnitudeB)
        let distance = 1 - similarity

        if config.debugMode {
            logger.debug(
                "🔍 CLUSTERING DEBUG: cosineDistance - similarity: \(String(format: "%.4f", similarity)), distance: \(String(format: "%.4f", distance)), magA: \(String(format: "%.4f", magnitudeA)), magB: \(String(format: "%.4f", magnitudeB))"
            )
        }

        return distance
    }

    func createTimedSegments(
        binarizedSegments: [[[Float]]],
        slidingWindow: SlidingWindow,
        embeddings: [[Float]],
        speakerLabels: [String],
        speakerActivities: [Float]
    ) -> [TimedSpeakerSegment] {
        let segmentation = binarizedSegments[0]
        let numFrames = segmentation.count
        var segments: [TimedSpeakerSegment] = []

        var frameSpeakers: [Int] = []
        for frame in segmentation {
            if let maxIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) {
                frameSpeakers.append(maxIdx)
            } else {
                frameSpeakers.append(0)
            }
        }

        var currentSpeaker = frameSpeakers[0]
        var startFrame = 0

        for i in 1..<numFrames {
            if frameSpeakers[i] != currentSpeaker {
                if let segment = createSegmentIfValid(
                    speakerIndex: currentSpeaker,
                    startFrame: startFrame,
                    endFrame: i,
                    slidingWindow: slidingWindow,
                    embeddings: embeddings,
                    speakerLabels: speakerLabels,
                    speakerActivities: speakerActivities
                ) {
                    segments.append(segment)
                }
                currentSpeaker = frameSpeakers[i]
                startFrame = i
            }
        }

        if let segment = createSegmentIfValid(
            speakerIndex: currentSpeaker,
            startFrame: startFrame,
            endFrame: numFrames,
            slidingWindow: slidingWindow,
            embeddings: embeddings,
            speakerLabels: speakerLabels,
            speakerActivities: speakerActivities
        ) {
            segments.append(segment)
        }

        return segments
    }

    private func createSegmentIfValid(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        slidingWindow: SlidingWindow,
        embeddings: [[Float]],
        speakerLabels: [String],
        speakerActivities: [Float]
    ) -> TimedSpeakerSegment? {
        guard speakerIndex < speakerLabels.count,
            !speakerLabels[speakerIndex].isEmpty,
            speakerIndex < embeddings.count
        else {
            return nil
        }

        let startTime = slidingWindow.time(forFrame: startFrame)
        let endTime = slidingWindow.time(forFrame: endFrame)
        let duration = endTime - startTime

        if Float(duration) < config.minDurationOn {
            return nil
        }

        let embedding = embeddings[speakerIndex]
        let activity = speakerActivities[speakerIndex]
        let quality =
            calculateEmbeddingQuality(embedding) * (activity / Float(endFrame - startFrame))

        return TimedSpeakerSegment(
            speakerId: speakerLabels[speakerIndex],
            embedding: embedding,
            startTimeSeconds: Float(startTime),
            endTimeSeconds: Float(endTime),
            qualityScore: quality
        )
    }

    private func calculateEmbeddingQuality(_ embedding: [Float]) -> Float {
        let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
        return min(1.0, magnitude / 10.0)
    }
}
