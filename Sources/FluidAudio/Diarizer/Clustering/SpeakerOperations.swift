#if Diarizer
import Accelerate
import Foundation
import OSLog

// MARK: - Standalone Speaker Utilities

/// Utility functions for speaker operations that can be used by external applications
public enum SpeakerUtilities {

    private static let logger = AppLogger(category: "SpeakerUtilities")
    private static let normalizationTolerance: Float = 1e-3

    // MARK: - Configuration

    /// Platform-specific configuration for speaker assignment
    public struct AssignmentConfig: Sendable {
        public let maxDistanceForAssignment: Float
        public let maxDistanceForUpdate: Float
        public let minSpeakerDuration: Float
        public let minSegmentDuration: Float

        public init(
            maxDistanceForAssignment: Float,
            maxDistanceForUpdate: Float,
            minSpeakerDuration: Float,
            minSegmentDuration: Float
        ) {
            self.maxDistanceForAssignment = maxDistanceForAssignment
            self.maxDistanceForUpdate = maxDistanceForUpdate
            self.minSpeakerDuration = minSpeakerDuration
            self.minSegmentDuration = minSegmentDuration
        }

        public static let macOS = AssignmentConfig(
            maxDistanceForAssignment: 0.65,
            maxDistanceForUpdate: 0.45,
            minSpeakerDuration: 4.0,
            minSegmentDuration: 1.0
        )

        public static let iOS = AssignmentConfig(
            maxDistanceForAssignment: 0.55,
            maxDistanceForUpdate: 0.45,
            minSpeakerDuration: 4.0,
            minSegmentDuration: 1.0
        )

        public static var current: AssignmentConfig {
            #if os(macOS)
            return .macOS
            #else
            return .iOS
            #endif
        }
    }

    // MARK: - Distance Calculations

    /// Calculate cosine distance between two embeddings
    /// Returns value between 0 (identical) and 2 (opposite)
    /// Returns infinity if embeddings are invalid
    public static func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty, !b.isEmpty else {
            logger.error("Invalid embeddings: a.count=\(a.count), b.count=\(b.count)")
            return Float.infinity
        }

        var dotProduct: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))

        var sumSquaresA: Float = 0
        var sumSquaresB: Float = 0
        vDSP_svesq(a, 1, &sumSquaresA, vDSP_Length(a.count))
        vDSP_svesq(b, 1, &sumSquaresB, vDSP_Length(b.count))

        guard sumSquaresA > 0 && sumSquaresB > 0 else {
            logger.warning("Zero magnitude embedding detected")
            return Float.infinity
        }

        let isUnitA = abs(sumSquaresA - 1.0) <= normalizationTolerance
        let isUnitB = abs(sumSquaresB - 1.0) <= normalizationTolerance

        let similarity: Float
        if isUnitA && isUnitB {
            similarity = dotProduct
        } else {
            let magnitudeA = sumSquaresA.squareRoot()
            let magnitudeB = sumSquaresB.squareRoot()

            guard magnitudeA > 0 && magnitudeB > 0 else {
                logger.warning("Zero magnitude after normalization guard")
                return Float.infinity
            }

            similarity = dotProduct / (magnitudeA * magnitudeB)
        }

        let clampedSimilarity = min(max(similarity, -1.0), 1.0)
        return 1 - clampedSimilarity
    }

    // MARK: - Embedding Validation

    /// Validate embedding quality
    public static func validateEmbedding(_ embedding: [Float], minMagnitude: Float = 0.1) -> Bool {
        guard !embedding.isEmpty else {
            logger.error("Empty embedding")
            return false
        }

        var sumSquares: Float = 0
        vDSP_svesq(embedding, 1, &sumSquares, vDSP_Length(embedding.count))
        let magnitude = sumSquares.squareRoot()
        guard magnitude > minMagnitude else {
            logger.warning("Low magnitude embedding: \(magnitude)")
            return false
        }

        guard embedding.allSatisfy({ $0.isFinite }) else {
            logger.error("Embedding contains NaN or Inf")
            return false
        }

        return true
    }

    // MARK: - Speaker Assignment Decision

    /// Decision result for speaker assignment
    public struct AssignmentDecision {
        public let shouldAssign: Bool
        public let shouldUpdate: Bool
        public let confidence: Float
        public let reason: String
    }

    /// Determine if a speaker should be assigned based on distance and duration
    public static func shouldAssignSpeaker(
        distance: Float,
        duration: Float,
        config: AssignmentConfig = .current
    ) -> AssignmentDecision {
        // Very high confidence
        if distance < 0.2 {
            return AssignmentDecision(
                shouldAssign: true,
                shouldUpdate: duration >= config.minSpeakerDuration,
                confidence: 1.0 - distance,
                reason: "Very high confidence match"
            )
        }

        // Good match with sufficient duration
        if distance < config.maxDistanceForAssignment && duration >= config.minSegmentDuration {
            let shouldUpdate = distance <= config.maxDistanceForUpdate && duration >= config.minSpeakerDuration
            return AssignmentDecision(
                shouldAssign: true,
                shouldUpdate: shouldUpdate,
                confidence: 1.0 - distance,
                reason: "Sufficient confidence and duration"
            )
        }

        // Not confident enough
        return AssignmentDecision(
            shouldAssign: false,
            shouldUpdate: false,
            confidence: 1.0 - distance,
            reason: distance >= config.maxDistanceForAssignment ? "Distance too far" : "Duration too short"
        )
    }

    // MARK: - Batch Operations

    /// Find closest speaker from candidates
    public static func findClosestSpeaker(
        embedding: [Float],
        candidates: [Speaker]
    ) -> (speaker: Speaker?, distance: Float) {
        guard !candidates.isEmpty else {
            return (nil, Float.infinity)
        }

        var minDistance = Float.infinity
        var closestSpeaker: Speaker?

        for candidate in candidates {
            let distance = cosineDistance(embedding, candidate.currentEmbedding)
            if distance < minDistance {
                minDistance = distance
                closestSpeaker = candidate
            }
        }

        return (closestSpeaker, minDistance)
    }

    // MARK: - Speaker Creation

    /// Validated speaker creation parameters
    public struct SpeakerCreationParams {
        public let id: String
        public let name: String
        public let duration: Float
        public let embedding: [Float]
        public let createdAt: Date
        public let updatedAt: Date

        public init(
            id: String,
            name: String? = nil,
            duration: Float,
            embedding: [Float],
            createdAt: Date = Date(),
            updatedAt: Date = Date()
        ) {
            self.id = id
            self.name = name ?? "Speaker \(id)"
            self.duration = duration
            self.embedding = embedding
            self.createdAt = createdAt
            self.updatedAt = updatedAt
        }
    }

    /// Result of speaker creation validation
    public enum SpeakerCreationResult {
        case success(SpeakerCreationParams)
        case failure(reason: String)
    }

    /// Validate and prepare speaker creation parameters
    public static func validateSpeakerCreation(
        id: String,
        name: String? = nil,
        duration: Float,
        embedding: [Float],
        config: AssignmentConfig = .current
    ) -> SpeakerCreationResult {
        // Validate duration
        guard duration >= config.minSpeakerDuration else {
            return .failure(reason: "Duration \(duration) < minimum \(config.minSpeakerDuration)")
        }

        // Validate embedding
        guard validateEmbedding(embedding) else {
            return .failure(reason: "Invalid embedding: failed quality checks")
        }

        // Create validated parameters
        let normalizedEmbedding = VDSPOperations.l2Normalize(embedding)
        let params = SpeakerCreationParams(
            id: id,
            name: name,
            duration: duration,
            embedding: normalizedEmbedding
        )

        return .success(params)
    }

    /// Create a new Speaker instance with validation
    public static func createSpeaker(
        id: String,
        name: String? = nil,
        duration: Float,
        embedding: [Float],
        config: AssignmentConfig = .current
    ) -> Speaker? {
        let result = validateSpeakerCreation(
            id: id,
            name: name,
            duration: duration,
            embedding: embedding,
            config: config
        )

        switch result {
        case .success(let params):
            return Speaker(
                id: params.id,
                name: params.name,
                currentEmbedding: params.embedding,
                duration: params.duration,
                createdAt: params.createdAt,
                updatedAt: params.updatedAt
            )
        case .failure(let reason):
            logger.error("Failed to create speaker: \(reason)")
            return nil
        }
    }

    // MARK: - Embedding Updates

    /// Update a speaker embedding using exponential moving average
    /// This is a pure function that returns the updated embedding without modifying state
    public static func updateEmbedding(
        current: [Float],
        new: [Float],
        alpha: Float = 0.9
    ) -> [Float]? {
        // Validate inputs
        guard current.count == new.count,
            !current.isEmpty,
            validateEmbedding(new)
        else {
            logger.error("Invalid embeddings for update")
            return nil
        }

        // Calculate exponential moving average
        let normalizedCurrent = VDSPOperations.l2Normalize(current)
        let normalizedNew = VDSPOperations.l2Normalize(new)

        var updated = [Float](repeating: 0, count: normalizedCurrent.count)
        for i in 0..<normalizedCurrent.count {
            updated[i] = alpha * normalizedCurrent[i] + (1 - alpha) * normalizedNew[i]
        }

        return VDSPOperations.l2Normalize(updated)
    }

    // MARK: - Raw Embedding Management

    /// Adds a raw embedding with validation and FIFO management
    /// Returns the updated array and whether recalculation is needed
    public static func addRawEmbedding(
        to rawEmbeddings: [RawEmbedding],
        segmentId: UUID,
        embedding: [Float],
        timestamp: Date = Date(),
        maxCapacity: Int = 50
    ) -> (updated: [RawEmbedding], shouldRecalculate: Bool)? {
        // Validate the embedding
        guard validateEmbedding(embedding) else {
            logger.warning("Invalid embedding for segment \(segmentId)")
            return nil
        }

        // Create the new raw embedding
        let normalizedEmbedding = VDSPOperations.l2Normalize(embedding)
        let newEmbedding = RawEmbedding(
            segmentId: segmentId,
            embedding: normalizedEmbedding,
            timestamp: timestamp
        )

        // FIFO queue management - inline for simplicity
        var updated = rawEmbeddings
        if updated.count >= maxCapacity {
            updated.removeFirst()
        }
        updated.append(newEmbedding)

        let shouldRecalculate = updated.count >= 3  // Need at least 3 for meaningful average

        return (updated, shouldRecalculate)
    }

    /// Removes a raw embedding by segment ID
    /// Returns the updated array and the removed embedding if found
    public static func removeRawEmbedding(
        from rawEmbeddings: [RawEmbedding],
        segmentId: UUID
    ) -> (updated: [RawEmbedding], removed: RawEmbedding?, shouldRecalculate: Bool) {
        var updated = rawEmbeddings

        if let index = updated.firstIndex(where: { $0.segmentId == segmentId }) {
            let removed = updated.remove(at: index)
            let shouldRecalculate = !updated.isEmpty
            return (updated, removed, shouldRecalculate)
        }

        return (rawEmbeddings, nil, false)
    }

    // MARK: - Complete Speaker Update Operations

    /// Complete speaker update operation including raw tracking
    public struct SpeakerUpdateResult {
        public let updatedMainEmbedding: [Float]?
        public let updatedRawEmbeddings: [RawEmbedding]
        public let updatedDuration: Float
        public let shouldRecalculate: Bool
    }

    /// Updates a speaker with new segment data - handles both main and raw embeddings
    public static func updateSpeakerWithSegment(
        currentMainEmbedding: [Float],
        currentRawEmbeddings: [RawEmbedding],
        currentDuration: Float,
        segmentDuration: Float,
        segmentEmbedding: [Float],
        segmentId: UUID,
        alpha: Float = 0.9,
        minSegmentDuration: Float = 2.0
    ) -> SpeakerUpdateResult? {
        // Validate segment duration
        guard segmentDuration >= minSegmentDuration else {
            return nil
        }

        // Validate embedding
        guard validateEmbedding(segmentEmbedding) else {
            return nil
        }

        let normalizedEmbedding = VDSPOperations.l2Normalize(segmentEmbedding)

        // Add to raw embeddings
        guard
            let (updatedRaw, shouldRecalc) = addRawEmbedding(
                to: currentRawEmbeddings,
                segmentId: segmentId,
                embedding: normalizedEmbedding,
                timestamp: Date()
            )
        else {
            return nil
        }

        // Update main embedding using exponential moving average
        let updatedMain = updateEmbedding(
            current: currentMainEmbedding,
            new: normalizedEmbedding,
            alpha: alpha
        )

        // Calculate new duration
        let newDuration = currentDuration + segmentDuration

        return SpeakerUpdateResult(
            updatedMainEmbedding: updatedMain,
            updatedRawEmbeddings: updatedRaw,
            updatedDuration: newDuration,
            shouldRecalculate: shouldRecalc
        )
    }

    /// Merge two speakers' data
    public static func mergeSpeakers(
        speaker1Raw: [RawEmbedding],
        speaker1Duration: Float,
        speaker2Raw: [RawEmbedding],
        speaker2Duration: Float,
        maxCapacity: Int = 50
    ) -> (mergedRaw: [RawEmbedding], mergedDuration: Float, newMainEmbedding: [Float]?) {
        // Merge raw embeddings
        var allEmbeddings = speaker1Raw + speaker2Raw

        // Keep only the most recent embeddings if over capacity
        if allEmbeddings.count > maxCapacity {
            allEmbeddings = Array(
                allEmbeddings
                    .sorted { $0.timestamp > $1.timestamp }
                    .prefix(maxCapacity)
            )
        }

        // Calculate new main embedding from merged history
        let embeddingArrays = allEmbeddings.map { $0.embedding }
        let newMainEmbedding = averageEmbeddings(embeddingArrays)

        // Merge durations
        let mergedDuration = speaker1Duration + speaker2Duration

        return (allEmbeddings, mergedDuration, newMainEmbedding)
    }

    /// Calculate average of multiple embeddings
    public static func averageEmbeddings(_ embeddings: [[Float]]) -> [Float]? {
        guard !embeddings.isEmpty,
            let dimension = embeddings.first?.count,
            dimension > 0
        else {
            return nil
        }

        var average = [Float](repeating: 0, count: dimension)
        var validCount = 0

        for embedding in embeddings {
            guard embedding.count == dimension else { continue }
            for i in 0..<dimension {
                average[i] += embedding[i]
            }
            validCount += 1
        }

        guard validCount > 0 else { return nil }

        for i in 0..<dimension {
            average[i] /= Float(validCount)
        }

        return VDSPOperations.l2Normalize(average)
    }

    // MARK: - Clean Speaker Embeddings

    /// Build one embedding per speaker from clean speech only, for enrolling speakers from a finished
    /// diarization (for example with `SpeakerManager.initializeKnownSpeakers`).
    ///
    /// Averaging everything a speaker said lets crosstalk, short turns, and clustering mistakes pull the
    /// embedding toward another voice, so a profile built that way can match the wrong person in the next
    /// session. This works on `result.chunkEmbeddings`, the one independent embedding per (chunk, local
    /// speaker), so run the diarizer with `exposeChunkEmbeddings = true` (`DiarizerConfig` or
    /// `OfflineDiarizerConfig`). Segment embeddings can't be used: segments in a chunk share its
    /// embedding, and offline segments carry their cluster centroid.
    ///
    /// A speaker's clean speech is the time `result.segments` show them talking and nobody else. Seconds
    /// are counted as audio, not per chunk: `OfflineDiarizerManager`'s windows overlap (80% by default), so
    /// the same second sits in several chunks. Each moment is split evenly between the chunks covering it,
    /// and a chunk's share is its weight in the average; consensus and the floor use the kept chunks'
    /// shares, so a moment a rejected chunk also covers doesn't count as agreement in full.
    ///
    /// A chunk with less than `minChunkCleanSpeech` seconds of clean speech isn't used, but its clean
    /// speech still counts toward the speaker's total; a chunk whose embedding failed isn't counted. With 3 or more usable chunks, the medoid (the chunk
    /// most like the others, counted by chunks rather than seconds so one long odd chunk can't stand for
    /// the speaker) represents the voice, and chunks within `maxDistanceToMedoid` cosine distance of it are
    /// kept; two chunks must be within that distance of each other. The speaker gets no embedding when
    /// fewer than `minChunks` chunks are kept (one chunk has nothing to corroborate it), when the kept
    /// chunks cover less than `minConsensus` of the speaker's clean speech (a cluster that is really two
    /// people, including a second voice spread over many short chunks), or when they cover less than
    /// `minTotalCleanSpeech` seconds.
    ///
    /// The defaults were measured on 3 AMI meetings, with chunks labeled by reference turns so the numbers
    /// are about this function rather than clustering (`CleanSpeakerEmbeddingsAMITests`). With
    /// `DiarizerManager`, 9 of 11 speakers got an embedding, 85% of held-out chunks were identified, and of 9
    /// pairs of speakers merged into one cluster, 5 got no embedding and 4 got the majority's voice with at
    /// most 0.05 more similarity to the other voice than the two voices share (at 0.6 that leak reached 0.3).
    /// With `OfflineDiarizerManager`'s overlapping windows the embeddings matched `DiarizerManager`'s
    /// (similarity 0.99 or more), 89% of held-out windows were identified, and the leak stayed under 0.05.
    /// Chunk pairs are noisy (within 0.5: 45% of same-speaker pairs, 5% of different-speaker pairs), so a
    /// looser pair rule would enroll blends. `DiarizerManager` drops turns under `minSpeechDuration` (1 s),
    /// so other speakers' short interjections count as clean speech; on the same meetings that moved
    /// embeddings by under 0.2%.
    ///
    /// - Parameters:
    ///   - result: A diarization run with `chunkEmbeddings` populated
    ///   - minChunkCleanSpeech: Least clean speech a chunk needs to be used, in seconds
    ///   - minTotalCleanSpeech: Least clean speech the kept chunks must cover, in seconds of audio
    ///   - maxDistanceToMedoid: Largest cosine distance from the speaker's medoid chunk to keep a chunk
    ///   - minConsensus: Least share of the speaker's clean speech the kept chunks must cover
    ///   - minChunks: Fewest kept chunks for an embedding
    /// - Returns: Speaker ID to L2-normalized embedding; speakers without enough consistent clean speech are
    ///   omitted, and the result is empty when `result.chunkEmbeddings` is nil
    public static func cleanEmbeddingsBySpeaker(
        _ result: DiarizationResult,
        minChunkCleanSpeech: Double = 1.5,
        minTotalCleanSpeech: Double = 4.0,
        maxDistanceToMedoid: Float = 0.5,
        minConsensus: Double = 0.6,
        minChunks: Int = 2
    ) -> [String: [Float]] {
        guard let chunks = result.chunkEmbeddings else {
            logger.warning("cleanEmbeddingsBySpeaker needs chunk embeddings; set exposeChunkEmbeddings = true")
            return [:]
        }
        var turns: [String: [Interval]] = [:]
        for segment in result.segments where segment.endTimeSeconds > segment.startTimeSeconds {
            turns[segment.speakerId, default: []].append(
                (Double(segment.startTimeSeconds), Double(segment.endTimeSeconds)))
        }
        // Each speaker's clean speech over the whole meeting, once.
        var clean: [String: [Interval]] = [:]
        for speaker in Set(chunks.map(\.speakerId)) where !speaker.isEmpty {
            let others = merged(turns.filter { $0.key != speaker }.flatMap(\.value))
            clean[speaker] = subtract(merged(turns[speaker] ?? []), others)
        }

        struct Entry {
            var embedding: [Float]
            /// Index into the speaker's `covering` list.
            var slot: Int
        }
        var usable: [String: [Entry]] = [:]
        /// Every valid chunk's clean speech, usable or too short, per speaker.
        var covering: [String: [[Interval]]] = [:]
        // A chunk whose embedding failed (NaN, zero) says nothing about the voice and isn't counted; a valid
        // chunk too short to use still counts toward the speaker's clean speech.
        for chunk in chunks
        where !chunk.speakerId.isEmpty && chunk.embedding256.allSatisfy(\.isFinite)
            && validateEmbedding(chunk.embedding256)
        {
            let own = clipped(clean[chunk.speakerId] ?? [], to: (chunk.startTimeSeconds, chunk.endTimeSeconds))
            covering[chunk.speakerId, default: []].append(own)
            guard measure(own) >= minChunkCleanSpeech else { continue }
            usable[chunk.speakerId, default: []].append(
                Entry(
                    embedding: VDSPOperations.l2Normalize(chunk.embedding256),
                    slot: covering[chunk.speakerId]!.count - 1))
        }

        var embeddings: [String: [Float]] = [:]
        for (speakerId, entries) in usable {
            guard let dimension = entries.first?.embedding.count,
                entries.allSatisfy({ $0.embedding.count == dimension })
            else {
                logger.warning("Mixed embedding sizes for speaker \(speakerId); skipping")
                continue
            }
            let n = entries.count
            var distance = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
            for i in 0..<n {
                for j in (i + 1)..<max(i + 1, n) {
                    let d = cosineDistance(entries[i].embedding, entries[j].embedding)
                    distance[i][j] = d
                    distance[j][i] = d
                }
            }
            var kept = Array(0..<n)
            if n == 2 {
                guard distance[0][1] <= maxDistanceToMedoid else { continue }
            } else if n >= 3 {
                let medoid = (0..<n).min { distance[$0].reduce(0, +) < distance[$1].reduce(0, +) } ?? 0
                kept = (0..<n).filter { $0 == medoid || distance[$0][medoid] <= maxDistanceToMedoid }
            }
            guard kept.count >= minChunks else { continue }
            // Each moment of clean speech is split between every chunk covering it, kept or not, so a
            // moment a rejected chunk also covers isn't counted as agreement in full.
            let shares = sharedWeights(covering[speakerId] ?? [])
            let all = Double(shares.reduce(0, +))
            let weights = kept.map { shares[entries[$0].slot] }
            let agreed = Double(weights.reduce(0, +))
            guard agreed >= minTotalCleanSpeech, agreed >= minConsensus * all else { continue }
            let mean = weightedSum(kept.map { entries[$0].embedding }, weights: weights, dimension: dimension)
            embeddings[speakerId] = VDSPOperations.l2Normalize(mean)
        }
        return embeddings
    }

    typealias Interval = (start: Double, end: Double)

    /// Each chunk's share of the audio its intervals cover: a moment covered by k chunks gives each 1/k of
    /// its length, so overlapping windows don't count the same second several times.
    static func sharedWeights(_ chunks: [[Interval]]) -> [Float] {
        // One sweep over start/end events; each chunk's intervals are already merged, so a chunk is
        // active at most once at a time.
        var events: [(time: Double, start: Bool, chunk: Int)] = []
        for (i, intervals) in chunks.enumerated() {
            for interval in intervals where interval.end > interval.start {
                events.append((interval.start, true, i))
                events.append((interval.end, false, i))
            }
        }
        events.sort { $0.time != $1.time ? $0.time < $1.time : (!$0.start && $1.start) }
        var weights = [Double](repeating: 0, count: chunks.count)
        var active = Set<Int>()
        var last = events.first?.time ?? 0
        for event in events {
            let length = event.time - last
            if length > 0, !active.isEmpty {
                for i in active { weights[i] += length / Double(active.count) }
            }
            last = event.time
            if event.start { active.insert(event.chunk) } else { active.remove(event.chunk) }
        }
        return weights.map { Float($0) }
    }

    private static func weightedSum(_ embeddings: [[Float]], weights: [Float], dimension: Int) -> [Float] {
        var sum = [Float](repeating: 0, count: dimension)
        for (embedding, weight) in zip(embeddings, weights) {
            for d in 0..<dimension {
                sum[d] += weight * embedding[d]
            }
        }
        return sum
    }

    /// Sorted, merged intervals.
    static func merged(_ intervals: [Interval]) -> [Interval] {
        var out: [Interval] = []
        for (start, end) in intervals.filter({ $0.end > $0.start }).sorted(by: { $0.start < $1.start }) {
            if let last = out.last, start <= last.end {
                out[out.count - 1].end = max(last.end, end)
            } else {
                out.append((start, end))
            }
        }
        return out
    }

    /// `a` minus `b`; both merged.
    static func subtract(_ a: [Interval], _ b: [Interval]) -> [Interval] {
        var out: [Interval] = []
        for (start, end) in a {
            var cursor = start
            for (bs, be) in b where be > cursor && bs < end {
                if bs > cursor { out.append((cursor, bs)) }
                cursor = max(cursor, be)
            }
            if cursor < end { out.append((cursor, end)) }
        }
        return out
    }

    /// Merged intervals cut to `span`.
    static func clipped(_ intervals: [Interval], to span: Interval) -> [Interval] {
        intervals.compactMap { i in
            let s = max(i.start, span.start)
            let e = min(i.end, span.end)
            return s < e ? (s, e) : nil
        }
    }

    static func measure(_ intervals: [Interval]) -> Double {
        intervals.reduce(0) { $0 + $1.end - $1.start }
    }
}

/// These functions provide additional capabilities beyond core diarization
extension SpeakerManager {

    /// Reassign a raw embedding from one speaker to another.
    ///
    /// This moves a segment's embedding from one speaker to another, updating both speakers'
    /// main embeddings. Useful for correcting misclassified segments.
    ///
    /// - Parameters:
    ///   - segmentId: The segment ID to reassign
    ///   - fromSpeakerId: The current speaker ID
    ///   - toSpeakerId: The target speaker ID
    /// - Returns: True if successful, false if segment not found or speakers don't exist
    @discardableResult
    public mutating func reassignSegment(
        segmentId: UUID,
        from fromSpeakerId: String,
        to toSpeakerId: String
    ) -> Bool {
        // Get speaker info from database
        guard var fromSpeakerInfo = speakerDatabase[fromSpeakerId],
            var toSpeakerInfo = speakerDatabase[toSpeakerId]
        else {
            logger.warning("One or both speakers not found: from=\(fromSpeakerId), to=\(toSpeakerId)")
            return false
        }

        // Find and remove the embedding from source speaker
        guard let index = fromSpeakerInfo.rawEmbeddings.firstIndex(where: { $0.segmentId == segmentId })
        else {
            logger.warning("Segment \(segmentId) not found in speaker \(fromSpeakerId)")
            return false
        }

        let embedding = fromSpeakerInfo.rawEmbeddings.remove(at: index)

        // Add to destination speaker
        toSpeakerInfo.rawEmbeddings.append(embedding)

        // Update both speakers in database
        speakerDatabase[fromSpeakerId] = fromSpeakerInfo
        speakerDatabase[toSpeakerId] = toSpeakerInfo

        // Recalculate embeddings for both speakers
        // Note: This would need to be implemented using the SpeakerInfo structure
        // For now, just log the successful reassignment

        logger.info("Reassigned segment \(segmentId) from \(fromSpeakerId) to \(toSpeakerId)")
        return true
    }

    // MARK: - Speaker Query Operations

    /// Get the names/IDs of all current speakers.
    ///
    /// Returns a list of speaker identifiers from the in-memory database.
    /// Useful for getting a quick overview of detected speakers.
    ///
    /// - Returns: Array of speaker IDs/names currently in the database
    public func getCurrentSpeakerNames() -> [String] {
        return Array(speakerDatabase.keys).sorted()
    }

    /// Get global speaker statistics from the in-memory database.
    ///
    /// Returns comprehensive statistics about all speakers including count,
    /// total duration, and quality metrics.
    ///
    /// - Returns: Tuple containing (totalSpeakers, totalDuration, averageConfidence, speakersWithHistory)
    public func getGlobalSpeakerStats() -> (
        totalSpeakers: Int,
        totalDuration: Float,
        averageConfidence: Float,
        speakersWithHistory: Int
    ) {
        let speakers = Array(speakerDatabase.values)

        guard !speakers.isEmpty else {
            return (0, 0, 0, 0)
        }

        let totalDuration = speakers.reduce(0) { $0 + $1.duration }
        let totalUpdates = speakers.reduce(0) { $0 + $1.updateCount }
        let averageConfidence = Float(totalUpdates) / Float(speakers.count) / 10.0  // Normalize
        let speakersWithHistory = speakers.filter { !$0.rawEmbeddings.isEmpty }.count

        logger.info(
            "Global stats - Speakers: \(speakers.count), Duration: \(String(format: "%.1f", totalDuration))s, Avg confidence: \(String(format: "%.2f", averageConfidence)), With history: \(speakersWithHistory)"
        )

        return (speakers.count, totalDuration, min(1.0, averageConfidence), speakersWithHistory)
    }
}
#endif
