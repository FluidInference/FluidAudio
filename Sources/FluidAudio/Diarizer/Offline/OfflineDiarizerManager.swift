import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public final class OfflineDiarizerManager {
    private let logger = AppLogger(category: "OfflineDiarizer")
    private let config: OfflineDiarizerConfig
    private let cleanFrameRatioThreshold = 0.2
    private let presenceThreshold: Float = 0.5

    private var models: OfflineDiarizerModels?

    public init(config: OfflineDiarizerConfig = .default) {
        self.config = config
    }

    public func initialize(models: OfflineDiarizerModels) {
        self.models = models
        logger.info("Offline diarizer models initialized")
    }

    public func process(audio: [Float]) async throws -> DiarizationResult {
        try config.validate()
        guard let models else {
            throw OfflineDiarizationError.modelNotLoaded("offline-diarizer")
        }

        let totalStart = Date()

        let segmentationProcessor = OfflineSegmentationProcessor()
        let segmentationStart = Date()
        let segmentation = try segmentationProcessor.process(
            audioSamples: audio,
            segmentationModel: models.segmentationModel,
            config: config
        )
        let segmentationTime = Date().timeIntervalSince(segmentationStart)
        logger.debug("Segmentation completed in \(segmentationTime)s")

        let pldaTransform = PLDATransform(pldaRhoModel: models.pldaRhoModel, psi: models.pldaPsi)
        let embeddingExtractor = OfflineEmbeddingExtractor(
            embeddingModel: models.embeddingModel,
            pldaTransform: pldaTransform,
            config: config
        )

        let embeddingStart = Date()
        let timedEmbeddings = try embeddingExtractor.extractEmbeddings(
            audio: audio,
            segmentation: segmentation
        )
        let embeddingTime = Date().timeIntervalSince(embeddingStart)
        logger.debug("Embedding extraction produced \(timedEmbeddings.count) vectors in \(embeddingTime)s")

        guard !timedEmbeddings.isEmpty else {
            throw OfflineDiarizationError.noSpeechDetected
        }

        let embeddingFeatures = timedEmbeddings.map { $0.embedding256.map { Double($0) } }
        let rhoFeatures = timedEmbeddings.map { $0.rho128 }

        let clusteringStart = Date()
        let trainingIndices = selectTrainingEmbeddings(
            timedEmbeddings: timedEmbeddings,
            segmentation: segmentation
        )

        let trainingEmbeddings = trainingIndices.map { embeddingFeatures[$0] }
        let trainingRho = trainingIndices.map { rhoFeatures[$0] }

        logger.debug(
            "Clustering will use \(trainingEmbeddings.count)/\(timedEmbeddings.count) embeddings (clean frames ≥ 20%)"
        )

        let initialClusters: [Int]
        if trainingEmbeddings.count >= 2 {
            initialClusters = AHCClustering().cluster(
                embeddingFeatures: trainingEmbeddings,
                threshold: config.clusteringThreshold,
                minClusterSize: config.minClusterSize
            )
        } else {
            initialClusters = Array(repeating: 0, count: trainingEmbeddings.count)
        }

        let vbxOutput: VBxOutput
        if !trainingRho.isEmpty, !initialClusters.isEmpty {
            vbxOutput = VBxClustering(config: config, pldaTransform: pldaTransform).refine(
                rhoFeatures: trainingRho,
                initialClusters: initialClusters
            )
        } else {
            vbxOutput = VBxOutput(
                gamma: [],
                pi: [],
                hardClusters: [initialClusters],
                centroids: [],
                numClusters: initialClusters.max().map { $0 + 1 } ?? 0,
                elbos: []
            )
        }

        let centroidComputation = computeCentroids(
            trainingEmbeddings: trainingEmbeddings,
            vbxOutput: vbxOutput,
            initialClusters: initialClusters
        )
        var centroids = centroidComputation.centroids
        if centroids.isEmpty {
            centroids = computeFallbackCentroids(from: embeddingFeatures)
        }
        let assignments = assignEmbeddings(
            embeddingFeatures: embeddingFeatures,
            centroids: centroids
        )

        let chunkAssignments = buildChunkAssignments(
            segmentation: segmentation,
            timedEmbeddings: timedEmbeddings,
            assignments: assignments,
            clusterCount: centroids.count
        )

        let clusteringTime = Date().timeIntervalSince(clusteringStart)
        if !assignments.isEmpty {
            let histogram = assignments.reduce(into: [:]) { partialResult, cluster in
                partialResult[cluster, default: 0] += 1
            }
            logger.debug(
                "Clustering completed in \(clusteringTime)s with \(centroids.count) centroids (assignment histogram: \(histogram))"
            )
        } else {
            logger.debug("Clustering completed in \(clusteringTime)s with no assignments")
        }

        let reconstruction = OfflineReconstruction(config: config)
        let segments = reconstruction.buildSegments(
            segmentation: segmentation,
            hardClusters: chunkAssignments,
            centroids: centroids
        )

        let speakerDatabase = reconstruction.buildSpeakerDatabase(segments: segments)

        if let exportPath = config.embeddingExportPath {
            try exportEmbeddings(
                embeddings: timedEmbeddings,
                assignments: assignments,
                path: exportPath
            )
        }

        let totalProcessing = Date().timeIntervalSince(totalStart)
        let timings = PipelineTimings(
            modelDownloadSeconds: models.downloadDuration,
            modelCompilationSeconds: models.compilationDuration,
            audioLoadingSeconds: 0,
            segmentationSeconds: segmentationTime,
            embeddingExtractionSeconds: embeddingTime,
            speakerClusteringSeconds: clusteringTime,
            postProcessingSeconds: max(0, totalProcessing - segmentationTime - embeddingTime - clusteringTime)
        )

        return DiarizationResult(
            segments: segments,
            speakerDatabase: speakerDatabase,
            timings: timings
        )
    }

    private func selectTrainingEmbeddings(
        timedEmbeddings: [TimedEmbedding],
        segmentation: SegmentationOutput
    ) -> [Int] {
        var selected: [Int] = []
        selected.reserveCapacity(timedEmbeddings.count)

        for (index, embedding) in timedEmbeddings.enumerated() {
            let hasNaN = embedding.embedding256.contains { $0.isNaN || $0.isInfinite }
            if hasNaN {
                continue
            }

            guard
                let ratio = cleanFrameRatio(
                    for: embedding,
                    segmentation: segmentation
                )
            else {
                continue
            }

            if ratio >= cleanFrameRatioThreshold {
                selected.append(index)
            }
        }

        if selected.isEmpty {
            return Array(timedEmbeddings.indices)
        }

        return selected
    }

    private func cleanFrameRatio(
        for embedding: TimedEmbedding,
        segmentation: SegmentationOutput
    ) -> Double? {
        guard
            embedding.chunkIndex >= 0,
            embedding.chunkIndex < segmentation.speakerWeights.count
        else {
            return nil
        }

        let chunkWeights = segmentation.speakerWeights[embedding.chunkIndex]
        guard !chunkWeights.isEmpty else { return nil }

        var cleanFrames = 0
        var activeFrames = 0

        for frame in chunkWeights {
            guard embedding.speakerIndex < frame.count else { continue }
            let value = frame[embedding.speakerIndex]
            if value > presenceThreshold {
                activeFrames += 1
                var hasOverlap = false
                for (speakerIdx, other) in frame.enumerated() where speakerIdx != embedding.speakerIndex {
                    if other > presenceThreshold {
                        hasOverlap = true
                        break
                    }
                }
                if !hasOverlap {
                    cleanFrames += 1
                }
            }
        }

        guard !chunkWeights.isEmpty else { return nil }
        guard activeFrames > 0 else { return nil }
        return Double(cleanFrames) / Double(chunkWeights.count)
    }

    private func computeCentroids(
        trainingEmbeddings: [[Double]],
        vbxOutput: VBxOutput,
        initialClusters: [Int]
    ) -> (centroids: [[Double]], mapping: [Int: Int]) {
        guard let dimension = trainingEmbeddings.first?.count else {
            return ([], [:])
        }

        let epsilon = 1e-7
        let gamma = vbxOutput.gamma
        let pi = vbxOutput.pi

        if !gamma.isEmpty, !pi.isEmpty {
            let activeSpeakers = pi.enumerated().filter { $0.element > epsilon }
            if !activeSpeakers.isEmpty {
                var centroids: [[Double]] = []
                centroids.reserveCapacity(activeSpeakers.count)
                var mapping: [Int: Int] = [:]

                for (index, speaker) in activeSpeakers.enumerated() {
                    let speakerIdx = speaker.offset
                    mapping[speakerIdx] = index
                    var numerator = [Double](repeating: 0, count: dimension)
                    var denominator = 0.0

                    for frameIdx in 0..<min(gamma.count, trainingEmbeddings.count) {
                        let weight = gamma[frameIdx][speakerIdx]
                        guard weight > 0 else { continue }
                        denominator += weight
                        let embedding = trainingEmbeddings[frameIdx]
                        for dim in 0..<dimension {
                            numerator[dim] += embedding[dim] * weight
                        }
                    }

                    if denominator > 0 {
                        centroids.append(numerator.map { $0 / denominator })
                    } else {
                        centroids.append([Double](repeating: 0, count: dimension))
                    }
                }

                return (centroids, mapping)
            }
        }

        return computeCentroidsFromClusters(
            embeddings: trainingEmbeddings,
            clusters: initialClusters
        )
    }

    private func computeCentroidsFromClusters(
        embeddings: [[Double]],
        clusters: [Int]
    ) -> (centroids: [[Double]], mapping: [Int: Int]) {
        guard !embeddings.isEmpty, embeddings.count == clusters.count else {
            return ([], [:])
        }

        var grouped: [Int: (sum: [Double], count: Int)] = [:]
        for (embedding, cluster) in zip(embeddings, clusters) {
            if grouped[cluster] == nil {
                grouped[cluster] = (sum: [Double](repeating: 0, count: embedding.count), count: 0)
            }
            var entry = grouped[cluster]!
            for dim in 0..<embedding.count {
                entry.sum[dim] += embedding[dim]
            }
            entry.count += 1
            grouped[cluster] = entry
        }

        let sortedKeys = grouped.keys.sorted()
        var centroids: [[Double]] = []
        var mapping: [Int: Int] = [:]

        for (newIndex, key) in sortedKeys.enumerated() {
            mapping[key] = newIndex
            let entry = grouped[key]!
            if entry.count > 0 {
                centroids.append(entry.sum.map { $0 / Double(entry.count) })
            } else {
                centroids.append(entry.sum)
            }
        }

        return (centroids, mapping)
    }

    private func computeFallbackCentroids(from embeddings: [[Double]]) -> [[Double]] {
        guard let first = embeddings.first else { return [] }
        let average = embeddings.reduce(into: [Double](repeating: 0, count: first.count)) { partial, vector in
            for dim in 0..<vector.count {
                partial[dim] += vector[dim]
            }
        }.map { $0 / Double(embeddings.count) }
        return [average]
    }

    private func assignEmbeddings(
        embeddingFeatures: [[Double]],
        centroids: [[Double]]
    ) -> [Int] {
        guard !embeddingFeatures.isEmpty else { return [] }
        guard !centroids.isEmpty else {
            return Array(repeating: 0, count: embeddingFeatures.count)
        }

        let normalizedCentroids = centroids.map(normalize)
        return embeddingFeatures.map { embedding in
            let normalizedEmbedding = normalize(embedding)
            var bestIndex = 0
            var bestScore = -Double.infinity
            for (index, centroid) in normalizedCentroids.enumerated() {
                let score = dot(normalizedEmbedding, centroid)
                if score > bestScore {
                    bestScore = score
                    bestIndex = index
                }
            }
            return bestIndex
        }
    }

    private func normalize(_ vector: [Double]) -> [Double] {
        var sumSquares = 0.0
        for value in vector {
            sumSquares += value * value
        }
        if sumSquares <= 0 {
            return vector
        }
        let scale = 1.0 / sqrt(sumSquares)
        return vector.map { $0 * scale }
    }

    private func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhs.count else { return 0 }
        var result = 0.0
        for (l, r) in zip(lhs, rhs) {
            result += l * r
        }
        return result
    }

    private func buildChunkAssignments(
        segmentation: SegmentationOutput,
        timedEmbeddings: [TimedEmbedding],
        assignments: [Int],
        clusterCount: Int
    ) -> [[Int]] {
        var matrix = Array(
            repeating: Array(repeating: -2, count: segmentation.numSpeakers),
            count: segmentation.numChunks
        )

        for (embedding, cluster) in zip(timedEmbeddings, assignments) {
            guard
                embedding.chunkIndex >= 0,
                embedding.chunkIndex < matrix.count,
                embedding.speakerIndex >= 0,
                embedding.speakerIndex < matrix[embedding.chunkIndex].count,
                cluster >= 0,
                cluster < clusterCount
            else {
                continue
            }
            matrix[embedding.chunkIndex][embedding.speakerIndex] = cluster
        }

        return matrix
    }

    private func exportEmbeddings(
        embeddings: [TimedEmbedding],
        assignments: [Int],
        path: String
    ) throws {
        struct ExportPayload: Codable {
            let chunkIndex: Int
            let speakerIndex: Int
            let startFrame: Int
            let endFrame: Int
            let startTime: Double
            let endTime: Double
            let embedding256: [Float]
            let rho128: [Double]
            let cluster: Int
        }

        var payload: [ExportPayload] = []
        payload.reserveCapacity(embeddings.count)
        for (index, embedding) in embeddings.enumerated() {
            let cluster =
                assignments.indices.contains(index)
                ? assignments[index] : -1
            payload.append(
                ExportPayload(
                    chunkIndex: embedding.chunkIndex,
                    speakerIndex: embedding.speakerIndex,
                    startFrame: embedding.startFrame,
                    endFrame: embedding.endFrame,
                    startTime: embedding.startTime,
                    endTime: embedding.endTime,
                    embedding256: embedding.embedding256,
                    rho128: embedding.rho128,
                    cluster: cluster
                )
            )
        }

        let data = try JSONEncoder().encode(payload)
        let url = URL(fileURLWithPath: path)
        try data.write(to: url)
        logger.info("Exported \(payload.count) embeddings to \(path)")
    }
}
