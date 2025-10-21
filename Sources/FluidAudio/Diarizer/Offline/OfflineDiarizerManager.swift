import Accelerate
import CoreML
import Foundation
import OSLog

@available(macOS 14.0, iOS 17.0, *)
public final class OfflineDiarizerManager {
    private let logger = AppLogger(category: "OfflineDiarizer")
    private let config: OfflineDiarizerConfig

    private var models: OfflineDiarizerModels?

    public init(config: OfflineDiarizerConfig = .default) {
        self.config = config
    }

    public func initialize(models: OfflineDiarizerModels) {
        self.models = models
        logger.info("Offline diarizer models initialized")
    }

    /// Ensure offline diarizer models are available, downloading and compiling them when needed.
    /// - Parameters:
    ///   - directory: Custom cache directory. Defaults to `OfflineDiarizerModels.defaultModelsDirectory()`.
    ///   - configuration: Optional CoreML configuration to use during compilation.
    ///   - forceRedownload: When `true`, the cached repo is deleted before attempting to load.
    public func prepareModels(
        directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        forceRedownload: Bool = false
    ) async throws {
        if !forceRedownload, models != nil {
            logger.debug("Offline diarizer models already prepared; skipping load")
            return
        }

        let targetDirectory =
            directory?.standardizedFileURL
            ?? OfflineDiarizerModels.defaultModelsDirectory().standardizedFileURL

        if forceRedownload {
            do {
                try purgeDiarizerRepo(at: targetDirectory)
            } catch {
                logger.warning(
                    "Failed to purge diarizer cache during forced reload: \(error.localizedDescription)")
            }
        }

        do {
            let loadedModels = try await OfflineDiarizerModels.load(
                from: targetDirectory,
                configuration: configuration
            )
            initialize(models: loadedModels)
            await prewarmModelsIfNeeded(loadedModels)
            logger.info("Offline diarizer models loaded from \(targetDirectory.path)")
        } catch {
            logger.error(
                "Initial offline diarizer model load failed: \(error.localizedDescription)")
            logger.info("Attempting fallback download and compilation")

            do {
                try purgeDiarizerRepo(at: targetDirectory)
            } catch {
                logger.warning(
                    "Failed to remove cached diarizer repo before fallback: \(error.localizedDescription)")
            }

            do {
                let reloadedModels = try await OfflineDiarizerModels.load(
                    from: targetDirectory,
                    configuration: configuration
                )
                initialize(models: reloadedModels)
                await prewarmModelsIfNeeded(reloadedModels)

                let durationText = String(format: "%.2f", reloadedModels.compilationDuration)
                logger.info(
                    "Fallback download + compile completed in \(durationText)s at \(targetDirectory.path)")
            } catch {
                logger.error(
                    "Fallback offline diarizer model load failed: \(error.localizedDescription)")
                throw error
            }
        }
    }

    public func process(audio: [Float]) async throws -> DiarizationResult {
        try await process(
            audioSource: ArrayAudioSampleSource(samples: audio),
            audioLoadingSeconds: 0
        )
    }

    /// Process audio from a file URL using memory-mapped streaming for efficiency.
    /// Automatically converts the audio to the target sample rate and processes in chunks.
    /// - Parameter url: Path to the audio file
    /// - Returns: Diarization result with speaker segments
    public func process(_ url: URL) async throws -> DiarizationResult {
        let factory = StreamingAudioSourceFactory()
        let (source, loadDuration) = try factory.makeDiskBackedSource(
            from: url,
            targetSampleRate: config.segmentation.sampleRate
        )
        defer { source.cleanup() }

        return try await process(
            audioSource: source,
            audioLoadingSeconds: loadDuration
        )
    }

    public func process(
        audioSource: StreamingAudioSampleSource,
        audioLoadingSeconds: TimeInterval
    ) async throws -> DiarizationResult {
        try config.validate()
        if models == nil {
            try await prepareModels()
        }

        guard let models else {
            throw OfflineDiarizationError.modelNotLoaded("offline-diarizer")
        }

        let totalStart = Date()

        let streamPair = AsyncThrowingStream<SegmentationChunk, Error>.makeStream()
        let chunkStream = streamPair.stream
        let chunkContinuation = streamPair.continuation

        let segmentationTask = Task(priority: .userInitiated) { () throws -> (SegmentationOutput, TimeInterval) in
            let processor = OfflineSegmentationProcessor()
            let start = Date()
            do {
                let segmentation = try await processor.process(
                    audioSource: audioSource,
                    segmentationModel: models.segmentationModel,
                    config: config,
                    chunkHandler: { chunk in
                        switch chunkContinuation.yield(chunk) {
                        case .enqueued, .dropped:
                            return .continue
                        case .terminated:
                            return .stop
                        @unknown default:
                            return .stop
                        }
                    }
                )
                chunkContinuation.finish()
                return (segmentation, Date().timeIntervalSince(start))
            } catch {
                chunkContinuation.finish(throwing: error)
                throw error
            }
        }

        let embeddingTask = Task(priority: .userInitiated) { () throws -> ([TimedEmbedding], TimeInterval) in
            let extractor = OfflineEmbeddingExtractor(
                fbankModel: models.fbankModel,
                embeddingModel: models.embeddingModel,
                pldaTransform: PLDATransform(pldaRhoModel: models.pldaRhoModel, psi: models.pldaPsi),
                config: config
            )
            let start = Date()
            let embeddings = try await extractor.extractEmbeddings(
                audioSource: audioSource,
                segmentationStream: chunkStream
            )
            return (embeddings, Date().timeIntervalSince(start))
        }

        let segmentationResult: (SegmentationOutput, TimeInterval)
        let embeddingResult: ([TimedEmbedding], TimeInterval)
        do {
            async let awaitedSegmentation = segmentationTask.value
            async let awaitedEmbeddings = embeddingTask.value
            segmentationResult = try await awaitedSegmentation
            embeddingResult = try await awaitedEmbeddings
        } catch {
            segmentationTask.cancel()
            embeddingTask.cancel()
            chunkContinuation.finish(throwing: error)
            throw error
        }

        let (segmentation, segmentationTime) = segmentationResult
        logger.debug("Segmentation completed in \(segmentationTime)s (async)")

        let (timedEmbeddings, embeddingTime) = embeddingResult
        logger.debug("Embedding extraction produced \(timedEmbeddings.count) vectors in \(embeddingTime)s (async)")

        let pldaTransform = PLDATransform(pldaRhoModel: models.pldaRhoModel, psi: models.pldaPsi)

        guard !timedEmbeddings.isEmpty else {
            throw OfflineDiarizationError.noSpeechDetected
        }

        let embeddingFeatures = timedEmbeddings.map { $0.embedding256.map { Double($0) } }
        let rhoFeatures = timedEmbeddings.map { $0.rho128 }

        let clusteringStart = Date()
        let trainingIndices = selectTrainingEmbeddings(
            timedEmbeddings: timedEmbeddings
        )

        let trainingEmbeddings = trainingIndices.map { embeddingFeatures[$0] }
        let trainingRho = trainingIndices.map { rhoFeatures[$0] }

        logger.debug(
            "Clustering will use \(trainingEmbeddings.count)/\(timedEmbeddings.count) embeddings (NaN filtered)"
        )

        let initialClusters: [Int]
        if trainingEmbeddings.count >= 2 {
            initialClusters = AHCClustering().cluster(
                embeddingFeatures: trainingEmbeddings,
                threshold: config.clusteringThreshold
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
            modelCompilationSeconds: models.compilationDuration,
            audioLoadingSeconds: audioLoadingSeconds,
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

    private func purgeDiarizerRepo(at baseDirectory: URL) throws {
        let repoDirectory = baseDirectory.appendingPathComponent(
            Repo.diarizer.folderName,
            isDirectory: true
        )
        if FileManager.default.fileExists(atPath: repoDirectory.path) {
            try FileManager.default.removeItem(at: repoDirectory)
        }
    }

    private func prewarmModelsIfNeeded(_ models: OfflineDiarizerModels) async {
        do {
            let start = Date()
            try prewarmSegmentationModel(models.segmentationModel)
            let elapsed = Date().timeIntervalSince(start)
            let elapsedString = String(format: "%.3f", elapsed)
            logger.debug("Segmentation model prewarmed in \(elapsedString)s")
        } catch {
            logger.debug("Segmentation prewarm skipped: \(error.localizedDescription)")
        }

        do {
            let start = Date()
            try await prewarmEmbeddingStack(models: models)
            let elapsed = Date().timeIntervalSince(start)
            let elapsedString = String(format: "%.3f", elapsed)
            logger.debug("Embedding stack prewarmed in \(elapsedString)s")
        } catch {
            logger.debug("Embedding prewarm skipped: \(error.localizedDescription)")
        }
    }

    private func prewarmSegmentationModel(_ model: MLModel) throws {
        let shape: [NSNumber] = [
            1,
            1,
            NSNumber(value: config.samplesPerWindow),
        ]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        vDSP_vclr(pointer, 1, vDSP_Length(array.count))

        let provider = ZeroCopyDiarizerFeatureProvider(
            features: ["audio": MLFeatureValue(multiArray: array)]
        )
        let options = MLPredictionOptions()
        array.prefetchToNeuralEngine()
        _ = try model.prediction(from: provider, options: options)
    }

    private func prewarmEmbeddingStack(models: OfflineDiarizerModels) async throws {
        let extractor = OfflineEmbeddingExtractor(
            fbankModel: models.fbankModel,
            embeddingModel: models.embeddingModel,
            pldaTransform: PLDATransform(pldaRhoModel: models.pldaRhoModel, psi: models.pldaPsi),
            config: config
        )

        let dummyAudio = [Float](repeating: 0, count: config.samplesPerWindow)
        let dummySegmentation = SegmentationOutput(
            logProbs: [[[0]]],
            speakerWeights: [[[1.0]]],
            numChunks: 1,
            numFrames: 1,
            numSpeakers: 1,
            chunkOffsets: [0],
            frameDuration: max(1e-3, config.windowDuration)
        )

        _ = try await extractor.extractEmbeddings(
            audio: dummyAudio,
            segmentation: dummySegmentation
        )
    }

    private func selectTrainingEmbeddings(
        timedEmbeddings: [TimedEmbedding]
    ) -> [Int] {
        var selected: [Int] = []
        selected.reserveCapacity(timedEmbeddings.count)

        for (index, embedding) in timedEmbeddings.enumerated() {
            let hasNaN = embedding.embedding256.contains { $0.isNaN || $0.isInfinite }
            if hasNaN {
                continue
            }

            selected.append(index)
        }

        if selected.isEmpty {
            return Array(timedEmbeddings.indices)
        }

        return selected
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
                    let dimensionIndex = makeBlasIndexOrFatal(dimension, label: "centroid dimension")
                    let unitStride = BlasIndex(1)
                    let frameLimit = min(gamma.count, trainingEmbeddings.count)

                    for frameIdx in 0..<frameLimit {
                        let weight = gamma[frameIdx][speakerIdx]
                        guard weight > 0 else { continue }
                        denominator += weight
                        let embedding = trainingEmbeddings[frameIdx]
                        precondition(
                            embedding.count == dimension,
                            "Jagged training embeddings are not supported"
                        )
                        embedding.withUnsafeBufferPointer { sourcePointer in
                            numerator.withUnsafeMutableBufferPointer { destinationPointer in
                                guard
                                    let sourceBase = sourcePointer.baseAddress,
                                    let destinationBase = destinationPointer.baseAddress
                                else { return }
                                cblas_daxpy(
                                    dimensionIndex,
                                    weight,
                                    sourceBase,
                                    unitStride,
                                    destinationBase,
                                    unitStride
                                )
                            }
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
            precondition(
                embedding.count == grouped[cluster]!.sum.count,
                "Jagged training embeddings are not supported"
            )
            var entry = grouped[cluster]!
            let countIndex = makeBlasIndexOrFatal(embedding.count, label: "centroid accumulation length")
            let unitStride = BlasIndex(1)
            embedding.withUnsafeBufferPointer { sourcePointer in
                entry.sum.withUnsafeMutableBufferPointer { destinationPointer in
                    guard
                        let sourceBase = sourcePointer.baseAddress,
                        let destinationBase = destinationPointer.baseAddress
                    else { return }
                    cblas_daxpy(
                        countIndex,
                        1.0,
                        sourceBase,
                        unitStride,
                        destinationBase,
                        unitStride
                    )
                }
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
        var accumulator = [Double](repeating: 0, count: first.count)
        let countIndex = makeBlasIndexOrFatal(first.count, label: "fallback centroid length")
        let unitStride = BlasIndex(1)
        for vector in embeddings {
            precondition(
                vector.count == first.count,
                "Jagged training embeddings are not supported"
            )
            vector.withUnsafeBufferPointer { sourcePointer in
                accumulator.withUnsafeMutableBufferPointer { destinationPointer in
                    guard
                        let sourceBase = sourcePointer.baseAddress,
                        let destinationBase = destinationPointer.baseAddress
                    else { return }
                    cblas_daxpy(
                        countIndex,
                        1.0,
                        sourceBase,
                        unitStride,
                        destinationBase,
                        unitStride
                    )
                }
            }
        }
        var scale = 1.0 / Double(embeddings.count)
        accumulator.withUnsafeMutableBufferPointer { pointer in
            guard let baseAddress = pointer.baseAddress else { return }
            vDSP_vsmulD(
                baseAddress,
                1,
                &scale,
                baseAddress,
                1,
                vDSP_Length(first.count)
            )
        }
        return [accumulator]
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
        guard !vector.isEmpty else { return vector }
        var sumSquares = 0.0
        vector.withUnsafeBufferPointer { pointer in
            guard let baseAddress = pointer.baseAddress else { return }
            vDSP_dotprD(
                baseAddress,
                1,
                baseAddress,
                1,
                &sumSquares,
                vDSP_Length(vector.count)
            )
        }
        if sumSquares <= 0 {
            return vector
        }
        var scale = 1.0 / sqrt(sumSquares)
        var normalized = [Double](repeating: 0, count: vector.count)
        vector.withUnsafeBufferPointer { sourcePointer in
            normalized.withUnsafeMutableBufferPointer { destinationPointer in
                guard
                    let sourceBase = sourcePointer.baseAddress,
                    let destinationBase = destinationPointer.baseAddress
                else { return }
                vDSP_vsmulD(
                    sourceBase,
                    1,
                    &scale,
                    destinationBase,
                    1,
                    vDSP_Length(vector.count)
                )
            }
        }
        return normalized
    }

    private func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhs.count else { return 0 }
        if lhs.isEmpty { return 0 }
        var result = 0.0
        lhs.withUnsafeBufferPointer { lhsPointer in
            rhs.withUnsafeBufferPointer { rhsPointer in
                guard
                    let lhsBase = lhsPointer.baseAddress,
                    let rhsBase = rhsPointer.baseAddress
                else { return }
                vDSP_dotprD(
                    lhsBase,
                    1,
                    rhsBase,
                    1,
                    &result,
                    vDSP_Length(lhs.count)
                )
            }
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
