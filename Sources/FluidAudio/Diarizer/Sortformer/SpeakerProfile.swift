//
//  SpeakerProfile.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/18/26.
//

import Foundation
import Accelerate
import OrderedCollections
import AHCClustering

public class SpeakerProfile: Hashable {
    public let id = UUID()
    public let config: ClusteringConfig
    public private(set) var finalizedClusters: [SpeakerClusterCentroid] = []
    public private(set) var tentativeClusters: [SpeakerClusterCentroid] = []
    public private(set) var outliers: [SpeakerClusterCentroid] = []
    
    public private(set) var finalizedSegments: OrderedSet<AnonymousSpeakerSegment> = []
    public private(set) var tentativeSegments: Set<AnonymousSpeakerSegment> = []
    
    private var lastTentativeSegment: AnonymousSpeakerSegment? = nil
    
    public var segments: [SpeakerSegment] {
        (finalizedSegments.sorted() + tentativeSegments.sorted()).map(\.speakerSegment)
    }
    
    /// The ID of the speaker whose outliers formed this speaker profile
    public private(set) var parent: SpeakerProfile? = nil {
        didSet { oldValue?.children.remove(self) }
    }
    
    public private(set) var children: Set<SpeakerProfile> = []
    
    /// Set of speaker IDs with which this profile cannot link
    public var cannotLink: Set<Int> = []
    
    /// Speaker ID
    public var speakerId: Int
    
    public var isFinalized: Bool = false
    
    /// Whether this speaker has any clusters
    public var hasClusters: Bool {
        !(finalizedClusters.isEmpty && tentativeClusters.isEmpty)
    }
    
    /// Whether this speaker has any segments
    public var hasSegments: Bool {
        !(finalizedSegments.isEmpty && tentativeSegments.isEmpty)
    }
    
    /// Whether this speaker has any outliers
    public var hasOutliers: Bool {
        !outliers.isEmpty
    }
    
    /// The combined weight of all confirmed clusters
    public var finalizedWeight: Float {
        finalizedClusters.reduce(0) { $0 + $1.weight }
    }
    
    /// The combined weight of all tentative clusters
    public var tentativeWeight: Float {
        tentativeClusters.reduce(0) { $0 + $1.weight }
    }
    
    /// The combined weight of all tentative clusters
    public var outlierWeight: Float {
        outliers.reduce(0) { $0 + $1.weight }
    }
    
    public var finalizedSegmentCount: Int { finalizedSegments.count }
    public var tentativeSegmentCount: Int { tentativeSegments.count }
    public var finalizedClusterCount: Int { finalizedClusters.count }
    public var tentativeClusterCount: Int { tentativeClusters.count }
    public var outlierCount: Int { outliers.count }
    public var childCount: Int { children.count }
    
    public var isDroppable: Bool {
        outliers.isEmpty && parent == nil && children.isEmpty
    }
    
    public var droppabilityScore: Float {
        (parent == nil ? 1 : 0) - outlierWeight - Float(children.count)
    }
    
    // MARK: - Init
    
    init(config: ClusteringConfig, speakerId: Int) {
        self.config = config
        self.speakerId = speakerId
    }
    
    private init(
        config: ClusteringConfig,
        speakerId: Int,
        parent: SpeakerProfile? = nil,
        finalizedSegments: OrderedSet<AnonymousSpeakerSegment>,
        tentativeSegments: Set<AnonymousSpeakerSegment>,
        finalizedClusters: [SpeakerClusterCentroid],
        tentativeClusters: [SpeakerClusterCentroid],
        cannotLink: Set<Int>
    ) {
        self.config = config
        self.finalizedClusters = finalizedClusters
        self.tentativeClusters = tentativeClusters
        self.finalizedSegments = finalizedSegments
        self.tentativeSegments = tentativeSegments
        self.lastTentativeSegment = nil
        self.parent = parent
        self.speakerId = speakerId
        self.cannotLink = cannotLink
    }
    
    deinit {
        detachFromParent()
    }
    
    // MARK: - Segment Updates
    
    @inline(__always)
    public func appendFinalizedSegment<S>(_ segment: S) where S: SpeakerSegment64 {
        detachFromParent() // This confirms that this speaker is new
        let newSegment = AnonymousSpeakerSegment(from: segment.reassigned(toSpeaker: speakerId))
        finalizedSegments.updateOrAppend(newSegment)
    }
    
    @inline(__always)
    public func appendTentativeSegment<S>(_ segment: S)
    where S: SpeakerSegment64 {
        let newSegment = AnonymousSpeakerSegment(from: segment.reassigned(toSpeaker: speakerId))
        tentativeSegments.insert(newSegment)
        lastTentativeSegment = newSegment
    }
    
    @inline(__always)
    public func appendSegment<S>(_ segment: S)
    where S: SpeakerSegment64 {
        if segment.isFinalized {
            appendFinalizedSegment(segment)
        } else {
            appendTentativeSegment(segment)
        }
    }
    
    @inline(__always) @discardableResult
    public func popFinalizedSegment() -> AnonymousSpeakerSegment? {
        guard !finalizedSegments.isEmpty else { return nil }
        return finalizedSegments.removeLast()
    }
    
    @inline(__always) @discardableResult
    public func popTentativeSegment() -> AnonymousSpeakerSegment? {
        guard let lastTentativeSegment else { return nil }
        self.lastTentativeSegment = nil
        return finalizedSegments.remove(lastTentativeSegment)
    }
    
    @inline(__always) @discardableResult
    public func popSegment(finalized: Bool) -> AnonymousSpeakerSegment? {
        return finalized ? popFinalizedSegment() : popTentativeSegment()
    }
    
    @inline(__always)
    public func clearTentativeSegments() {
        lastTentativeSegment = nil
        return tentativeSegments.removeAll()
    }
    
    // MARK: - Embedding Updates
    
    public func stream(
        newFinalized: [EmbeddingSegment],
        newTentative: [EmbeddingSegment],
        updateOutliers: Bool = false
    ) {
        outliers.removeAll(keepingCapacity: true)
        
        // Update finalized clusters
        for segment in newFinalized {
            guard let centroid = segment.centroid else {
                debugPrint("Some idiot forgot to initialize the finalized centroids " +
                           "before streaming the segments to the speaker profile")
                continue
            }
            
            // Find the best match
            if let (cluster, _) = findCluster(for: centroid, in: finalizedClusters) {
                cluster.update(with: segment)
            } else if !updateOutliers {
                finalizedClusters.append(centroid.deepCopy())
            } else {
                outliers.append(centroid.deepCopy())
                debugPrint("WARNING: A finalized segment was an outlier in speaker \(speakerId).")
            }
        }
        
        // Initialize tentative clusters
        let oldClusters: [SpeakerClusterCentroid]
        
        if finalizedClusters.isEmpty {
            oldClusters = tentativeClusters
            tentativeClusters.removeAll(keepingCapacity: true)
        } else {
            oldClusters = finalizedClusters
            tentativeClusters = finalizedClusters.map {
                $0.deepCopy(keepingId: true)
            }
        }
            
        // Update tentative clusters
        for segment in newTentative {
            guard let centroid = segment.centroid else {
                debugPrint("Some idiot forgot to initialize the tentative centroids " +
                           "before streaming the segments to the speaker profile")
                continue
            }
            
            // Find the best match
            if let (cluster, _) = findCluster(for: centroid, in: tentativeClusters) {
                cluster.update(with: segment)
                continue
            }
            
            // Create a new cluster if we aren't checking for outliers
            if !updateOutliers || hasMatchingCluster(for: centroid, in: oldClusters) {
                tentativeClusters.append(centroid.deepCopy())
                continue
            }
            
            // Create an outlier cluster
            if let (cluster, _) = findCluster(for: centroid, in: outliers) {
                cluster.update(with: segment)
            } else {
                outliers.append(centroid.deepCopy())
            }
        }
        
        self.isFinalized = false
    }
    
    public func findCluster<E>(
        for embedding: E,
        in clusters: [SpeakerClusterCentroid],
        maxDistance: Float? = nil
    ) -> (cluster: SpeakerClusterCentroid, distance: Float)?
    where E: EmbeddingVector {
        if clusters.isEmpty { return nil }
        
        var bestDistance = (maxDistance ?? config.clusteringThreshold).nextUp
        var bestCluster: SpeakerClusterCentroid? = nil
        
        for cluster in clusters {
            let distance = cluster.cosineDistance(to: embedding)
            if distance < bestDistance {
                bestDistance = distance
                bestCluster = cluster
            }
        }
        
        guard let bestCluster else { return nil }
        return (bestCluster, bestDistance)
    }
    
    @inline(__always)
    public func hasMatchingCluster<E>(
        for embedding: E,
        in clusters: [SpeakerClusterCentroid],
        maxDistance: Float? = nil
    ) -> Bool where E: EmbeddingVector{
        let threshold = maxDistance ?? config.clusteringThreshold
        return clusters.contains {
            $0.cosineDistance(to: embedding) <= threshold
        }
    }
    
    /// Get the chamfer distance between two SpeakerProfiles
    /// - Parameters:
    ///   - other: Another speaker profile
    ///   - useTentative: Whether to use tentative clusters in the distance calculation (defaults to `true`)
    /// - Returns: The chamfer distance to the other speaker profile
    public func distance(to other: SpeakerProfile, useTentative: Bool = true) -> Float {
        guard !other.cannotLink.contains(self.speakerId),
              !self.cannotLink.contains(other.speakerId) else {
            return .infinity
        }
        
        let clustersA = (useTentative && !self.isFinalized)
            ? self.tentativeClusters : self.finalizedClusters
        let clustersB = (useTentative && !other.isFinalized)
            ? other.tentativeClusters : other.finalizedClusters
        
        var bestB: [(dist: Float, weight: Float)] = Array(repeating: (.infinity, 0), count: clustersB.count)
        var sumA: Float = 0
        var sumWeightsA: Float = 0
        
        for embA in clustersA {
            var minDistA = Float.infinity
            var bestWeightA: Float = 0
            for (i, embB) in clustersB.enumerated() {
                let dist: Float = embA.cosineDistance(to: embB)
                if dist < minDistA {
                    minDistA = dist
                    bestWeightA = embB.weight
                }
                if dist < bestB[i].dist {
                    bestB[i] = (dist, embB.weight * embA.weight)
                }
            }
            
            let weight = embA.weight * bestWeightA
            
            sumWeightsA += weight
            sumA += weight * minDistA
        }
        
        let (sumB, sumWeightsB) = bestB.reduce((dist: 0 as Float, weight: 0 as Float)) {
            ($0.dist + $1.dist * $1.weight, $0.weight + $1.weight)
        }
        
        return (sumA / sumWeightsA + sumB / sumWeightsB) / 2
    }
    
    
    /// Separate the outlier clusters speaker profile from the outlier clusters.
    public func takeOutliers(speakerId: Int, cannotLink: Set<Int> = []) -> SpeakerProfile {
        // Collect segments
        var outlierSegments = outliers
            .flatMap(\.segments)
            .map { AnonymousSpeakerSegment(from: $0.reassigned(toSpeaker: speakerId)) }
        let numTentativeOutlierSegments = outlierSegments.partition(by: \.isFinalized)
        
        outlierSegments[numTentativeOutlierSegments...].sort()
        
        var outlierTentativeSegments = Set(outlierSegments
            .prefix(numTentativeOutlierSegments))
        var outlierFinalizedSegments = OrderedSet(outlierSegments
            .suffix(from: numTentativeOutlierSegments))
        
        self.tentativeSegments.subtract(outlierTentativeSegments)
        self.finalizedSegments.subtract(outlierFinalizedSegments)
        lastTentativeSegment = nil
        
        let result = SpeakerProfile(
            config: self.config,
            speakerId: speakerId,
            finalizedSegments: outlierFinalizedSegments,
            tentativeSegments: outlierTentativeSegments,
            finalizedClusters: [],
            tentativeClusters: outliers,
            cannotLink: cannotLink
        )
        
        self.outliers.removeAll()
        self.children.insert(result)
        
        return result
    }
    
    @inline(__always)
    public func detachFromParent() {
        parent = nil
    }
    
    public func returnToParent() {
        guard let parent else { return }
        for outlier in outliers {
            if let (cluster, _) = self.findCluster(for: outlier, in: parent.tentativeClusters) {
                cluster.update(with: outlier)
            } else {
                parent.outliers.append(outlier)
            }
        }
        
        parent.mergeFinalizedSegments(of: self)
        
        self.parent = nil
    }
    
    /// Finalize the speaker
    public func finalize(keepOutliers: Bool = false) {
        guard !isFinalized else { return }
        
        if !tentativeClusters.isEmpty {
            finalizedClusters = tentativeClusters
            tentativeClusters.removeAll()
        }
        
        if !tentativeSegments.isEmpty {
            finalizedSegments.formUnion(tentativeSegments)
            tentativeSegments.removeAll()
        }
        
        if !keepOutliers {
            outliers.removeAll()
        }
        
        isFinalized = true
        detachFromParent()
    }
    
    public func updateCannotLink(with speakerIds: Set<Int>) {
        cannotLink.formUnion(speakerIds)
        cannotLink.remove(speakerId)
    }
    
    public func updateCannotLink(with speakerId: Int) {
        guard speakerId != self.speakerId else { return }
        cannotLink.insert(speakerId)
    }
    
    /// Absorb a speaker profile and finalize both this Speaker and the other being absorbed.
    /// - Parameters:
    ///   - other: The speaker to absorb
    public func absorbAndFinalize(_ other: SpeakerProfile) {
        self.finalize(keepOutliers: true)
        other.finalize(keepOutliers: true)
        
        // 1. Compress clusters and outliers
        
        // Initialize and fill pair-wise distance matrix
        var matrix = EmbeddingDistanceMatrix(.upgma)
        matrix.reserve(self.finalizedClusterCount + other.finalizedClusterCount +
                       self.outlierCount + other.outlierCount)
        
        for cluster in self.finalizedClusters {
            matrix.append(cluster.cppView)
        }
        for cluster in other.finalizedClusters {
            matrix.append(cluster.cppView)
        }
        for outlier in self.outliers {
            matrix.append(outlier.cppOutlierView)
        }
        for outlier in other.outliers {
            matrix.append(outlier.cppOutlierView)
        }
        
        // Extract clusters
        let dendrogram = matrix.dendrogram()
        let clusters = dendrogram.extractClusters(config.clusteringThreshold)
        
        var newClusters: [SpeakerClusterCentroid] = []
        var newOutliers: [SpeakerClusterCentroid] = []
        newClusters.reserveCapacity(clusters.count)
        newOutliers.reserveCapacity(clusters.count)
        
        for cluster in clusters {
            var isOutlier: Bool = false
            let centroid = SpeakerClusterCentroid(
                cluster: cluster,
                dendrogram: dendrogram,
                matrix: matrix,
                isFinalized: true,
                isOutlierResult: &isOutlier
            )
            if isOutlier {
                newOutliers.append(centroid)
            } else {
                newClusters.append(centroid)
            }
        }
        
        // Free the matrix so its embeddings don't become dangling pointers
        matrix.free()
        
        self.finalizedClusters = newClusters
        self.outliers = newOutliers
        
        // 2. Update segments
        self.mergeFinalizedSegments(of: other)
        
        // 3. Inherit cannot-link constraints
        self.cannotLink.formUnion(other.cannotLink)
        self.cannotLink.remove(speakerId)
    }
    
    private func mergeFinalizedSegments(of other: SpeakerProfile) {
        guard !other.finalizedSegments.isEmpty else { return }
        let wasEmpty = finalizedSegments.isEmpty
        
        if self.speakerId == other.speakerId {
            finalizedSegments.append(contentsOf: other.finalizedSegments)
        } else {
            finalizedSegments.append(contentsOf:
                other.finalizedSegments.map { $0.reassigned(toSpeaker: self.speakerId) })
        }
        
        if !wasEmpty {
            self.finalizedSegments.sort()
        }
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    public static func == (lhs: SpeakerProfile, rhs: SpeakerProfile) -> Bool {
        return lhs.id == rhs.id
    }
}

public class SpeakerClusterCentroid: EmbeddingVector {
    
    public var id: UUID { embedding.id }
    public let embedding: SpeakerEmbedding
    public var weight: Float
    public var segments: [SpeakerSegment]
    public var isFinalized: Bool
    
    public var bufferView: UnsafeBufferPointer<Float> { embedding.bufferView }
    public var baseAddress: UnsafePointer<Float>? { embedding.baseAddress }
    public var magnitude: Float { embedding.magnitude }
    
    @inline(__always)
    public var cppView: SpeakerEmbeddingWrapper { cppWrapper(isOutlier: false) }
    
    @inline(__always)
    public var cppOutlierView: SpeakerEmbeddingWrapper { cppWrapper(isOutlier: true) }
    
    public init(id: UUID = UUID(), segments: [SpeakerSegment] = [], weight: Float, isFinalized: Bool = true) {
        self.embedding = SpeakerEmbedding(id: id, startFrame: 0, endFrame: 0)
        self.segments = segments
        self.isFinalized = isFinalized
        self.weight = weight
    }
    
    public init(id: UUID = UUID(), embedding: SpeakerEmbedding, segments: [SpeakerSegment] = [], weight: Float, isFinalized: Bool = true) {
        // Create a deep copy of the embedding
        self.embedding = SpeakerEmbedding(id: id, embedding: embedding.bufferView, startFrame: embedding.startFrame, endFrame: embedding.endFrame)
        self.segments = segments
        self.isFinalized = isFinalized
        self.weight = weight
    }
    
    init(
        cluster: Cluster,
        dendrogram: borrowing Dendrogram,
        matrix: borrowing EmbeddingDistanceMatrix,
        isFinalized: Bool = true,
        isOutlierResult: inout Bool
    ) {
        self.weight = cluster.weight()
        self.embedding = SpeakerEmbedding(startFrame: 0, endFrame: 0)
        self.isFinalized = isFinalized
        
        var embeddingView = embedding.withUnsafeMutableBufferPointer { embeddingBuf in
            SpeakerEmbeddingWrapper(embeddingBuf.baseAddress)
        }
        
        matrix.computeClusterCentroid(cluster, &embeddingView)
        
        // Copy segments
        let segmentCount: Int = embeddingView.segmentCount()
        self.segments = Array(unsafeUninitializedCapacity: segmentCount) { buffer, count in
            embeddingView.segments().withContiguousStorageIfAvailable { segmentsBuf in
                segmentsBuf.withMemoryRebound(to: SpeakerSegment.self) { srcBuf in
                    _ = buffer.initialize(fromContentsOf: srcBuf)
                }
            }
            count = segmentCount
        }
        
        isOutlierResult = embeddingView.isOutlier()
    }
    
    public func update(with centroid: SpeakerClusterCentroid) {
        self.weight += centroid.weight
        var alpha = centroid.weight / self.weight
        
        self.withUnsafeMutableBufferPointer { muPtr in
            // µ_n = µ_{n-1} + w/W_n (x - µ_{n-1})
            vDSP_vintb(
                muPtr.baseAddress!, 1,
                centroid.baseAddress!, 1,
                &alpha,
                muPtr.baseAddress!, 1,
                vDSP_Length(muPtr.count)
            )
        }
        
        self.segments.append(contentsOf: centroid.segments)
    }
    
    /// Update the centroid in place
    @inline(__always)
    public func update(with segment: EmbeddingSegment) {
        guard let centroid = segment.centroid else {
            return
        }
        update(with: centroid)
    }
    
    public func setTo(_ other: SpeakerClusterCentroid) {
        self.embedding.withUnsafeMutableBufferPointer { muPtr in
            _ = muPtr.initialize(fromContentsOf: other.embedding.bufferView)
        }
        self.weight = other.weight
        self.segments = other.segments
        self.isFinalized = other.isFinalized
    }
    
    @inline(__always)
    public func deepCopy(keepingId: Bool = false) -> SpeakerClusterCentroid {
        SpeakerClusterCentroid(
            id: keepingId ? self.id : UUID(),
            embedding: self.embedding,
            segments: self.segments,
            weight: self.weight,
            isFinalized: self.isFinalized
        )
    }
    
    /// Get the cosine distance to another embedding vector
    /// - Parameter other: Another embedding vector
    /// - Returns: The cosine distance between this centroid to the speaker embedding
    @inline(__always)
    public func cosineDistance<E>(to other: E) -> Float where E: EmbeddingVector {
        return embedding.cosineDistance(to: other)
    }

    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        return try embedding.withUnsafeBufferPointer(body)
    }
    
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        return try embedding.withUnsafeMutableBufferPointer(body)
    }
    
    public static func ==(lhs: SpeakerClusterCentroid, rhs: SpeakerClusterCentroid) -> Bool {
        return lhs.id == rhs.id
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(self.id)
    }
    
    private func cppWrapper(isOutlier: Bool) -> SpeakerEmbeddingWrapper {
        embedding.withUnsafeMutableBufferPointer { embeddingBuf in
            segments.withUnsafeBufferPointer { segmentsBuf in
                segmentsBuf.withMemoryRebound(to: UInt64.self) { segmentsBuf in
                    SpeakerEmbeddingWrapper.init(
                        embeddingBuf.baseAddress,
                        weight,
                        isOutlier,
                        segmentsBuf.baseAddress,
                        segmentsBuf.count
                    )
                }
            }
        }
    }
}
