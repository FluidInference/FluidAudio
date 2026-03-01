//
//  SortformerSegmentEmbeddings.swift
//  SortformerTest
//

import Foundation
import Accelerate
import CoreML


public protocol EmbeddingVector: Identifiable, Hashable {
    var id: UUID { get }
    var bufferView: UnsafeBufferPointer<Float> { get }
    var baseAddress: UnsafePointer<Float>? { get }
    var magnitude: Float { get }
    
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R
    
    func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R
    
    func cosineDistance<E>(to other: E) -> Float where E: EmbeddingVector
}



/// Represents a region where an embedding was extracted
public class SpeakerEmbedding: EmbeddingVector, SortformerFrameRange, Comparable {
    /// Embedding ID
    public let id: UUID
    
    /// Start frame of the embedding region
    public var startFrame: Int { frames.lowerBound }
    
    /// End frame of the embedding region (non-inclusive)
    public var endFrame: Int { frames.upperBound }
    
    /// Frame range
    public let frames: Range<Int>
    
    /// The actual embedding vector
    public var bufferView: UnsafeBufferPointer<Float> {
        UnsafeBufferPointer(buffer)
    }
    
    public var baseAddress: UnsafePointer<Float>? {
        return bufferView.baseAddress
    }
    
    /// Number of features in the embedding vector
    public var count: Int { bufferView.count }
    
    /// Length in frames
    public var length: Int { frames.count }
    
    /// Vector magnitude
    public var magnitude: Float {
        if let m = cachedMagnitude { return m }
        let m = sqrt(vDSP.sumOfSquares(bufferView))
        cachedMagnitude = m
        return m
    }
    
    /// Buffer holding the embedding vector
    private let buffer: UnsafeMutableBufferPointer<Float>
    
    /// Cached magnitude
    private var cachedMagnitude: Float? = nil
    
    public init(id: UUID = UUID(), startFrame: Int, endFrame: Int) {
        self.id = id
        self.frames = startFrame..<endFrame
        self.buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: EmbeddingConfig.embeddingFeatures)
    }
    
    public convenience init<C>(id: UUID = UUID(), embedding: C, startFrame: Int, endFrame: Int, shouldNormalize: Bool = false)
    where C: Sequence, C.Element == Float {
        self.init(id: id, startFrame: startFrame, endFrame: endFrame)
        _ = self.buffer.initialize(from: embedding)
        
        if shouldNormalize {
            self.normalize()
        }
    }
    
    deinit {
        self.buffer.deallocate()
    }
    
    /// Get the cosine distance to another embedding
    public func cosineDistance<E>(to other: E) -> Float
    where E: EmbeddingVector {
        let length = vDSP_Length(buffer.count)
        var dot: Float = 0
        vDSP_dotpr(
            self.buffer.baseAddress!, 1,
            other.bufferView.baseAddress!, 1,
            &dot, length
        )
        return 1.0 - dot / (self.magnitude * other.magnitude)
    }
    
    /// Normalize this embedding vector to a new magnitude
    public func normalize(toLength newMagnitude: Float = 1) {
        var normalizer = newMagnitude / sqrt(vDSP.sumOfSquares(buffer))
        vDSP_vsmul(buffer.baseAddress!, 1, &normalizer,
                   buffer.baseAddress!, 1,
                   vDSP_Length(buffer.count))
        cachedMagnitude = newMagnitude
    }
    
    /// Set cached magnitude. Only use this if you normalized know the resulting magnitude after
    /// manipulating the mutable buffer pointer.
    /// - Parameter newMagnitude: The new magnitude to cache
    @inline(__always)
    public func setMagnitudeUnsafe(to newMagnitude: Float) {
        self.cachedMagnitude = newMagnitude
    }
    
    /// Get the embedding buffer (non-mutable)
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        return try body(bufferView)
    }
    
    /// Get the embedding buffer (mutable)
    public func withUnsafeMutableBufferPointer<R>(_ body: ( UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        self.cachedMagnitude = nil
        return try body(buffer)
    }
    
    /// Check if this region contains a frame
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }
    
    /// Check if this region is contiguous with another one
    public func isContiguous<T>(with other: T) -> Bool
    where T : SortformerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    /// Check if this region overlaps another one
    public func overlaps<T>(with other: T) -> Bool
    where T : SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
    
    /// Calculate the overlap length with a segment
    public func overlapLength<T>(with segment: T) -> Int
    where T: SortformerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, segment)
    }
    
    /// Calculate the overlap length with a segment
    public func framesOutside<T>(of segment: T) -> Int
    where T: SortformerFrameRange {
        return length - SortformerFrameRangeHelpers.overlapLength(self, segment)
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(startFrame)
        hasher.combine(endFrame)
    }
    
    public static func < (lhs: SpeakerEmbedding, rhs: SpeakerEmbedding) -> Bool {
        (lhs.startFrame, lhs.endFrame) < (rhs.startFrame, rhs.endFrame)
    }
    
    public static func == (lhs: SpeakerEmbedding, rhs: SpeakerEmbedding) -> Bool {
        lhs.id == rhs.id
    }
    
    public static func === (lhs: SpeakerEmbedding, rhs: SpeakerEmbedding) -> Bool {
        lhs.buffer.baseAddress == rhs.buffer.baseAddress
    }
}


// MARK: - Embedding Segment
/// Tracks embeddings for a disjoint segment
public class EmbeddingSegment: SpeakerFrameRange, Identifiable {
    /// Segment ID
    public private(set) var id: UUID = .zero

    /// Speaker index in Sortformer output
    public var speakerId: Int

    /// Index of segment start frame
    public var startFrame: Int

    /// Index of segment end frame
    public var endFrame: Int

    /// Range of frames that this segment covers
    public var frames: Range<Int> { startFrame..<endFrame }

    /// Length of the segment in frames
    public var length: Int { endFrame - startFrame }
    
    /// Whether this segment is subject to updates
    public var isFinalized: Bool
    
    public private(set) var centroid: SpeakerClusterCentroid?
    
    /// Extracted embedding regions for this segment
    public private(set) var embeddings: [SpeakerEmbedding]
    
    /// IDs of the corresponding `SortformerSegments`
    public var segments: [SpeakerSegment]
    
    public var isNone: Bool { speakerId < 0 }
    public var isValid: Bool { speakerId >= 0 }
    
    public static let none: EmbeddingSegment = .init(slot: -1, startFrame: 0, endFrame: 0, segments: [])
    
    // MARK: - Init
    public init(
        slot: Int,
        startFrame: Int,
        endFrame: Int,
        finalized: Bool = true,
        segments: [SpeakerSegment] = []
    ) {
        self.speakerId = slot
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self.embeddings = []
        self.segments = segments
    }
    
    public init(
        slot: Int,
        startFrame: Int,
        endFrame: Int,
        finalized: Bool = true,
        segment: SpeakerSegment
    ) {
        self.speakerId = slot
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self.embeddings = []
        self.segments = [segment]
    }
    
    // MARK: - Speaker Frame Range
    public func contains(_ frame: Int) -> Bool {
        frames.contains(frame)
    }
    
    public func isContiguous<T>(with other: T) -> Bool where T : SortformerFrameRange {
        SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    public func isContiguous<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool where T : SpeakerFrameRange {
        SortformerFrameRangeHelpers.isContiguous(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    public func overlaps<T>(with other: T) -> Bool where T : SortformerFrameRange {
        frames.overlaps(other.frames)
    }
    
    public func overlaps<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool where T : SpeakerFrameRange {
        SortformerFrameRangeHelpers.overlaps(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    public func overlapLength<T>(with other: T) -> Int where T : SortformerFrameRange {
        SortformerFrameRangeHelpers.overlapLength(self, other)
    }
    
    public func overlapLength<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Int where T : SpeakerFrameRange {
        SortformerFrameRangeHelpers.overlapLength(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    // MARK: - Embedding extraction
    
    public func clearEmbeddings() {
        embeddings.removeAll()
    }
    
    /// Determine the required embeddings needed to close all gaps
    /// Uses a greedy interval covering algorithm to minimize the number of requests
    /// - Parameters:
    ///   - extractor: Embedding Manager
    ///   - streamingHorizonFrame: If the segment ends before this frame, then its end frame is probably finalized.
    public func initializeEmbeddings(
        with extractor: EmbeddingManager,
        streamingHorizonFrame: Int
    ) throws {
        if embeddings.isEmpty {
            embeddings = extractor.takeMatches(for: self)
        } else {
            embeddings.append(contentsOf: extractor.takeMatches(for: self))
        }
        
        let isComplete = endFrame < streamingHorizonFrame || isFinalized
        let requests: [EmbeddingRequest]
        let minEmbeddingLength = extractor.config.minEmbeddingFrames
        let maxEmbeddingLength = extractor.config.maxEmbeddingFrames
        let maxGapLength = extractor.config.maxFramesSkipped
        
        if embeddings.isEmpty {
            requests = generateInitialRequests(
                minEmbeddingLength: minEmbeddingLength,
                maxEmbeddingLength: maxEmbeddingLength,
                maxGapLength: maxGapLength,
                isComplete: isComplete
            )
        } else {
            // Sort in decending order by end frame
            let embeddingFrames = embeddings.map(\.frames).sorted {
                ($0.upperBound, $0.lowerBound) > ($1.upperBound, $1.lowerBound)
            }
            var cutoff = endFrame
            if !isComplete {
                cutoff -= maxEmbeddingLength
            }
            
            // Gets gaps in descending order
            let gaps = getGaps(
                in: embeddingFrames,
                before: cutoff,
                ifAnyExceed: extractor.config.maxFramesSkipped
            )
            
            // Since frames are in descending order, the first range is the last one chronologically
            guard let lastEmbeddingEnd = embeddingFrames.first?.upperBound else {
                preconditionFailure("Last embedding has no upper bound")
            }
            
            let requestStarts = generateCovering(
                for: gaps,
                biggerThan: extractor.config.maxFramesSkipped,
                withCoverLength: extractor.config.maxEmbeddingFrames,
                lastEmbeddingEnd: lastEmbeddingEnd,
            )
            
            requests = requestStarts.map { start in
                EmbeddingRequest(
                    startFrame: max(startFrame, start),
                    endFrame: start + extractor.config.maxEmbeddingFrames
                )
            }
        }
        
        let newEmbeddings = try extractor.processRequests(requests)
        embeddings.append(contentsOf: newEmbeddings)
        
        initializeId()
    }
    
    /// - Parameters:
    ///   - gaps: Disjoint gap intervals sorted by start frame in decending order
    ///   - maxGapSize: Maximum gap between embeddings
    ///   - coverLength: Number of frames that one embedding can cover
    ///   - lastEmbeddingEnd: The end frame of the last embedding
    /// - Returns: List of cover start indices
    /// - Precondition: `gaps` is disjoint and sorted in decending order by start/end frame
    /// - Precondition: The union of all intervals in `gaps` is a subset of `self.startFrame ..< self.endFrame`
    /// - Precondition: `lastEmbeddingEnd <= self.endFrame`
    /// - Precondition: If `lastEmbeddingEnd < self.endFrame`, then `gaps[0] = lastEmbeddingEnd ..< self.endFrame`
    /// - Precondition: `coverLength <= self.length`
    private func generateCovering(
        for gaps: [(start: Int, end: Int)],
        biggerThan maxGapSize: Int,
        withCoverLength coverLength: Int,
        lastEmbeddingEnd: Int,
    ) -> [Int] {
        guard gaps.count > 0 else { return [] }
        
        var covers: [(start: Int, gapStart: Int)] = []
        
        // It's covered starting from this frame
        var coveredFrom = gaps[0].end
        
        // Backward pass to generate coverings
        for (start, end) in gaps {
            // Check if the gap overlaps with the last cover
            if let previousCover = covers.last, previousCover.start < end {
                covers[covers.count-1].gapStart = start
            }
            
            // There is an embedding that starts at this gap's end, so extend the range of covered frames
            coveredFrom = min(coveredFrom, end)
            
            // Tile the entire gap
            while coveredFrom - maxGapSize > start {
                coveredFrom = coveredFrom - maxGapSize - coverLength
                covers.append((start: coveredFrom, gapStart: start))
            }
        }
        
        guard covers.count > 0 else {
            return []
        }
        
        // Forward pass to optimize coverage
        
        var previousEnd: Int = .min
        var nextUncovered: Int = 0
        
        for i in (0..<covers.count).reversed() {
            // Shift the segment as far forward as possible so it's not touching something in front
            nextUncovered = max(previousEnd, covers[i].gapStart)
            
            if nextUncovered > covers[i].start {
                covers[i].start = nextUncovered
            }
            previousEnd = covers[i].start + coverLength
        }
        
        // Partial backwards pass to make the last cover align with the end frame if possible.
        // Coverings are in reverse order, so the last cover is at index 0
        var lastCover: (start: Int, gapStart: Int) {
            get { covers[0] }
            set { covers[0] = newValue }
        }
        
        let lastCoverEnd = lastCover.start + coverLength
        if lastCoverEnd > lastEmbeddingEnd {
            var shift = endFrame - lastCoverEnd
            if shift > 0 {
                let currentGap = lastCover.start - nextUncovered
                let maxShift = maxGapSize - currentGap
                shift = min(shift, maxShift)
            }
            lastCover.start += shift
            
            // Ensure that the previous segments don't pass the end frame
            if shift < 0 && covers.count > 1 {
                var nextStart = covers[0].start
                
                for i in 1..<covers.count {
                    let overlap = covers[i].start + coverLength - nextStart
                    guard overlap > 0 else { break }
                    nextStart = covers[i].start - overlap
                    covers[i].start = max(startFrame, nextStart)
                    guard nextStart > startFrame else { break }
                }
            }
        }
        
        return covers.reversed().map(\.start)
    }
    
    /// Get embedding gaps in reversed order
    /// - Parameters:
    ///   - embeddingRanges: Covered embedding ranges sorted in descending order by end frame
    ///   - cutoffFrame: Gaps will be ignored begining at this frame
    ///   - maxGapSize: Maximum allowed gap size. If no gaps exceed this threshold, then nothing is returned.
    /// - Returns: Frame intervals `[start, end)` that not covered by the embeddings
    private func getGaps(
        in embeddingRanges: [Range<Int>],
        before cutoffFrame: Int,
        ifAnyExceed maxGapSize: Int
    ) -> [(start: Int, end: Int)] {
        var gaps: [(start: Int, end: Int)] = []
        var end = cutoffFrame
        var hasLargeGap = false
        
        for range in embeddingRanges {
            if range.upperBound < end {
                // There's a gap after this embedding
                gaps.append((range.upperBound, end))
                
                if end - range.upperBound > maxGapSize {
                    hasLargeGap = true
                }
            }
            // Next gap may end at the start of this range
            end = min(end, range.lowerBound)
        }
        
        // Check for gap at the start
        if (hasLargeGap && end > startFrame) || (end - startFrame > maxGapSize) {
            gaps.append((startFrame, end))
            hasLargeGap = true
        }
        
        guard hasLargeGap else { return [] }
        return gaps
    }
    
    /// Generate initial embedding requests when no embeddings exist
    /// Uses non-overlapping windows to minimize redundancy
    private func generateInitialRequests(
        minEmbeddingLength: Int,
        maxEmbeddingLength: Int,
        maxGapLength: Int,
        isComplete: Bool
    ) -> [EmbeddingRequest] {
        // Don't generate requests for segments that are too short
        guard length >= minEmbeddingLength else { return [] }
        
        // If segment fits in one embedding
        if length <= maxEmbeddingLength {
            return [EmbeddingRequest(for: self)]
        }
        
        var requests: [EmbeddingRequest] = []
        var currentStart = startFrame

        // Since start = endFrame - 1 -> gap = 0, gap = endFrame - start - 1
        // So, endFrame - start - 1 ≤ maxGap, or start < endFrame - maxGap
        let maxEndGap = isComplete ? maxGapLength : maxEmbeddingLength - 1
        let firstOptionalFrame = endFrame - maxEndGap
        
        while currentStart < firstOptionalFrame {
            var currentEnd = currentStart + maxEmbeddingLength
            let overflow = currentEnd - endFrame
            
            if overflow > 0 {
                currentStart = max(startFrame, currentStart - overflow)
                currentEnd = endFrame
            }
            
            let currentLength = currentEnd - currentStart
            
            // Only add if the chunk is at least minEmbeddingLength
            // This guard should always pass
            guard currentLength >= minEmbeddingLength else {
                print("WARNING: Found a segment with an embedding length too short (\(currentLength) < \(minEmbeddingLength)). Terminating.")
                return requests
            }
            
            requests.append(EmbeddingRequest(
                startFrame: currentStart,
                endFrame: currentEnd
            ))
            
            currentStart = currentEnd + maxGapLength
        }
        
        return requests
    }
    
    private func initializeId() {
        guard !embeddings.isEmpty else { return }
        self.id = embeddings[0].id
        guard embeddings.count > 1 else { return }

        // 2. Access the memory as two UInt64 values
        withUnsafeMutableBytes(of: &self.id) { idPtr in
            let s = idPtr.assumingMemoryBound(to: UInt64.self)
            
            for i in 1..<embeddings.count {
                var next = embeddings[i].id.uuid
                withUnsafeBytes(of: &next) { nextPtr in
                    let o = nextPtr.assumingMemoryBound(to: UInt64.self)
                    let (newLow, overflow) = s[0].addingReportingOverflow(o[0])
                    s[0] = newLow
                    s[1] &+= o[1] &+ (overflow ? 1 : 0)
                }
            }
        }
    }
    
    // MARK: - Concatenating two embedding segments
    
    /// Check whether this embedding segment must link with another segment.
    /// This requires that the last segment ID in this segment is the first segment ID in `other`.
    /// - Note: Segments can only absorb segments built after itself.
    internal func mustLink(with other: EmbeddingSegment) -> Bool {
        return (self.speakerId == other.speakerId &&
                self.segments.last == other.segments.first)
    }
    
    /// Check if this segment must link with another segment and absorb it if so.
    /// - Parameter other: Another embedding segment
    /// - Returns: `true` if the segment was absorbed, `false` if not.
    internal func successfullyAbsorbed(_ other: EmbeddingSegment) -> Bool {
        guard self.mustLink(with: other) else {
            return false
        }
        
        self.startFrame = min(self.startFrame, other.startFrame)
        self.endFrame = max(self.endFrame, other.endFrame)
        self.isFinalized = self.isFinalized && other.isFinalized
        
        self.embeddings.append(contentsOf: other.embeddings)
        self.segments.append(contentsOf: other.segments.dropFirst())
        
        // Combine IDs
        withUnsafeMutableBytes(of: &self.id) { idPtr in
            withUnsafeBytes(of: &other.id) { otherIdPtr in
                let s = idPtr.assumingMemoryBound(to: UInt64.self)
                let o = otherIdPtr.assumingMemoryBound(to: UInt64.self)
                let (newLow, overflow) = s[0].addingReportingOverflow(o[0])
                
                s[0] = newLow
                s[1] &+= o[1] &+ (overflow ? 1 : 0)
            }
        }
        
        return true
    }
    
    // MARK: - Centroid computation
    public func initializeCentroid(withCache cache: [UUID : SpeakerClusterCentroid] = [:]) {
        guard !embeddings.isEmpty else { return }

        if let cached = cache[id] {
            self.centroid = cached
            return
        }
        
        guard embeddings.count > 1 else {
            self.centroid = SpeakerClusterCentroid(
                id: id,
                embedding: embeddings[0],
                segments: segments,
                weight: Float(embeddings[0].length),
                isFinalized: isFinalized
            )
            return
        }

        var w0 = Float(embeddings[0].length)
        var w1 = Float(embeddings[1].length)
        
        let centroid = SpeakerClusterCentroid(
            id: id,
            segments: segments,
            weight: w0 + w1,
            isFinalized: isFinalized
        )
        
        centroid.withUnsafeMutableBufferPointer { centroidPtr in
            let dims = vDSP_Length(embeddings[0].count)
            
            // Centroid = (embedding_0 * weight_0) + (embedding_1 * weight_1)
            vDSP_vsmsma(
                embeddings[0].baseAddress!, 1, &w0,
                embeddings[1].baseAddress!, 1, &w1,
                centroidPtr.baseAddress!, 1,
                dims
            )
            
            // Accumulate remaining embeddings
            for embedding in embeddings.dropFirst(2) {
                var weight = Float(embedding.length)
                centroid.weight += weight // This shouldn't violate the exclusive access
                
                // Centroid += embedding * weight
                vDSP_vsma(
                    embedding.baseAddress!, 1, &weight,
                    centroidPtr.baseAddress!, 1,
                    centroidPtr.baseAddress!, 1,
                    dims
                )
            }
        }

        centroid.embedding.normalize()
        self.centroid = centroid
    }
}

public struct EmbeddingRequest: SortformerFrameRange, Hashable {
    public let startFrame: Int
    public let endFrame: Int
    public var frames: Range<Int> { startFrame..<endFrame }
    public var length: Int { endFrame - startFrame }
    
    /// Create a request with explicit frame range
    public init(startFrame: Int, endFrame: Int) {
        self.startFrame = startFrame
        self.endFrame = endFrame
    }
    
    /// Create a request covering an entire segment's range
    public init<T>(for segment: T) where T: SortformerFrameRange {
        self.startFrame = segment.startFrame
        self.endFrame = segment.endFrame
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(startFrame)
        hasher.combine(endFrame)
    }
    
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }
    
    public func isContiguous<T>(with other: T) -> Bool where T : SortformerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    public func overlaps<T>(with other: T) -> Bool where T : SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
    
    public func overlapLength<T>(with other: T) -> Int where T : SortformerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other)
    }
}


extension UUID {
    static var zero: UUID {
        UUID(uuid: (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00))
    }
}

