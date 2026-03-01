//
//  SpeakerSegment.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/26/26.
//

import Foundation

public protocol SpeakerSegment64:
    SpeakerFrameRange,
    Identifiable,
    ExpressibleByIntegerLiteral,
    Hashable,
    Comparable,
    Equatable
{
    associatedtype IntegerLiteralType = UInt64
    
    var bits: UInt64 { get set }
    var speakerId: Int { get set }
    var startFrame: Int { get set }
    var endFrame: Int { get set }
    var length: Int { get set }
    var isFinalized: Bool { get set }
    
    var finalized: Self { get }
    var unfinalized: Self { get }
    var anonymized: Self { get }
    
    var anonymizedBits: UInt64 { get }
    
    var startTime: Float { get }
    var endTime: Float { get }
    
    func reassigned(toSpeaker speakerId: Int) -> Self
}


// MARK: - SpeakerSegment
public struct SpeakerSegment: SpeakerSegment64 {
    /*
     * MEMORY LAYOUT:
     *     | speaker ID | is finalized | segment length | start frame |
     * LSB |   0 - 11   |      12      |     13 - 33    |   34 - 63   | MSB
     */
    
    // Sizes
    private static let startBits:   Int = 31    // About 5.5 years
    private static let lengthBits:  Int = 20    // About 1 day
    private static let speakerBits: Int = 12    // 4096 speaker slots
    
    // Offsets
    private static let finalizedOffset: Int = speakerBits
    private static let lengthOffset: Int = finalizedOffset + 1
    private static let startOffset: Int = lengthBits + lengthOffset
    
    // Bit masks
    private static let startMaskInv: UInt64 = (1 << (UInt64.bitWidth - startBits)) - 1
    private static let lengthFieldMask: UInt64 = (1 << lengthBits) - 1
    private static let lengthMaskInv: UInt64 = ~(lengthFieldMask << lengthOffset)
    private static let finalizedMask: UInt64 = 1 << finalizedOffset
    private static let finalizedMaskInv: UInt64 = ~finalizedMask
    private static let speakerFieldMask: UInt64 = (1 << speakerBits) - 1
    private static let speakerMaskInv: UInt64 = ~speakerFieldMask
    private static let intervalMask: UInt64 = speakerMaskInv & finalizedMaskInv
    
    /// The segment's actual contents
    public var bits: UInt64
    
    // - MARK: - Attribute retrieval
    
    /// Sortable segment identifier
    @inline(__always)
    public var id: UInt64 {
        bits & Self.finalizedMaskInv
    }
    
    /// Copy of the segment with the speaker identity removed
    @inline(__always)
    public var anonymized: SpeakerSegment { .init(integerLiteral: anonymizedBits) }
    
    @inline(__always)
    public var anonymizedBits: UInt64 { bits | Self.speakerFieldMask }
    
    /// Copy of the segment with the finalized flag set to `true`
    @inline(__always)
    public var finalized: SpeakerSegment { .init(integerLiteral: bits | Self.finalizedMask) }
    
    /// Copy of the segment with the finalized flag set to `false`
    @inline(__always)
    public var unfinalized: SpeakerSegment { .init(integerLiteral: bits & Self.finalizedMaskInv) }
    
    /// Segment start frame
    @inline(__always)
    public var startFrame: Int {
        get {
            Int(truncatingIfNeeded: bits >> Self.startOffset)
        }
        set {
            let v = UInt64(truncatingIfNeeded: newValue)
            bits = (bits & Self.startMaskInv) | (v << Self.startOffset)
        }
    }
    
    /// Segment end frame
    @inline(__always)
    public var endFrame: Int {
        get { startFrame + length }
        set { length = newValue - startFrame }
    }
    
    /// Segment length
    @inline(__always)
    public var length: Int {
        get {
            Int(truncatingIfNeeded: (bits >> Self.lengthOffset) & Self.lengthFieldMask)
        }
        set {
            let v = UInt64(truncatingIfNeeded: newValue) & Self.lengthFieldMask
            bits = (bits & Self.lengthMaskInv) | (v << Self.lengthOffset)
        }
    }
    
    /// Speaker ID
    @inline(__always)
    public var speakerId: Int {
        get {
            Int(truncatingIfNeeded: bits & Self.speakerFieldMask)
        }
        set {
            let v = UInt64(truncatingIfNeeded: newValue) & Self.speakerFieldMask
            bits = (bits & Self.speakerMaskInv) | v
        }
    }
    
    /// Whether the segment is finalized
    @inline(__always)
    public var isFinalized: Bool {
        get {
            (bits & Self.finalizedMask) != 0
        }
        set {
            let v = newValue ? 1 : 0 as UInt64
            bits = (bits & Self.finalizedMaskInv) | (v << Self.finalizedOffset)
        }
    }
    
    /// Whether the segment is tentative
    @inline(__always)
    public var isTentative: Bool {
        get { !isFinalized }
        set { isFinalized = !newValue }
    }
    
    /// Frame range
    @inline(__always)
    public var frames: Range<Int> { startFrame..<endFrame }
    
    /// Start time in seconds
    public var startTime: Float { Float(startFrame) * SortformerTimelineConfig.frameDurationSeconds }

    /// End time in seconds
    public var endTime: Float { Float(endFrame) * SortformerTimelineConfig.frameDurationSeconds }
    
    // MARK: - Init
    @inlinable
    public init<S>(from segment: S) where S: SpeakerSegment64 {
        self.bits = segment.bits
    }
    
    @inline(__always)
    public init(speakerId: Int, startFrame: Int, length: Int, isFinalized: Bool = true) {
        let startU64 = UInt64(truncatingIfNeeded: startFrame)
        let lengthU64 = UInt64(truncatingIfNeeded: length) & Self.lengthFieldMask
        let slotU64 = UInt64(truncatingIfNeeded: speakerId) & Self.speakerFieldMask
        let isFinalU64 = isFinalized ? 1 : 0 as UInt64
        
        self.bits = (  startU64 << Self.startOffset     | lengthU64 << Self.lengthOffset |
                     isFinalU64 << Self.finalizedOffset | slotU64)
    }
    
    @inline(__always)
    public init(speakerId: Int, startFrame: Int, endFrame: Int, isFinalized: Bool = true) {
        self.init(speakerId: speakerId, startFrame: startFrame, length: endFrame - startFrame, isFinalized: isFinalized)
    }
    
    @inline(__always)
    public init<T>(from segment: T, isFinalized: Bool = true)
    where T: SpeakerFrameRange {
        self.init(speakerId: segment.speakerId, startFrame: segment.startFrame, length: segment.length, isFinalized: isFinalized)
    }
    
    @inline(__always)
    public init(integerLiteral value: UInt64) {
        self.bits = value
    }
    
    // MARK: - Modified Copy
    
    public func reassigned(toSpeaker speakerId: Int) -> Self {
        var result = self
        result.speakerId = speakerId
        return result
    }
    
    // MARK: - Operators
    
    @inline(__always)
    public func hash(into hasher: inout Hasher) {
        hasher.combine(bits)
    }
    
    @inline(__always)
    public static func < <S>(lhs: Self, rhs: S) -> Bool
    where S: SpeakerSegment64 {
        return lhs.bits < rhs.bits
    }
    
    @inline(__always)
    public static func == <S>(lhs: Self, rhs: S) -> Bool
    where S: SpeakerSegment64 {
        return lhs.bits == rhs.bits
    }
    
    // MARK: - Speaker Frame Range
    
    /// Check if the segment contains a frame
    @inline(__always)
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }

    /// Check if this segment overlaps or touches another segment
    @inline(__always)
    public func isContiguous<T>(with other: T) -> Bool
    where T: SortformerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    /// Check if this segment part of the same segment as another one
    @inline(__always)
    public func isContiguous<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool
    where T: SpeakerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlaps<T>(with other: T) -> Bool
    where T: SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlaps<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool
    where T: SpeakerFrameRange {
        SortformerFrameRangeHelpers.overlaps(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlapLength<T>(with other: T) -> Int
    where T: SortformerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlapLength<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Int
    where T: SpeakerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
}


public struct AnonymousSpeakerSegment: SpeakerSegment64 {    
    public var speakerSegment: SpeakerSegment
    
    @inlinable
    public init(from segment: SpeakerSegment) {
        self.speakerSegment = segment
    }

    @inlinable
    public init(from segment: AnonymousSpeakerSegment) {
        self.speakerSegment = segment.speakerSegment
    }
    
    @inlinable
    public init<S>(from segment: S) where S: SpeakerSegment64 {
        self.speakerSegment = .init(integerLiteral: segment.bits)
    }
    
    @inlinable
    public init(integerLiteral: UInt64) {
        self.speakerSegment = .init(integerLiteral: integerLiteral)
    }
    
    @inline(__always)
    public init(speakerId: Int, startFrame: Int, length: Int, isFinalized: Bool = true) {
        speakerSegment = .init(speakerId: speakerId, startFrame: startFrame, length: length, isFinalized: isFinalized)
    }
    
    @inline(__always)
    public init(speakerId: Int, startFrame: Int, endFrame: Int, isFinalized: Bool = true) {
        speakerSegment = .init(speakerId: speakerId, startFrame: startFrame, endFrame: endFrame, isFinalized: isFinalized)
    }
    
    /// The segment's actual contents
    @inline(__always)
    public var bits: UInt64 {
        get { speakerSegment.bits }
        set { speakerSegment.bits = newValue }
    }
    
    // - MARK: - Attribute retrieval
    
    /// Sortable segment identifier
    @inline(__always)
    public var id: UInt64 { speakerSegment.id }
    
    /// Copy of the segment with the speaker identity removed
    @inline(__always)
    public var anonymized: Self {
        .init(from: speakerSegment.anonymized)
    }
    
    @inline(__always)
    public var anonymizedBits: UInt64 {
        speakerSegment.anonymizedBits
    }
    
    /// Copy of the segment with the finalized flag set to `true`
    @inline(__always)
    public var finalized: Self {
        var result = self
        result.isFinalized = true
        return result
    }
    
    /// Copy of the segment with the finalized flag set to `false`
    @inline(__always)
    public var unfinalized: Self {
        var result = self
        result.isFinalized = false
        return result
    }
    
    /// Segment start frame
    @inline(__always)
    public var startFrame: Int {
        get { speakerSegment.startFrame }
        set { speakerSegment.startFrame = newValue }
    }
    
    /// Segment end frame
    @inline(__always)
    public var endFrame: Int {
        get { speakerSegment.endFrame }
        set { speakerSegment.endFrame = newValue }
    }
    
    /// Segment length
    @inline(__always)
    public var length: Int {
        get { speakerSegment.length }
        set { speakerSegment.length = newValue }
    }
    
    /// Speaker ID
    @inline(__always)
    public var speakerId: Int {
        get { speakerSegment.speakerId }
        set { speakerSegment.speakerId = newValue }
    }
    
    /// Whether the segment is finalized
    @inline(__always)
    public var isFinalized: Bool {
        get { speakerSegment.isFinalized }
        set { speakerSegment.isFinalized = newValue }
    }
    
    /// Whether the segment is tentative
    @inline(__always)
    public var isTentative: Bool {
        get { speakerSegment.isTentative }
        set { speakerSegment.isTentative = newValue }
    }
    
    /// Frame range
    @inline(__always)
    public var frames: Range<Int> { speakerSegment.frames }
    
    /// Start time in seconds
    public var startTime: Float { speakerSegment.startTime }

    /// End time in seconds
    public var endTime: Float { speakerSegment.endTime }
    
    // MARK: - Modified Copy
    
    @inline(__always)
    public func reassigned(toSpeaker speakerId: Int) -> Self {
        var result = self
        result.speakerId = speakerId
        return result
    }
    
    // MARK: - Operators
    
    @inline(__always)
    public func hash(into hasher: inout Hasher) {
        hasher.combine( speakerSegment.anonymizedBits )
    }
    
    @inline(__always)
    public static func < <S>(lhs: Self, rhs: S) -> Bool
    where S: SpeakerSegment64 {
        return lhs.bits < rhs.bits
    }
    
    @inline(__always)
    public static func == <S>(lhs: Self, rhs: S) -> Bool
    where S: SpeakerSegment64 {
        return lhs.anonymizedBits == rhs.anonymizedBits
    }
    
    // MARK: - Speaker Frame Range
    
    /// Check if the segment contains a frame
    @inline(__always)
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }

    /// Check if this segment overlaps or touches another segment
    @inline(__always)
    public func isContiguous<T>(with other: T) -> Bool
    where T: SortformerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    /// Check if this segment part of the same segment as another one
    @inline(__always)
    public func isContiguous<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool
    where T: SpeakerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlaps<T>(with other: T) -> Bool
    where T: SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlaps<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool
    where T: SpeakerFrameRange {
        SortformerFrameRangeHelpers.overlaps(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlapLength<T>(with other: T) -> Int
    where T: SortformerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlapLength<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Int
    where T: SpeakerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
}
