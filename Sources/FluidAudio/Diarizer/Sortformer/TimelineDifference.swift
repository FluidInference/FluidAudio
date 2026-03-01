//
//  SortformerTimelineDifference.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/15/26.
//

import Foundation
import Accelerate
import OrderedCollections


public struct SortformerTimelineDifference {
    public private(set) var insertions: Set<SpeakerSegment> = []
    public private(set) var deletions: Set<SpeakerSegment> = []
    
    public var isEmpty: Bool {
        return insertions.isEmpty && deletions.isEmpty
    }
    
    public var inverse: SortformerTimelineDifference {
        return SortformerTimelineDifference(
            insertions: deletions,
            deletions: insertions
        )
    }
    
    public init(
        insertions: Set<SpeakerSegment>,
        deletions: Set<SpeakerSegment>
    ) {
        self.insertions = insertions
        self.deletions  = deletions
    }
    
    /// Compute the difference between old and new segment arrays
    /// Assumes the segments for each speaker are disjoint and sorted from oldest to newest
    /// Two segments are considered "matched" if they are identical
    public init(
        old: [SpeakerSegment],
        new: [SpeakerSegment]
    ) {
        let oldSet = Set(old)
        let newSet = Set(new)
        
        insertions = newSet.subtracting(oldSet)
        deletions  = oldSet.subtracting(newSet)
    }
    
    /// Empty difference
    public init() {
        self.insertions = []
        self.deletions = []
    }
    
    public mutating func insert(_ segment: SpeakerSegment) {
        // Try cancelling with a deletion
        if deletions.contains(segment) {
            deletions.remove(segment)
        } else {
            insertions.insert(segment)
        }
    }
    
    public mutating func insert(_ segments: [SpeakerSegment]) {
        insertions.formUnion(Set(segments).subtracting(deletions))
    }
    
    public mutating func delete(_ segment: SpeakerSegment) {
        // Try cancelling with an insertion
        if insertions.contains(segment) {
            insertions.remove(segment)
        } else {
            deletions.insert(segment)
        }
    }
    
    public mutating func delete(_ segments: [SpeakerSegment]) {
        deletions.formUnion(Set(segments).subtracting(insertions))
    }
    
    public mutating func replace(_ segment: SpeakerSegment, with newSegment: SpeakerSegment) {
        delete(segment)
        insert(newSegment)
    }
    
    public mutating func merge(_ segments: [SpeakerSegment], into newSegment: SpeakerSegment) {
        delete(segments)
        insert(newSegment)
    }
    
    public mutating func split(_ segment: SpeakerSegment, into newSegments: [SpeakerSegment]) {
        delete(segment)
        insert(newSegments)
    }
    
    /// Merge another difference into this one
    /// Insertions and deletions from the other difference are added to this one
    /// with proper cancellation (insertion + deletion of same segment = no-op)
    public mutating func apply(_ other: SortformerTimelineDifference) {
        let ins = other.insertions.subtracting(deletions)
        let del = other.deletions.subtracting(insertions)
        insertions.subtract(other.deletions)
        deletions.subtract(other.insertions)
        insertions.formUnion(ins)
        deletions.formUnion(del)
    }
    
    public func apply(to timeline: inout OrderedSet<SpeakerSegment>) -> Bool {
        guard deletions.isSubset(of: timeline) else {
            return false
        }
        
        timeline.subtract(deletions)
        for insertion in insertions {
            let index = timeline.lastIndex { $0 < insertion }.map { $0 + 1 } ?? 0
            timeline.updateOrInsert(insertion, at: index)
        }
        
        return true
    }
    
    // TODO: Optimize this
    /// - Note: Timeline *must* be sorted chronologically from oldest to newest
    public func compile<C>(for timeline: C) -> (deletions: [Int], insertions: [(index: Int, segment: SpeakerSegment)])?
    where C: RandomAccessCollection, C.Element == SpeakerSegment, C.Index == Int {
        var deletionIndices: [Int] = []
        deletionIndices.reserveCapacity(deletions.count)
        
        for deletion in deletions {
            guard let index = timeline.lastIndex(of: deletion) else {
                return nil
            }
            deletionIndices.append(index)
        }
        
        deletionIndices.sort(by: >)
        
        var compiledInsertions: [(index: Int, segment: SpeakerSegment)] = []
        for insertion in insertions {
            var index = timeline.lastIndex { $0 < insertion }.map { $0 + 1 } ?? 0
            index -= deletions.count { $0 < insertion }
            compiledInsertions.append((index, insertion))
        }
        
        compiledInsertions.sort {
            $0.index > $1.index
        }
        
        return (deletions: deletionIndices, insertions: compiledInsertions)
    }
}
