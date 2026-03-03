//
//  SpeakerDatabase.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/19/26.
//

import Foundation

public class SpeakerDatabase {
    public let config: ClusteringConfig
    public private(set) var inactiveSpeakers: [SpeakerProfile] = []
    public private(set) var activeSpeakers: [Int : SpeakerProfile] = [:]
    
    private var nextSpeakerId: Int = 0
    
    public var droppableSlots: [Int] {
        activeSpeakers
            .filter { $1.isDroppable }
            .map(\.key)
    }
    
    public var numSlots: Int {
        config.numSlots
    }
    
    public var hasVacantSlots: Bool {
        activeSpeakers[numSlots - 1] == nil
    }
    
    // MARK: - Init
    public init(config: SortformerTimelineConfig) {
        self.config = .init(from: config)
        self.activeSpeakers.reserveCapacity(config.numSpeakers)
    }
    
    public func stream<F, T>(
        newFinalized: F,
        newTentative: T,
        onSlotFreed: ((Int) -> Void)?
    ) where F: Sequence, F.Element == EmbeddingSegment,
            T: Sequence, T.Element == EmbeddingSegment
    {
        var binnedSegments: [Int : (finalized: [EmbeddingSegment], tentative: [EmbeddingSegment])] = [:]
        binnedSegments.reserveCapacity(activeSpeakers.count)
        
        for (slot, _) in activeSpeakers {
            binnedSegments[slot] = ([], [])
        }
        
        for segment in newFinalized {
            binnedSegments[segment.speakerId]?.finalized.append(segment)
        }
        
        for segment in newTentative {
            binnedSegments[segment.speakerId]?.tentative.append(segment)
        }
        
        for (slot, (finalized, tentative)) in binnedSegments {
            // Initialize the speaker if needed
            guard let speaker = activeSpeakers[slot] else {
                debugPrint("WARNING: Found a new speaker in the stream, but it was not active.")
                continue
            }

            // Trial speaker received no assignments, so parent speaker reclaimed ownership.
            // Return trial segments/clusters before this update path clears tentative clusters.
            if speaker.parent != nil, finalized.isEmpty, tentative.isEmpty {
                speaker.returnToParent()
                activeSpeakers[slot] = nil
                continue
            }

            let checkOutliers = !self.hasVacantSlots
            
            speaker.stream(
                newFinalized: finalized,
                newTentative: tentative,
                updateOutliers: checkOutliers
            )
            
            // Return speaker to their parent if their clusters were moved to another speaker
            guard speaker.hasClusters || (speaker.parent == nil && speaker.hasSegments) else {
                speaker.returnToParent()
                activeSpeakers[slot] = nil
                continue
            }
        }
        
        updateSpeakerIds()
        
        while let droppedSlot = pickSlotToDrop() {
            freeSlot(droppedSlot)
            onSlotFreed?(droppedSlot)
        }
    }
    
    public func finalizeAll() {
        for (_, speaker) in activeSpeakers {
            speaker.finalize()
        }
        for speaker in inactiveSpeakers {
            speaker.finalize()
        }
    }
    
    public func clear() {
        activeSpeakers.removeAll()
        inactiveSpeakers.removeAll()
        nextSpeakerId = 0
    }
    
    public func getSpeaker(atSlot slot: Int) -> SpeakerProfile {
        if let existing = activeSpeakers[slot] {
            return existing
        }
        
        let newSpeaker = SpeakerProfile(config: config, speakerId: inactiveSpeakers.count + slot)
        
        return newSpeaker
    }
    
    public func freeSlot(_ slot: Int) {
        // Remove the old speaker
        var removedId: Int? = nil
        if let speaker = activeSpeakers.removeValue(forKey: slot) {
            if let match = findMatchingSpeaker(for: speaker) {
                match.absorbAndFinalize(speaker)
            } else {
                inactiveSpeakers.append(speaker)
            }
            
            removedId = speaker.speakerId
            
            // Only update cannot-link constraints from speakers as they are deactivated to avoid stale IDs
            for other in activeSpeakers.values {
                other.updateCannotLink(with: speaker.speakerId)
            }
            
            // I don't think this is necessary since it only needs to be one sided anyway
//            let cannotLink = speaker.cannotLink
//            for other in inactiveSpeakers where cannotLink.contains(other.speakerId) {
//                other.updateCannotLink(with: speaker.speakerId)
//            }
        }
        shiftSlotsLeft(startingAt: slot)
        
        // Take outliers from the speaker with the most outliers
        guard let splitSpeaker = activeSpeakers.values.max(by: {
            $0.outlierWeight < $1.outlierWeight
        }), splitSpeaker.hasOutliers
        else {
            guard let removedId else { return }
            return
        }
        
        // Initialize the new speaker profile
        let newSlot = numSlots - 1
        let newId = inactiveSpeakers.count + newSlot
        let newSpeaker = splitSpeaker.extractOutlierProfile(speakerId: newId)
        
        if let removedId {
            newSpeaker.updateCannotLink(with: removedId)
        }
         
        // Check for matches
        if let match = findMatchingSpeaker(for: newSpeaker) {
            newSpeaker.speakerId = match.speakerId
        }
        
        activeSpeakers[newSlot] = newSpeaker
    }
    
    private func findMatchingSpeaker(for speaker: SpeakerProfile) -> SpeakerProfile? {
        var bestDistance: Float = config.matchThreshold.nextUp
        var bestMatch: SpeakerProfile? = nil
        
        let activeIds = activeSpeakers.values.map(\.speakerId)
        
        for candidate in inactiveSpeakers {
            // Skip if the speaker is assigned to someone already
            guard !activeIds.contains(candidate.speakerId) else {
                continue
            }
            
            // Determine how close it is
            let distance = speaker.distance(to: candidate)
            if distance <= bestDistance {
                bestMatch = candidate
                bestDistance = distance
            }
        }
        
        return bestMatch
    }
    
    private func pickSlotToDrop() -> Int? {
        guard activeSpeakers.values.contains(where: \.hasOutliers) else {
            return nil
        }
        
        guard activeSpeakers.count == numSlots else {
            // Return the first unused slot
            var isUnused = Array(repeating: true, count: config.numSlots)
            for slot in activeSpeakers.keys {
                isUnused[slot] = false
            }
            return isUnused.firstIndex(of: true)
        }
        
        guard let droppedSlot = droppableSlots.min(by: {
            guard let end0 = activeSpeakers[$0]?.lastActiveFrame,
                  let end1 = activeSpeakers[$1]?.lastActiveFrame else {
                return false
            }
            return end0 < end1
        }) else {
            return nil
        }
        
        return droppedSlot
    }
    
    private func shiftSlotsLeft(startingAt slot: Int) {
        guard slot < config.numSlots - 1 else {
            return
        }
        
        for i in (slot + 1)..<config.numSlots {
            activeSpeakers[i - 1] = activeSpeakers[i]
        }
        activeSpeakers[config.numSlots - 1] = nil
    }
    
    // TODO: I might need a better ID selection mechanism to minimize ID swaps
    private func updateSpeakerIds() {
        guard !inactiveSpeakers.isEmpty else { return }
        
        let threshold = config.clusteringThreshold

        var numRows = activeSpeakers.count
        var numCols = inactiveSpeakers.count
        
        var costMatrix: [Float] = []
        var rowToSlot: [Int] = []
        var columnToIndex: [Int] = Array(0..<numCols)
        costMatrix.reserveCapacity(numRows * numCols)
        rowToSlot.reserveCapacity(numRows)
        
        var isColumnMatched = Array(repeating: false, count: numCols)
        
        // Build cost matrix
        for (slot, speaker) in activeSpeakers {
            var foundMatch = false
            
            for (i, candidate) in inactiveSpeakers.enumerated() {
                let distance = speaker.distance(to: candidate)
                
                guard distance <= threshold else {
                    costMatrix.append(.infinity)
                    continue
                }
                
                foundMatch = true
                isColumnMatched[i] = true
            }
            
            guard foundMatch else {
                // Undo the row
                costMatrix.removeLast(numCols)
                continue
            }
            
            rowToSlot.append(slot)
        }
        
        numRows = rowToSlot.count
        
        // Note: if numRows > 0, then numCols > 0
        guard numRows > 0 else {
            for (slot, speaker) in activeSpeakers {
                speaker.speakerId = inactiveSpeakers.count + slot
            }
            return
        }
        
        // Remove columns with no matches
        while let removedCol = isColumnMatched.lastIndex(of: false) {
            for row in (0..<numRows).reversed() {
                costMatrix.remove(at: row * numCols + removedCol)
            }
            isColumnMatched.remove(at: removedCol)
            columnToIndex.remove(at: removedCol)
            numCols -= 1
        }
        
        // Solve the assignment
        guard let assignments = solveRectangularLinearAssignment(
            numRows: numRows, numCols: numCols, costMatrix: costMatrix) else {
            fatalError("Failed to solve ID assignments")
        }
        
        // Assign IDs
        var isSlotMatched = Array(repeating: false, count: numSlots)
        for (row, col) in zip(assignments.rows, assignments.cols) {
            let slot = rowToSlot[row]
            let inactiveIndex = columnToIndex[col]
            isSlotMatched[slot] = true
            
            activeSpeakers[row]?.speakerId = inactiveSpeakers[inactiveIndex].speakerId
        }
        
        for (slot, speaker) in activeSpeakers where !isSlotMatched[slot] {
            speaker.speakerId = inactiveSpeakers.count + slot
        }
    }
}
