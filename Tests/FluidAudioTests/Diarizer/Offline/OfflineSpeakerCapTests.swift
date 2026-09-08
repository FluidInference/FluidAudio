import XCTest

@testable import FluidAudio

/// End-to-end checks that a speaker count request survives every stage below the
/// VBx refinement: the constraint gate, centroid construction, and assignment.
///
/// These run the real assignment rather than inspecting an intermediate census,
/// because that is where a cap is finally kept or broken — `Set(assignments).count`
/// is the speaker count a caller observes.
@available(macOS 14.0, iOS 17.0, *)
final class OfflineSpeakerCapTests: XCTestCase {

    /// Two embeddings sharing one segmentation chunk.
    private let embeddings = [[1.0, 0.0], [0.0, 1.0]]
    private let chunkIndices = [0, 0]
    private let initialClusters = [0, 1]

    /// VBx keeps two mixture components, but column 1 wins no embedding's argmax:
    /// `assignedClusterCount` is 1 while `activeClusterCount` is 2.
    private func splitOutput() -> VBxOutput {
        VBxOutput(
            gamma: [[0.6, 0.4], [0.6, 0.4]],
            pi: [0.6, 0.4],
            hardClusters: [[0, 0]],
            centroids: [],
            numClusters: 2,
            elbos: []
        )
    }

    private func assignments(
        for output: VBxOutput,
        numSpeakers: Int?,
        maxSpeakers: Int? = nil
    ) -> [Int] {
        let constraints: SpeakerCountConstraints? =
            (numSpeakers == nil && maxSpeakers == nil)
            ? nil
            : SpeakerCountConstraints.resolve(
                numEmbeddings: embeddings.count,
                numSpeakers: numSpeakers,
                minSpeakers: nil,
                maxSpeakers: maxSpeakers
            )

        let constrained = VBxClustering.applyConstraints(
            to: output,
            trainingEmbeddings: embeddings,
            constraints: constraints
        )

        return OfflineDiarizerManager().clusterAssignments(
            vbxOutput: constrained,
            trainingEmbeddings: embeddings,
            embeddingFeatures: embeddings,
            initialClusters: initialClusters,
            chunkIndices: chunkIndices
        ).assignments
    }

    // MARK: - The cap must hold

    func testNumSpeakersOneYieldsOneSpeaker() {
        let result = assignments(for: splitOutput(), numSpeakers: 1)
        XCTAssertEqual(
            Set(result).count, 1,
            "numSpeakers: 1 must not produce \(Set(result).count) speakers")
    }

    func testMaxSpeakersOneYieldsOneSpeaker() {
        let result = assignments(for: splitOutput(), numSpeakers: nil, maxSpeakers: 1)
        XCTAssertEqual(
            Set(result).count, 1,
            "maxSpeakers: 1 must not produce \(Set(result).count) speakers")
    }

    // MARK: - pyannote parity when nothing was requested

    /// Without a speaker count request, pyannote keeps every `sp > 1e-7` component
    /// as a centroid and lets constrained assignment place a co-chunk speaker on a
    /// component that won no argmax. Tightening the centroid census would silently
    /// change this default-config result, so it is pinned here.
    func testUnconstrainedRunKeepsBothMixtureComponents() {
        let result = assignments(for: splitOutput(), numSpeakers: nil)
        XCTAssertEqual(
            Set(result).count, 2,
            "an unconstrained run must keep pyannote's centroid census")
    }

    // MARK: - The adjustment itself

    /// #802: a request the argmax census misses must still re-cluster, and the
    /// count reported as detected is that census — not the wider pi census.
    func testUnderCountReclustersOnTheArgmaxCensus() {
        // Both rows win column 0, so one cluster is assigned while two survive.
        let output = VBxOutput(
            gamma: [[0.9, 0.1], [0.8, 0.2]],
            pi: [0.7, 0.3],
            hardClusters: [[0, 0]],
            centroids: [],
            numClusters: 2,
            elbos: []
        )
        let constraints = SpeakerCountConstraints.resolve(
            numEmbeddings: embeddings.count,
            numSpeakers: nil,
            minSpeakers: 2,
            maxSpeakers: nil
        )

        let adjusted = VBxClustering.applyConstraints(
            to: output,
            trainingEmbeddings: embeddings,
            constraints: constraints
        )

        XCTAssertTrue(adjusted.wasAdjusted)
        XCTAssertEqual(adjusted.numClusters, 2)
        XCTAssertEqual(
            adjusted.originalClusterCount, 1,
            "the argmax census is what the adjustment reports as detected")
    }

    /// #801: when both censuses already fit, nothing is re-clustered.
    func testBothCensusesWithinBoundsLeaveTheOutputUntouched() {
        let constraints = SpeakerCountConstraints.resolve(
            numEmbeddings: embeddings.count,
            numSpeakers: nil,
            minSpeakers: nil,
            maxSpeakers: 2
        )

        let adjusted = VBxClustering.applyConstraints(
            to: splitOutput(),
            trainingEmbeddings: embeddings,
            constraints: constraints
        )

        XCTAssertFalse(adjusted.wasAdjusted, "VBx already agrees; K-Means must not run")
        XCTAssertNil(adjusted.originalClusterCount)
    }
}
