import XCTest

@testable import FluidAudio

/// Unit tests for the pure clean-region frame selection (TitaNet `.cleanRegion`
/// pooling). No CoreML — operates on synthetic mel-frame masks.
final class CleanRegionSelectorTests: XCTestCase {

    /// Length-`n` mel mask with the given [start, end) runs set to 1.0.
    private func mask(_ n: Int, _ runs: [(Int, Int)]) -> [Float] {
        var m = [Float](repeating: 0, count: n)
        for (s, e) in runs { for i in s..<e { m[i] = 1 } }
        return m
    }

    /// A single contiguous run ≥ target is embedded in full (not capped).
    func testSingleLongRunEmbeddedInFull() {
        let m = mask(1000, [(100, 500)])  // 400 clean frames
        let f = CleanRegionSelector.selectCleanMelFrames(
            cleanMaskMel: m, collarLead: 0, collarTrail: 0,
            regionMin: 50, embedMin: 200, embedTarget: 300)
        XCTAssertEqual(f?.count, 400)
        XCTAssertEqual(f?.first, 100)
        XCTAssertEqual(f?.last, 499)
    }

    /// Two sub-target runs are concatenated in time order up to the target.
    func testTwoRunsConcatToTarget() {
        let m = mask(1000, [(100, 250), (400, 550)])  // two 150-frame runs
        let f = CleanRegionSelector.selectCleanMelFrames(
            cleanMaskMel: m, collarLead: 0, collarTrail: 0,
            regionMin: 50, embedMin: 200, embedTarget: 300)
        XCTAssertEqual(f?.count, 300)  // 150 + 150 capped at target
        XCTAssertEqual(Array(f![0..<150]), Array(100..<250))
        XCTAssertEqual(Array(f![150..<300]), Array(400..<550))
    }

    /// Total clean below embedMin → skip the speaker.
    func testBelowEmbedMinSkips() {
        let m = mask(1000, [(100, 220)])  // 120 clean frames < embedMin 200
        let f = CleanRegionSelector.selectCleanMelFrames(
            cleanMaskMel: m, collarLead: 0, collarTrail: 0,
            regionMin: 50, embedMin: 200, embedTarget: 300)
        XCTAssertNil(f)
    }

    /// A run shorter than regionMin is dropped; no runs survive → skip.
    func testShortRunDroppedByRegionMin() {
        let m = mask(1000, [(100, 140)])  // 40-frame run < regionMin 50
        let f = CleanRegionSelector.selectCleanMelFrames(
            cleanMaskMel: m, collarLead: 0, collarTrail: 0,
            regionMin: 50, embedMin: 30, embedTarget: 300)
        XCTAssertNil(f)
    }

    /// Collar erodes inward from both edges of a run.
    func testCollarErodesInward() {
        let m = mask(1000, [(100, 200)])  // 100-frame run
        let f = CleanRegionSelector.selectCleanMelFrames(
            cleanMaskMel: m, collarLead: 10, collarTrail: 10,
            regionMin: 50, embedMin: 50, embedTarget: 300)
        XCTAssertEqual(f?.count, 80)  // 100 - 10 - 10
        XCTAssertEqual(f?.first, 110)
        XCTAssertEqual(f?.last, 189)
    }

    /// The mel-frame packer selects the requested columns and zero-pads the tail.
    func testPackColumnsSelectsAndZeroPads() {
        // 2 mel bins × 4 src frames, row-major.
        let src: [Float] = [10, 11, 12, 13, 20, 21, 22, 23]
        let packed = CleanRegionSelector.packColumns(
            src: src, melBins: 2, srcFrames: 4, dstFrames: 4, frames: [1, 3])
        XCTAssertEqual(packed, [11, 13, 0, 0, 21, 23, 0, 0])
    }

    // MARK: - .cleanWaveform sample-region selection

    /// `[[Float]]` weights (frameCount × speakerCount) with the given per-speaker
    /// [start, end) active runs set to 1.0.
    private func weights(
        _ frameCount: Int, _ speakerCount: Int, _ active: [Int: [(Int, Int)]]
    ) -> [[Float]] {
        var w = [[Float]](
            repeating: [Float](repeating: 0, count: speakerCount), count: frameCount)
        for (spk, runs) in active { for (s, e) in runs { for f in s..<e { w[f][spk] = 1 } } }
        return w
    }

    /// A single clean run maps to one sample region; frame bounds are the clean span.
    func testSampleRegionsSingleRun() {
        // 100 frames over 16000 samples → spf = 160.
        let w = weights(100, 2, [0: [(10, 60)]])
        let out = CleanRegionSelector.selectCleanSampleRegions(
            weights: w, speaker: 0, availableSamples: 16000,
            regionMinSamples: 800, targetSamples: 48000, embedMinSamples: 1600, threshold: 0.5)
        XCTAssertEqual(out?.regions.count, 1)
        XCTAssertEqual(out?.regions.first?.start, 1600)  // 10 * 160
        XCTAssertEqual(out?.regions.first?.end, 9600)  // 60 * 160
        XCTAssertEqual(out?.firstFrame, 10)
        XCTAssertEqual(out?.lastFrame, 59)
    }

    /// Frames where a second speaker is active are excluded, splitting the run.
    func testSampleRegionsExcludesOverlap() {
        // spk0 active [10,60); spk1 active [30,40) → clean = [10,30) ∪ [40,60).
        let w = weights(100, 2, [0: [(10, 60)], 1: [(30, 40)]])
        let out = CleanRegionSelector.selectCleanSampleRegions(
            weights: w, speaker: 0, availableSamples: 16000,
            regionMinSamples: 800, targetSamples: 48000, embedMinSamples: 1600, threshold: 0.5)
        XCTAssertEqual(out?.regions.count, 2)
        XCTAssertEqual(out?.regions.first?.start, 1600)  // 10 * 160
        XCTAssertEqual(out?.regions.first?.end, 4800)  // 30 * 160
        XCTAssertEqual(out?.regions.last?.start, 6400)  // 40 * 160
        XCTAssertEqual(out?.regions.last?.end, 9600)  // 60 * 160
        XCTAssertEqual(out?.firstFrame, 10)
        XCTAssertEqual(out?.lastFrame, 59)
    }

    /// Total clean samples below embedMin → skip the speaker.
    func testSampleRegionsBelowEmbedMinSkips() {
        let w = weights(100, 2, [0: [(10, 20)]])  // 10 frames → 1600 samples
        let out = CleanRegionSelector.selectCleanSampleRegions(
            weights: w, speaker: 0, availableSamples: 16000,
            regionMinSamples: 800, targetSamples: 48000, embedMinSamples: 3200, threshold: 0.5)
        XCTAssertNil(out)
    }

    /// A run shorter than regionMin is dropped; nothing survives → skip.
    func testSampleRegionsShortRunDropped() {
        let w = weights(100, 2, [0: [(10, 15)]])  // 5 frames → 800 samples
        let out = CleanRegionSelector.selectCleanSampleRegions(
            weights: w, speaker: 0, availableSamples: 16000,
            regionMinSamples: 1600, targetSamples: 48000, embedMinSamples: 1600, threshold: 0.5)
        XCTAssertNil(out)
    }

    /// The feature-domain concat packer lays crops end-to-end and zero-pads the tail.
    func testPackConcatFeatures() {
        // Two crops, 2 mel bins each. Crop1: bin0=[1,2] bin1=[10,20]; Crop2: bin0=[3,4] bin1=[30,40].
        let col1: [Float] = [1, 2, 10, 20]  // mel-major [2 bins × 2 frames]
        let col2: [Float] = [3, 4, 30, 40]
        let out = CleanRegionSelector.packConcatFeatures(
            columns: [(col1, 2), (col2, 2)], melBins: 2, dstFrames: 5)
        XCTAssertEqual(out.usedFrames, 4)
        XCTAssertEqual(out.feats, [1, 2, 3, 4, 0, /* bin1 */ 10, 20, 30, 40, 0])
    }

    /// Concat truncates at dstFrames.
    func testPackConcatFeaturesTruncates() {
        let col: [Float] = [1, 2, 3, 4, 10, 20, 30, 40]  // [2 bins × 4 frames]
        let out = CleanRegionSelector.packConcatFeatures(
            columns: [(col, 4)], melBins: 2, dstFrames: 3)
        XCTAssertEqual(out.usedFrames, 3)
        XCTAssertEqual(out.feats, [1, 2, 3, /* bin1 */ 10, 20, 30])
    }

    /// numpy-'reflect' padding mirrors around the edge sample without repeating it.
    @available(macOS 14.0, iOS 17.0, *)
    func testReflectPadMatchesNumpy() {
        // np.pad([1,2,3,4,5], 2, 'reflect') == [3,2,1,2,3,4,5,4,3]
        let out = TitaNetWaveCropFeaturizer.reflectPad([1, 2, 3, 4, 5], 2)
        XCTAssertEqual(out, [3, 2, 1, 2, 3, 4, 5, 4, 3])
    }
}
