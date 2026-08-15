import Foundation
import XCTest

@testable import FluidAudio

final class OfflineSortformerTests: XCTestCase {

    // MARK: - SortformerSpeakerStitcher

    /// Window speakers already in global order -> identity mapping.
    func testStitcherIdentityWhenAligned() {
        let frames = 4
        let ns = 4
        // One-hot active speaker per frame, same order in both.
        var global = [Float](repeating: 0, count: frames * ns)
        for f in 0..<frames { global[f * ns + (f % ns)] = 1 }
        let window = global

        let mapping = SortformerSpeakerStitcher.alignment(
            global: global, window: window, frames: frames, numSpeakers: ns)
        XCTAssertEqual(mapping, [0, 1, 2, 3])
    }

    /// Window columns are a known permutation of global -> stitcher recovers the inverse mapping.
    func testStitcherRecoversPermutation() {
        let frames = 8
        let ns = 4
        // global speaker g active on frame g (and g+4). window uses permutation perm[g]=w.
        let perm = [2, 0, 3, 1]  // global g -> window column perm[g]
        var global = [Float](repeating: 0, count: frames * ns)
        var window = [Float](repeating: 0, count: frames * ns)
        for f in 0..<frames {
            let g = f % ns
            global[f * ns + g] = 1
            window[f * ns + perm[g]] = 1
        }

        let mapping = SortformerSpeakerStitcher.alignment(
            global: global, window: window, frames: frames, numSpeakers: ns)

        // mapping[windowColumn] == globalSpeaker. Remapping window via mapping must equal global.
        for f in 0..<frames {
            for w in 0..<ns where window[f * ns + w] > 0 {
                XCTAssertEqual(mapping[w], f % ns, "window col \(w) should map to global \(f % ns)")
            }
        }
    }

    /// Soft (non one-hot) activity still aligns by maximum correlation.
    func testStitcherSoftActivity() {
        let frames = 3
        let ns = 4
        // global: speaker 1 dominant; window: column 3 dominant -> col3 should map to global 1.
        var global = [Float](repeating: 0.1, count: frames * ns)
        var window = [Float](repeating: 0.1, count: frames * ns)
        for f in 0..<frames {
            global[f * ns + 1] = 0.9
            window[f * ns + 3] = 0.9
        }
        let mapping = SortformerSpeakerStitcher.alignment(
            global: global, window: window, frames: frames, numSpeakers: ns)
        XCTAssertEqual(mapping[3], 1)
    }

    /// No overlap frames -> identity (nothing to align on).
    func testStitcherZeroFramesIsIdentity() {
        let mapping = SortformerSpeakerStitcher.alignment(
            global: [], window: [], frames: 0, numSpeakers: 4)
        XCTAssertEqual(mapping, [0, 1, 2, 3])
    }

    /// Mapping is always a valid bijection of speaker indices.
    func testStitcherMappingIsBijection() {
        let frames = 5
        let ns = 4
        var global = [Float](repeating: 0, count: frames * ns)
        var window = [Float](repeating: 0, count: frames * ns)
        for f in 0..<frames {
            global[f * ns + (f % ns)] = Float(f + 1)
            window[f * ns + ((f + 2) % ns)] = Float(f + 1)
        }
        let mapping = SortformerSpeakerStitcher.alignment(
            global: global, window: window, frames: frames, numSpeakers: ns)
        XCTAssertEqual(Set(mapping), Set(0..<ns), "mapping must be a permutation")
    }

    // MARK: - OfflineSortformerConfig

    func testOfflineConfigDefaults() {
        let config = OfflineSortformerConfig.offlineV2_1
        XCTAssertEqual(config.precision, .fp16)
        XCTAssertEqual(config.windowOutputFrames, 384)
        XCTAssertEqual(config.subsamplingFactor, 8)
        XCTAssertEqual(config.windowMelFrames, 384 * 8)  // 3072
        XCTAssertEqual(config.numSpeakers, 4)
        XCTAssertEqual(config.frameDurationSeconds, 0.08, accuracy: 1e-6)
        XCTAssertEqual(config.overlapOutputFrames, 100)
    }

    func testOfflineConfigPrecisionSelectsSubdir() {
        var config = OfflineSortformerConfig.offlineV2_1
        XCTAssertEqual(config.precision.subdirectory, "v3/fp16")
        config.precision = .palettized
        XCTAssertEqual(config.precision.subdirectory, "v3/palettized")
    }

    // MARK: - ModelNames offline bundle

    func testOfflineBundlePaths() {
        XCTAssertEqual(
            ModelNames.Sortformer.offlineBundle(precision: .fp16),
            "v3/fp16/SortformerOffline_v2.1.mlmodelc")
        XCTAssertEqual(
            ModelNames.Sortformer.offlineBundle(precision: .palettized),
            "v3/palettized/SortformerOffline_v2.1.mlmodelc")
        // Default precision is fp16.
        XCTAssertEqual(
            ModelNames.Sortformer.offlineBundle(),
            "v3/fp16/SortformerOffline_v2.1.mlmodelc")
    }
}
