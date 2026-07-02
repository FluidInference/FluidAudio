import Foundation
import XCTest

@testable import FluidAudio

/// Tests for the long-form chunk stitcher (`CanaryManager.mergeTokenStreams`),
/// which joins adjacent 15 s window token streams at their overlapping seam.
final class CanaryChunkMergeTests: XCTestCase {

    func testEmptyPrefixReturnsSuffix() {
        XCTAssertEqual(CanaryManager.mergeTokenStreams(prefix: [], suffix: [1, 2, 3]), [1, 2, 3])
    }

    func testEmptySuffixReturnsPrefix() {
        XCTAssertEqual(CanaryManager.mergeTokenStreams(prefix: [1, 2, 3], suffix: []), [1, 2, 3])
    }

    func testCleanSeamDropsDuplicatedOverlap() {
        // prefix ends with [7,8,9,10]; suffix starts with the same overlap then continues.
        let prefix = [1, 2, 3, 7, 8, 9, 10]
        let suffix = [7, 8, 9, 10, 11, 12]
        let merged = CanaryManager.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 7, 8, 9, 10, 11, 12])
    }

    func testPartialOverlapAlignsAtLongestMatch() {
        // Overlap is the 5-token run [20,21,22,23,24]; the leading 99 in suffix is noise.
        let prefix = [1, 2, 20, 21, 22, 23, 24]
        let suffix = [99, 20, 21, 22, 23, 24, 30, 31]
        let merged = CanaryManager.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 20, 21, 22, 23, 24, 30, 31])
    }

    func testNoMatchConcatenatesPlainly() {
        // No shared run ≥ minMatch — better to duplicate than lose content.
        let prefix = [1, 2, 3]
        let suffix = [4, 5, 6]
        XCTAssertEqual(CanaryManager.mergeTokenStreams(prefix: prefix, suffix: suffix), [1, 2, 3, 4, 5, 6])
    }

    func testShortMatchBelowThresholdIsNotTrusted() {
        // A 3-token incidental match is below the default minMatch=4 → plain concat.
        let prefix = [1, 5, 6, 7]
        let suffix = [5, 6, 7, 8, 9]
        let merged = CanaryManager.mergeTokenStreams(prefix: prefix, suffix: suffix, minMatch: 4)
        XCTAssertEqual(merged, [1, 5, 6, 7, 5, 6, 7, 8, 9])
    }

    func testMatchAtThresholdIsTrusted() {
        let prefix = [1, 5, 6, 7, 8]
        let suffix = [5, 6, 7, 8, 9]
        let merged = CanaryManager.mergeTokenStreams(prefix: prefix, suffix: suffix, minMatch: 4)
        XCTAssertEqual(merged, [1, 5, 6, 7, 8, 9])
    }

    func testOverlapConfigContract() {
        XCTAssertEqual(CanaryConfig.chunkOverlapSamples, 48_000)  // 3 s @ 16 kHz
        XCTAssertEqual(
            CanaryConfig.chunkOverlapSamples,
            Int(CanaryConfig.chunkOverlapSeconds * Double(CanaryConfig.sampleRate)))
        XCTAssertLessThan(CanaryConfig.chunkOverlapSamples, CanaryConfig.maxSamples)
    }
}
