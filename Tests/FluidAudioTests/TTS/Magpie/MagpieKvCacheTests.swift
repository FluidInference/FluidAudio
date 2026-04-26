import CoreML
import XCTest

@testable import FluidAudio

final class MagpieKvCacheTests: XCTestCase {

    func testInitialShapeAndZeroPosition() throws {
        let cache = try MagpieKvCache(
            numLayers: MagpieConstants.numDecoderLayers,
            maxCacheLength: MagpieConstants.maxCacheLength,
            numHeads: MagpieConstants.numHeads,
            headDim: MagpieConstants.headDim)

        XCTAssertEqual(cache.cachesK.count, MagpieConstants.numDecoderLayers)
        XCTAssertEqual(cache.cachesV.count, MagpieConstants.numDecoderLayers)
        XCTAssertEqual(cache.positions.count, MagpieConstants.numDecoderLayers)
        XCTAssertEqual(cache.position, 0)

        // Rank-4 split-K/V layout: [1, T, H, D] per cache tensor.
        let expectedShape: [NSNumber] = [
            1,
            NSNumber(value: MagpieConstants.maxCacheLength),
            NSNumber(value: MagpieConstants.numHeads),
            NSNumber(value: MagpieConstants.headDim),
        ]
        XCTAssertEqual(cache.cachesK[0].shape, expectedShape)
        XCTAssertEqual(cache.cachesV[0].shape, expectedShape)
        XCTAssertEqual(cache.positions[0].shape, [1])
    }

    func testAddInputsProvidesAllLayerKeys() throws {
        let cache = try MagpieKvCache(
            numLayers: 3, maxCacheLength: 32, numHeads: 4, headDim: 8)
        var inputs: [String: MLMultiArray] = [:]
        cache.addInputs(to: &inputs)
        // 3 layers × (cache_k, cache_v, position) = 9 entries.
        XCTAssertEqual(inputs.count, 9)
        for i in 0..<3 {
            XCTAssertNotNil(inputs["cache_k\(i)"])
            XCTAssertNotNil(inputs["cache_v\(i)"])
            XCTAssertNotNil(inputs["position\(i)"])
        }
    }

    func testStaticOutputKeyCountMatchesLayers() {
        XCTAssertEqual(
            MagpieKvCache.cacheKOutputKeys.count, MagpieConstants.numDecoderLayers,
            "cacheKOutputKeys must match numDecoderLayers — regenerate list if the exporter changes.")
        XCTAssertEqual(
            MagpieKvCache.cacheVOutputKeys.count, MagpieConstants.numDecoderLayers,
            "cacheVOutputKeys must match numDecoderLayers — regenerate list if the exporter changes.")
        XCTAssertEqual(
            MagpieKvCache.positionOutputKeys.count, MagpieConstants.numDecoderLayers)
    }
}
