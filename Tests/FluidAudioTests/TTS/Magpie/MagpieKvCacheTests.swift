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

    /// Drives the slow-path fallback used by `MagpieSynthesizer.runDecoderStep`
    /// when CoreML rejects `outputBackings`. Builds a synthetic feature
    /// provider that mirrors the `decoder_step.mlmodelc` output schema, hands
    /// it to `absorbOutputs`, and verifies the cache front pointers + position
    /// were replaced (i.e. the fallback can take over without `swapBackings`).
    func testAbsorbOutputsReplacesFrontPointers() throws {
        let numLayers = 3
        let maxCacheLength = 16
        let numHeads = 2
        let headDim = 4
        let cache = try MagpieKvCache(
            numLayers: numLayers, maxCacheLength: maxCacheLength,
            numHeads: numHeads, headDim: headDim)

        let preK = (0..<numLayers).map { ObjectIdentifier(cache.cachesK[$0]) }
        let preV = (0..<numLayers).map { ObjectIdentifier(cache.cachesV[$0]) }
        let prePos = (0..<numLayers).map { ObjectIdentifier(cache.positions[$0]) }

        let cacheShape: [NSNumber] = [
            1,
            NSNumber(value: maxCacheLength),
            NSNumber(value: numHeads),
            NSNumber(value: headDim),
        ]
        var features: [String: MLFeatureValue] = [:]
        for i in 0..<numLayers {
            let kArr = try MLMultiArray(shape: cacheShape, dataType: .float16)
            kArr.zeroFillFloat16()
            let vArr = try MLMultiArray(shape: cacheShape, dataType: .float16)
            vArr.zeroFillFloat16()
            let posArr = try MLMultiArray(shape: [1], dataType: .float16)
            posArr.zeroFillFloat16()
            posArr[0] = NSNumber(value: Float(i + 1))
            features[MagpieKvCache.cacheKOutputKeys[i]] = MLFeatureValue(multiArray: kArr)
            features[MagpieKvCache.cacheVOutputKeys[i]] = MLFeatureValue(multiArray: vArr)
            features[MagpieKvCache.positionOutputKeys[i]] = MLFeatureValue(multiArray: posArr)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: features)

        try cache.absorbOutputs(provider)

        for i in 0..<numLayers {
            XCTAssertNotEqual(
                ObjectIdentifier(cache.cachesK[i]), preK[i],
                "absorbOutputs must replace cachesK[\(i)] front pointer")
            XCTAssertNotEqual(
                ObjectIdentifier(cache.cachesV[i]), preV[i],
                "absorbOutputs must replace cachesV[\(i)] front pointer")
            XCTAssertNotEqual(
                ObjectIdentifier(cache.positions[i]), prePos[i],
                "absorbOutputs must replace positions[\(i)] front pointer")
        }
        // positions[0] = 1 → cache.position reads layer-0 scalar.
        XCTAssertEqual(cache.position, 1)
    }

    func testAbsorbOutputsThrowsWhenCacheKOutputMissing() throws {
        let cache = try MagpieKvCache(
            numLayers: 2, maxCacheLength: 8, numHeads: 1, headDim: 2)

        // Provide a feature provider with the wrong key for cache_k_0 so the
        // first lookup fails. This guards the error message users will see
        // when the fallback path is actually exercised.
        let bogus = try MLMultiArray(shape: [1, 8, 1, 2], dataType: .float16)
        bogus.zeroFillFloat16()
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "wrong_key": MLFeatureValue(multiArray: bogus)
        ])

        XCTAssertThrowsError(try cache.absorbOutputs(provider)) { error in
            guard case MagpieError.inferenceFailed(_, let underlying) = error else {
                XCTFail("expected MagpieError.inferenceFailed, got \(error)")
                return
            }
            XCTAssertTrue(
                underlying.contains("missing K cache output key"),
                "underlying should mention the missing K key, got: \(underlying)")
        }
    }
}
