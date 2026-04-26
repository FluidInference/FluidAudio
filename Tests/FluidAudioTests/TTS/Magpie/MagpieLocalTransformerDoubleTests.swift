import XCTest

@testable import FluidAudio

/// Validates that `MagpieLocalTransformerDouble` agrees with the fp32
/// `MagpieLocalTransformer` to within fp32-roundoff and that the
/// `MagpieLtBackend` enum dispatches correctly.
final class MagpieLocalTransformerDoubleTests: XCTestCase {

    // MARK: - Synthetic weights

    /// Build a tiny, deterministic LT weight set sized so the forward pass is
    /// cheap to run in unit tests. The shapes match the production layout but
    /// scaled down (localDim=8, ffnDim=16, dModel=8, codebooks=2, codes=4).
    private func makeWeights() -> MagpieLocalTransformerWeights {
        let localDim = 8
        let ffnDim = 16
        let dModel = 8
        let maxPositions = 4
        let numCodebooks = 2
        let numCodes = 4

        // Deterministic pseudo-random fill in [-0.1, 0.1].
        var rng = MagpieMT19937(seed: 7)
        func arr(_ count: Int) -> [Float] {
            (0..<count).map { _ in
                Float(rng.uniformDouble() * 0.2 - 0.1)
            }
        }
        // LayerNorm weights initialized near 1.0 (matches PyTorch default).
        func ones(_ count: Int) -> [Float] {
            Array(repeating: 1.0, count: count)
        }

        let outProjWeights = (0..<numCodebooks).map { _ in arr(numCodes * localDim) }
        let outProjBiases = (0..<numCodebooks).map { _ in arr(numCodes) }

        return MagpieLocalTransformerWeights(
            inProjWeight: arr(localDim * dModel),
            inProjBias: arr(localDim),
            posEmbedding: arr(maxPositions * localDim),
            norm1Weight: ones(localDim),
            norm2Weight: ones(localDim),
            saQkvWeight: arr(3 * localDim * localDim),
            saOWeight: arr(localDim * localDim),
            ffnConv1Weight: arr(ffnDim * localDim),
            ffnConv2Weight: arr(localDim * ffnDim),
            outProjWeights: outProjWeights,
            outProjBiases: outProjBiases,
            localDim: localDim,
            dModel: dModel,
            ffnDim: ffnDim,
            maxPositions: maxPositions,
            numCodebooks: numCodebooks,
            numCodesPerCodebook: numCodes
        )
    }

    private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var m: Float = 0
        for i in 0..<a.count {
            let d = abs(a[i] - b[i])
            if d > m { m = d }
        }
        return m
    }

    // MARK: - Tests

    func testProjectInputAgreesAcrossPrecisions() {
        let weights = makeWeights()
        let lt32 = MagpieLocalTransformer(weights: weights)
        let lt64 = MagpieLocalTransformerDouble(weights: weights)

        let hidden = (0..<weights.dModel).map { i -> Float in
            Float(i) * 0.01 - 0.05
        }
        let p32 = lt32.projectInput(hidden: hidden)
        let p64 = lt64.projectInput(hidden: hidden)
        XCTAssertEqual(p32.count, p64.count)
        // fp32 vs fp64 within fp32 roundoff (single GEMV + bias).
        XCTAssertLessThan(maxAbsDiff(p32, p64), 1e-5)
    }

    func testForwardAgreesAcrossPrecisions() {
        let weights = makeWeights()
        let lt32 = MagpieLocalTransformer(weights: weights)
        let lt64 = MagpieLocalTransformerDouble(weights: weights)

        let seqLen = 3
        let dim = weights.localDim
        var rng = MagpieMT19937(seed: 99)
        let seq = (0..<(seqLen * dim)).map { _ in Float(rng.uniformDouble() * 0.2 - 0.1) }

        let out32 = lt32.forward(sequence: seq, length: seqLen)
        let out64 = lt64.forward(sequence: seq, length: seqLen)
        XCTAssertEqual(out32.count, out64.count)
        // Multi-stage compute (matmul + softmax + matmul + GELU + matmul) drifts
        // a bit more, but should stay within ~1e-4 for the small test sizes.
        XCTAssertLessThan(maxAbsDiff(out32, out64), 1e-4)
    }

    func testCodebookLogitsAgreeAcrossPrecisions() {
        let weights = makeWeights()
        let lt32 = MagpieLocalTransformer(weights: weights)
        let lt64 = MagpieLocalTransformerDouble(weights: weights)

        let hidden = (0..<weights.localDim).map { i -> Float in
            Float(i) * 0.01 - 0.05
        }
        for cb in 0..<weights.numCodebooks {
            let l32 = lt32.codebookLogits(lastHidden: hidden, codebook: cb)
            let l64 = lt64.codebookLogits(lastHidden: hidden, codebook: cb)
            XCTAssertEqual(l32.count, l64.count)
            XCTAssertLessThan(maxAbsDiff(l32, l64), 1e-5)
        }
    }

    // MARK: - Backend enum dispatch

    func testBackendEnumDispatchesToFp32() {
        let weights = makeWeights()
        let backend = MagpieLtBackend.fp32(MagpieLocalTransformer(weights: weights))
        let hidden = [Float](repeating: 0.0, count: weights.dModel)
        let out = backend.projectInput(hidden: hidden)
        XCTAssertEqual(out.count, weights.localDim)
        // With zero hidden + bias only, output should equal the bias.
        XCTAssertEqual(out, weights.inProjBias)
        XCTAssertEqual(backend.weights.localDim, weights.localDim)
    }

    func testBackendEnumDispatchesToFp64() {
        let weights = makeWeights()
        let backend = MagpieLtBackend.fp64(MagpieLocalTransformerDouble(weights: weights))
        let hidden = [Float](repeating: 0.0, count: weights.dModel)
        let out = backend.projectInput(hidden: hidden)
        XCTAssertEqual(out.count, weights.localDim)
        // fp64 round-trips bias exactly.
        for i in 0..<out.count {
            XCTAssertEqual(out[i], weights.inProjBias[i], accuracy: 1e-7)
        }
    }

    // MARK: - Synthesis-options plumbing

    func testSynthesisOptionsHasUseDoublePrecisionDefaultFalse() {
        XCTAssertEqual(MagpieSynthesisOptions.default.useDoublePrecision, false)
        let opts = MagpieSynthesisOptions(useDoublePrecision: true)
        XCTAssertTrue(opts.useDoublePrecision)
    }
}
