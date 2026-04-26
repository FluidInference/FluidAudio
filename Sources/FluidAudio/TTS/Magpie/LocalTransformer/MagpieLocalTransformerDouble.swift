import Accelerate
import Foundation

/// Double-precision twin of `MagpieLocalTransformer`.
///
/// Mathematically identical to the fp32 path but performs every matmul, softmax,
/// LayerNorm, and GELU in fp64 — closes the residual numerical gap against the
/// NumPy reference (which runs in fp64 by default). Slower than the fp32 path
/// (factor of ~2 on the LT hot loop); enabled per-call via
/// `MagpieSynthesisOptions.useDoublePrecision`.
///
/// Weights are stored as `[Double]` (upcast from fp32 once, on init).
public struct MagpieLocalTransformerDouble: Sendable {

    public let weights: MagpieLocalTransformerWeights
    private let inProjWeightD: [Double]
    private let inProjBiasD: [Double]
    private let posEmbeddingD: [Double]
    private let norm1WeightD: [Double]
    private let norm2WeightD: [Double]
    private let saQkvWeightD: [Double]
    private let saOWeightD: [Double]
    private let ffnConv1WeightD: [Double]
    private let ffnConv2WeightD: [Double]
    private let outProjWeightsD: [[Double]]
    private let outProjBiasesD: [[Double]]

    public init(weights: MagpieLocalTransformerWeights) {
        self.weights = weights
        self.inProjWeightD = weights.inProjWeight.map { Double($0) }
        self.inProjBiasD = weights.inProjBias.map { Double($0) }
        self.posEmbeddingD = weights.posEmbedding.map { Double($0) }
        self.norm1WeightD = weights.norm1Weight.map { Double($0) }
        self.norm2WeightD = weights.norm2Weight.map { Double($0) }
        self.saQkvWeightD = weights.saQkvWeight.map { Double($0) }
        self.saOWeightD = weights.saOWeight.map { Double($0) }
        self.ffnConv1WeightD = weights.ffnConv1Weight.map { Double($0) }
        self.ffnConv2WeightD = weights.ffnConv2Weight.map { Double($0) }
        self.outProjWeightsD = weights.outProjWeights.map { row in row.map { Double($0) } }
        self.outProjBiasesD = weights.outProjBiases.map { row in row.map { Double($0) } }
    }

    // MARK: - Public API (mirrors MagpieLocalTransformer)

    /// Forward pass over a sequence of length `T`. Input/output are `[Float]` to
    /// keep the sampler boundary stable; computation runs in fp64 internally.
    public func forward(sequence: [Float], length T: Int) -> [Float] {
        let D = weights.localDim
        let ffnD = weights.ffnDim
        precondition(sequence.count >= T * D, "sequence buffer too small")
        precondition(T <= weights.maxPositions, "sequence length exceeds maxPositions")

        // Upcast input row prefix to Double + add positional embeddings.
        var x = [Double](repeating: 0, count: T * D)
        for i in 0..<(T * D) { x[i] = Double(sequence[i]) + posEmbeddingD[i] }

        // ── Pre-norm causal self-attention ──
        var xNorm = layerNormD(x, length: T, weight: norm1WeightD)

        // QKV = xNorm @ saQkvWeight.T → (T, 3D)
        var qkv = [Double](repeating: 0, count: T * 3 * D)
        matmulTransBD(
            a: xNorm, aRows: T, aCols: D,
            b: saQkvWeightD, bRows: 3 * D, bCols: D,
            out: &qkv)

        // Split QKV.
        var q = [Double](repeating: 0, count: T * D)
        var k = [Double](repeating: 0, count: T * D)
        var v = [Double](repeating: 0, count: T * D)
        for t in 0..<T {
            let srcOff = t * 3 * D
            let dstOff = t * D
            for i in 0..<D {
                q[dstOff + i] = qkv[srcOff + i]
                k[dstOff + i] = qkv[srcOff + D + i]
                v[dstOff + i] = qkv[srcOff + 2 * D + i]
            }
        }

        // attn = Q @ Kᵀ * scale  (T × T)
        var attn = [Double](repeating: 0, count: T * T)
        matmulTransBD(
            a: q, aRows: T, aCols: D,
            b: k, bRows: T, bCols: D,
            out: &attn)
        let scale = 1.0 / sqrt(Double(D))
        for i in 0..<(T * T) { attn[i] *= scale }

        // Causal mask + softmax (fp64).
        for t in 0..<T {
            var maxVal: Double = -.infinity
            for j in 0...t {
                if attn[t * T + j] > maxVal { maxVal = attn[t * T + j] }
            }
            var denom: Double = 0
            for j in 0..<T {
                if j <= t {
                    let e = exp(attn[t * T + j] - maxVal)
                    attn[t * T + j] = e
                    denom += e
                } else {
                    attn[t * T + j] = 0
                }
            }
            if denom > 0 {
                let invDenom = 1.0 / denom
                for j in 0...t {
                    attn[t * T + j] *= invDenom
                }
            }
        }

        // saOut = attn @ V
        var saOut = [Double](repeating: 0, count: T * D)
        matmulD(
            a: attn, aRows: T, aCols: T,
            b: v, bRows: T, bCols: D,
            out: &saOut)

        // saOut = saOut @ saOWeight.T
        var saProj = [Double](repeating: 0, count: T * D)
        matmulTransBD(
            a: saOut, aRows: T, aCols: D,
            b: saOWeightD, bRows: D, bCols: D,
            out: &saProj)

        // x += saProj
        for i in 0..<(T * D) { x[i] += saProj[i] }

        // ── Pre-norm FFN ──
        xNorm = layerNormD(x, length: T, weight: norm2WeightD)

        // h = gelu(xNorm @ ffnConv1Weight.T)
        var h = [Double](repeating: 0, count: T * ffnD)
        matmulTransBD(
            a: xNorm, aRows: T, aCols: D,
            b: ffnConv1WeightD, bRows: ffnD, bCols: D,
            out: &h)
        applyGeluTanhD(into: &h)

        // x += h @ ffnConv2Weight.T
        var ffnOut = [Double](repeating: 0, count: T * D)
        matmulTransBD(
            a: h, aRows: T, aCols: ffnD,
            b: ffnConv2WeightD, bRows: D, bCols: ffnD,
            out: &ffnOut)
        for i in 0..<(T * D) { x[i] += ffnOut[i] }

        // Downcast back to fp32 at the boundary.
        var out = [Float](repeating: 0, count: T * D)
        for i in 0..<(T * D) { out[i] = Float(x[i]) }
        return out
    }

    /// Project a (dModel,) decoder hidden state through the input projection.
    public func projectInput(hidden: [Float]) -> [Float] {
        precondition(hidden.count == weights.dModel)
        let D = weights.localDim
        let M = weights.dModel

        var hiddenD = [Double](repeating: 0, count: M)
        for i in 0..<M { hiddenD[i] = Double(hidden[i]) }

        var outD = inProjBiasD  // copy bias
        inProjWeightD.withUnsafeBufferPointer { wPtr in
            hiddenD.withUnsafeBufferPointer { hPtr in
                outD.withUnsafeMutableBufferPointer { outPtr in
                    cblas_dgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(D), Int32(M),
                        1.0,
                        wPtr.baseAddress, Int32(M),
                        hPtr.baseAddress, 1,
                        1.0,
                        outPtr.baseAddress, 1)
                }
            }
        }
        var out = [Float](repeating: 0, count: D)
        for i in 0..<D { out[i] = Float(outD[i]) }
        return out
    }

    /// Per-codebook logits (numCodes,) computed from the last LT hidden state.
    public func codebookLogits(lastHidden: [Float], codebook: Int) -> [Float] {
        precondition(lastHidden.count == weights.localDim)
        let numCodes = weights.numCodesPerCodebook
        let D = weights.localDim

        var hiddenD = [Double](repeating: 0, count: D)
        for i in 0..<D { hiddenD[i] = Double(lastHidden[i]) }

        var logitsD = outProjBiasesD[codebook]  // copy bias
        outProjWeightsD[codebook].withUnsafeBufferPointer { wPtr in
            hiddenD.withUnsafeBufferPointer { hPtr in
                logitsD.withUnsafeMutableBufferPointer { outPtr in
                    cblas_dgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(numCodes), Int32(D),
                        1.0,
                        wPtr.baseAddress, Int32(D),
                        hPtr.baseAddress, 1,
                        1.0,
                        outPtr.baseAddress, 1)
                }
            }
        }
        var logits = [Float](repeating: 0, count: numCodes)
        for i in 0..<numCodes { logits[i] = Float(logitsD[i]) }
        return logits
    }

    // MARK: - Private fp64 helpers

    private func layerNormD(_ x: [Double], length T: Int, weight: [Double]) -> [Double] {
        let D = weights.localDim
        var out = [Double](repeating: 0, count: T * D)
        let eps: Double = 1e-5
        for t in 0..<T {
            var mean: Double = 0
            for i in 0..<D { mean += x[t * D + i] }
            mean /= Double(D)
            var variance: Double = 0
            for i in 0..<D {
                let c = x[t * D + i] - mean
                variance += c * c
            }
            variance /= Double(D)
            let invStd = 1.0 / sqrt(variance + eps)
            for i in 0..<D {
                out[t * D + i] = (x[t * D + i] - mean) * invStd * weight[i]
            }
        }
        return out
    }

    /// `out = A @ B`  (M×K) × (K×N) = (M×N).
    private func matmulD(
        a: [Double], aRows M: Int, aCols K: Int,
        b: [Double], bRows: Int, bCols N: Int,
        out: inout [Double]
    ) {
        precondition(K == bRows, "matmulD inner dimension mismatch")
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                out.withUnsafeMutableBufferPointer { outPtr in
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(M), Int32(N), Int32(K),
                        1.0,
                        aPtr.baseAddress, Int32(K),
                        bPtr.baseAddress, Int32(N),
                        0.0,
                        outPtr.baseAddress, Int32(N))
                }
            }
        }
    }

    /// `out = A @ Bᵀ`  (M×K) × (N×K)ᵀ = (M×N); B stored as (N, K).
    private func matmulTransBD(
        a: [Double], aRows M: Int, aCols K: Int,
        b: [Double], bRows N: Int, bCols bk: Int,
        out: inout [Double]
    ) {
        precondition(K == bk, "matmulTransBD inner dimension mismatch")
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                out.withUnsafeMutableBufferPointer { outPtr in
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(M), Int32(N), Int32(K),
                        1.0,
                        aPtr.baseAddress, Int32(K),
                        bPtr.baseAddress, Int32(K),
                        0.0,
                        outPtr.baseAddress, Int32(N))
                }
            }
        }
    }

    /// Tanh-approximation GELU in fp64.
    private func applyGeluTanhD(into buffer: inout [Double]) {
        let n = buffer.count
        let sqrt2pi: Double = 0.7978845608028654
        let coef: Double = 0.044715
        for i in 0..<n {
            let x = buffer[i]
            let inner = sqrt2pi * (x + coef * x * x * x)
            buffer[i] = 0.5 * x * (1 + tanh(inner))
        }
    }
}

// MARK: - Backend dispatch

/// Sampler-side LT backend: chooses between the fp32 and fp64 forward paths.
/// Constructed once per synthesis call and passed into the sampler.
public enum MagpieLtBackend: Sendable {
    case fp32(MagpieLocalTransformer)
    case fp64(MagpieLocalTransformerDouble)

    public var weights: MagpieLocalTransformerWeights {
        switch self {
        case .fp32(let lt): return lt.weights
        case .fp64(let lt): return lt.weights
        }
    }

    public func forward(sequence: [Float], length T: Int) -> [Float] {
        switch self {
        case .fp32(let lt): return lt.forward(sequence: sequence, length: T)
        case .fp64(let lt): return lt.forward(sequence: sequence, length: T)
        }
    }

    public func projectInput(hidden: [Float]) -> [Float] {
        switch self {
        case .fp32(let lt): return lt.projectInput(hidden: hidden)
        case .fp64(let lt): return lt.projectInput(hidden: hidden)
        }
    }

    public func codebookLogits(lastHidden: [Float], codebook: Int) -> [Float] {
        switch self {
        case .fp32(let lt): return lt.codebookLogits(lastHidden: lastHidden, codebook: codebook)
        case .fp64(let lt): return lt.codebookLogits(lastHidden: lastHidden, codebook: codebook)
        }
    }
}
