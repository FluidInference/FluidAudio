import Accelerate
import Foundation

/// Swift-side 1-layer Local Transformer forward pass.
///
/// Mirrors `local_transformer_forward` in
/// `mobius/models/tts/magpie/coreml/generate_coreml.py` (lines 108–155):
/// pre-norm causal self-attention → pre-norm FFN with tanh-GELU. Single attention
/// head, localDim=256. Uses BLAS (`cblas_sgemm`) for every matmul so the AR loop
/// stays cache-resident.
///
/// The transformer is stateless across frames — each call to
/// `MagpieLocalTransformerSampler.sample(...)` rebuilds the sequence from the
/// current decoder hidden state and the 8 tokens sampled so far.
public struct MagpieLocalTransformer: Sendable {

    public let weights: MagpieLocalTransformerWeights

    public init(weights: MagpieLocalTransformerWeights) {
        self.weights = weights
    }

    /// Forward pass for a sequence of length `T` (T ≤ numCodebooks+2).
    ///
    /// - Parameter sequence: `[T * localDim]` row-major fp32 (input sequence
    ///   including positional embeddings yet to be added — this routine adds them).
    ///   Caller must supply `T` explicitly to avoid ambiguity on partial buffers.
    /// - Returns: `[T * localDim]` row-major output.
    public func forward(sequence: [Float], length T: Int) -> [Float] {
        precondition(sequence.count >= T * weights.localDim, "sequence buffer too small")
        precondition(T <= weights.maxPositions, "sequence length exceeds maxPositions")

        let D = weights.localDim
        let ffnD = weights.ffnDim

        // x = sequence[:T*D] + posEmbedding[:T*D]
        var x = Swift.Array(sequence.prefix(T * D))
        addPositional(into: &x, length: T)

        // ── Pre-norm causal self-attention ──
        var xNorm = layerNorm(x, length: T, weight: weights.norm1Weight)

        // QKV = xNorm @ sa_qkv_weight.T   (T,D) × (3D,D)ᵀ → (T, 3D)
        var qkv = Swift.Array<Float>(repeating: 0, count: T * 3 * D)
        matmulTransB(
            a: xNorm, aRows: T, aCols: D,
            b: weights.saQkvWeight, bRows: 3 * D, bCols: D,
            out: &qkv)

        // Split QKV into Q, K, V (each T × D)
        var q = Swift.Array<Float>(repeating: 0, count: T * D)
        var k = Swift.Array<Float>(repeating: 0, count: T * D)
        var v = Swift.Array<Float>(repeating: 0, count: T * D)
        for t in 0..<T {
            let srcOff = t * 3 * D
            let dstOff = t * D
            memcpy(&q[dstOff], Swift.Array(qkv[srcOff..<(srcOff + D)]), D * MemoryLayout<Float>.size)
            memcpy(&k[dstOff], Swift.Array(qkv[(srcOff + D)..<(srcOff + 2 * D)]), D * MemoryLayout<Float>.size)
            memcpy(&v[dstOff], Swift.Array(qkv[(srcOff + 2 * D)..<(srcOff + 3 * D)]), D * MemoryLayout<Float>.size)
        }

        // attn = Q @ Kᵀ * scale  (T × T)
        var attn = Swift.Array<Float>(repeating: 0, count: T * T)
        matmulTransB(
            a: q, aRows: T, aCols: D,
            b: k, bRows: T, bCols: D,
            out: &attn)
        let scale = Float(1.0 / sqrt(Double(D)))
        var scaleVar = scale
        vDSP_vsmul(attn, 1, &scaleVar, &attn, 1, vDSP_Length(T * T))

        // Causal mask + softmax
        for t in 0..<T {
            // Mask out positions > t (future). Then softmax over [0, t].
            var maxVal: Float = -.infinity
            for j in 0...t {
                if attn[t * T + j] > maxVal { maxVal = attn[t * T + j] }
            }
            var denom: Float = 0
            for j in 0..<T {
                if j <= t {
                    let e = expf(attn[t * T + j] - maxVal)
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

        // saOut = attn @ V      (T × T) × (T × D) → (T × D)
        var saOut = Swift.Array<Float>(repeating: 0, count: T * D)
        matmul(
            a: attn, aRows: T, aCols: T,
            b: v, bRows: T, bCols: D,
            out: &saOut)

        // saOut = saOut @ sa_o_weight.T    (T, D) × (D, D)ᵀ → (T, D)
        var saProj = Swift.Array<Float>(repeating: 0, count: T * D)
        matmulTransB(
            a: saOut, aRows: T, aCols: D,
            b: weights.saOWeight, bRows: D, bCols: D,
            out: &saProj)

        // x += saProj
        vDSP_vadd(x, 1, saProj, 1, &x, 1, vDSP_Length(T * D))

        // ── Pre-norm FFN ──
        xNorm = layerNorm(x, length: T, weight: weights.norm2Weight)

        // h = gelu(xNorm @ ffn_conv1_weight.T)  → (T, ffnD)
        var h = Swift.Array<Float>(repeating: 0, count: T * ffnD)
        matmulTransB(
            a: xNorm, aRows: T, aCols: D,
            b: weights.ffnConv1Weight, bRows: ffnD, bCols: D,
            out: &h)
        applyGeluTanh(into: &h)

        // x += h @ ffn_conv2_weight.T           → (T, D)
        var ffnOut = Swift.Array<Float>(repeating: 0, count: T * D)
        matmulTransB(
            a: h, aRows: T, aCols: ffnD,
            b: weights.ffnConv2Weight, bRows: D, bCols: ffnD,
            out: &ffnOut)
        vDSP_vadd(x, 1, ffnOut, 1, &x, 1, vDSP_Length(T * D))

        return x
    }

    /// Project a (dModel,) decoder hidden state through the input projection
    /// → (localDim,). Used by the sampler to seed the LT sequence.
    public func projectInput(hidden: [Float]) -> [Float] {
        precondition(hidden.count == weights.dModel)
        var out = weights.inProjBias  // copy bias
        // out += inProjWeight @ hidden  (localDim, dModel) × (dModel,) → (localDim,)
        inProjWeightApply(hidden: hidden, accumulate: &out)
        return out
    }

    /// Compute logits for codebook `cb`: last-timestep out_proj head.
    public func codebookLogits(lastHidden: [Float], codebook: Int) -> [Float] {
        precondition(lastHidden.count == weights.localDim)
        let numCodes = weights.numCodesPerCodebook
        var logits = weights.outProjBiases[codebook]  // copy bias (numCodes,)
        // logits += outProjWeights[codebook] @ lastHidden  (numCodes, localDim) × (localDim,)
        let w = weights.outProjWeights[codebook]
        w.withUnsafeBufferPointer { wPtr in
            lastHidden.withUnsafeBufferPointer { hPtr in
                logits.withUnsafeMutableBufferPointer { outPtr in
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(numCodes), Int32(weights.localDim),
                        1.0,
                        wPtr.baseAddress, Int32(weights.localDim),
                        hPtr.baseAddress, 1,
                        1.0,
                        outPtr.baseAddress, 1)
                }
            }
        }
        return logits
    }

    // MARK: - Private helpers

    private func addPositional(into buffer: inout [Float], length T: Int) {
        let D = weights.localDim
        let count = T * D
        var tmp = buffer
        weights.posEmbedding.withUnsafeBufferPointer { posPtr in
            tmp.withUnsafeMutableBufferPointer { dstPtr in
                // Only use first T rows of posEmbedding.
                vDSP_vadd(
                    dstPtr.baseAddress!, 1,
                    posPtr.baseAddress!, 1,
                    dstPtr.baseAddress!, 1,
                    vDSP_Length(count))
            }
        }
        buffer = tmp
    }

    private func layerNorm(_ x: [Float], length T: Int, weight: [Float]) -> [Float] {
        let D = weights.localDim
        var out = Swift.Array<Float>(repeating: 0, count: T * D)
        let eps: Float = 1e-5
        for t in 0..<T {
            let row = Swift.Array(x[(t * D)..<(t * D + D)])
            var mean: Float = 0
            vDSP_meanv(row, 1, &mean, vDSP_Length(D))
            // Variance
            var negMean = -mean
            var centered = Swift.Array<Float>(repeating: 0, count: D)
            vDSP_vsadd(row, 1, &negMean, &centered, 1, vDSP_Length(D))
            var variance: Float = 0
            var sqr = Swift.Array<Float>(repeating: 0, count: D)
            vDSP_vsq(centered, 1, &sqr, 1, vDSP_Length(D))
            vDSP_meanv(sqr, 1, &variance, vDSP_Length(D))
            let invStd = 1.0 / sqrt(variance + eps)
            var invStdVar = invStd
            var normed = Swift.Array<Float>(repeating: 0, count: D)
            vDSP_vsmul(centered, 1, &invStdVar, &normed, 1, vDSP_Length(D))
            // Multiply by weight elementwise.
            vDSP_vmul(normed, 1, weight, 1, &normed, 1, vDSP_Length(D))
            for i in 0..<D { out[t * D + i] = normed[i] }
        }
        return out
    }

    /// Compute `inProjWeight @ hidden + bias` in-place (bias already copied into `accumulate`).
    private func inProjWeightApply(hidden: [Float], accumulate: inout [Float]) {
        let D = weights.localDim
        let M = weights.dModel
        weights.inProjWeight.withUnsafeBufferPointer { wPtr in
            hidden.withUnsafeBufferPointer { hPtr in
                accumulate.withUnsafeMutableBufferPointer { outPtr in
                    cblas_sgemv(
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
    }

    /// Row-major `out = A @ B`  (M×K) × (K×N) = (M×N)
    private func matmul(
        a: [Float], aRows M: Int, aCols K: Int,
        b: [Float], bRows: Int, bCols N: Int,
        out: inout [Float]
    ) {
        precondition(K == bRows, "matmul inner dimension mismatch")
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                out.withUnsafeMutableBufferPointer { outPtr in
                    cblas_sgemm(
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

    /// Row-major `out = A @ Bᵀ`  (M×K) × (N×K)ᵀ = (M×N); B is stored as (N, K).
    private func matmulTransB(
        a: [Float], aRows M: Int, aCols K: Int,
        b: [Float], bRows N: Int, bCols bk: Int,
        out: inout [Float]
    ) {
        precondition(K == bk, "matmulTransB inner dimension mismatch")
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                out.withUnsafeMutableBufferPointer { outPtr in
                    cblas_sgemm(
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

    /// Apply tanh-approximation GELU in-place.
    /// `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
    private func applyGeluTanh(into buffer: inout [Float]) {
        let n = buffer.count
        let sqrt2pi: Float = 0.7978845608
        let coef: Float = 0.044715
        for i in 0..<n {
            let x = buffer[i]
            let x3 = x * x * x
            let inner = sqrt2pi * (x + coef * x3)
            let t = tanhf(inner)
            buffer[i] = 0.5 * x * (1 + t)
        }
    }
}
