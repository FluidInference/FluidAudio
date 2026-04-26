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

        // Split QKV into Q, K, V (each T × D). Direct memcpy from packed (T, 3D)
        // buffer; no intermediate Swift sub-array allocations per row.
        var q = Swift.Array<Float>(repeating: 0, count: T * D)
        var k = Swift.Array<Float>(repeating: 0, count: T * D)
        var v = Swift.Array<Float>(repeating: 0, count: T * D)
        let bytesPerRow = D * MemoryLayout<Float>.size
        qkv.withUnsafeBufferPointer { srcPtr in
            q.withUnsafeMutableBufferPointer { qPtr in
                k.withUnsafeMutableBufferPointer { kPtr in
                    v.withUnsafeMutableBufferPointer { vPtr in
                        guard let src = srcPtr.baseAddress,
                            let qb = qPtr.baseAddress,
                            let kb = kPtr.baseAddress,
                            let vb = vPtr.baseAddress
                        else { return }
                        for t in 0..<T {
                            let srcRow = src.advanced(by: t * 3 * D)
                            let dstOff = t * D
                            memcpy(qb.advanced(by: dstOff), srcRow, bytesPerRow)
                            memcpy(kb.advanced(by: dstOff), srcRow.advanced(by: D), bytesPerRow)
                            memcpy(vb.advanced(by: dstOff), srcRow.advanced(by: 2 * D), bytesPerRow)
                        }
                    }
                }
            }
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
        x.withUnsafeBufferPointer { xPtr in
            weight.withUnsafeBufferPointer { wPtr in
                out.withUnsafeMutableBufferPointer { outPtr in
                    guard let xBase = xPtr.baseAddress,
                        let wBase = wPtr.baseAddress,
                        let outBase = outPtr.baseAddress
                    else { return }
                    for t in 0..<T {
                        let row = xBase.advanced(by: t * D)
                        let outRow = outBase.advanced(by: t * D)
                        // mean = avg(row)
                        var mean: Float = 0
                        vDSP_meanv(row, 1, &mean, vDSP_Length(D))
                        // outRow = row - mean (in-place via vsadd with -mean)
                        var negMean = -mean
                        vDSP_vsadd(row, 1, &negMean, outRow, 1, vDSP_Length(D))
                        // variance = mean(centered^2). Use vDSP_measqv for fused
                        // square + mean over the centered buffer (one pass).
                        var meanSq: Float = 0
                        vDSP_measqv(outRow, 1, &meanSq, vDSP_Length(D))
                        var invStd = 1.0 / sqrt(meanSq + eps)
                        // outRow = centered * invStd
                        vDSP_vsmul(outRow, 1, &invStd, outRow, 1, vDSP_Length(D))
                        // outRow *= weight (elementwise).
                        vDSP_vmul(outRow, 1, wBase, 1, outRow, 1, vDSP_Length(D))
                    }
                }
            }
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
    ///
    /// Vectorized via vDSP for the polynomial inner term and `vvtanhf` for the
    /// elementwise tanh. Avoids the per-element `tanhf` call from the scalar loop.
    private func applyGeluTanh(into buffer: inout [Float]) {
        let n = buffer.count
        guard n > 0 else { return }
        var sqrt2pi: Float = 0.7978845608
        var coef: Float = 0.044715
        var half: Float = 0.5
        var one: Float = 1.0
        var inner = Swift.Array<Float>(repeating: 0, count: n)
        var tanhOut = Swift.Array<Float>(repeating: 0, count: n)
        buffer.withUnsafeMutableBufferPointer { buf in
            inner.withUnsafeMutableBufferPointer { innerBuf in
                tanhOut.withUnsafeMutableBufferPointer { tanhBuf in
                    guard let xPtr = buf.baseAddress,
                        let inPtr = innerBuf.baseAddress,
                        let tPtr = tanhBuf.baseAddress
                    else { return }
                    // inner = x * x  (then x^3 = inner * x)
                    vDSP_vsq(xPtr, 1, inPtr, 1, vDSP_Length(n))
                    vDSP_vmul(inPtr, 1, xPtr, 1, inPtr, 1, vDSP_Length(n))
                    // inner = coef * x^3
                    vDSP_vsmul(inPtr, 1, &coef, inPtr, 1, vDSP_Length(n))
                    // inner = x + coef*x^3
                    vDSP_vadd(inPtr, 1, xPtr, 1, inPtr, 1, vDSP_Length(n))
                    // inner *= sqrt(2/π)
                    vDSP_vsmul(inPtr, 1, &sqrt2pi, inPtr, 1, vDSP_Length(n))
                    // tanhOut = tanh(inner)
                    var nVar = Int32(n)
                    vvtanhf(tPtr, inPtr, &nVar)
                    // tanhOut = 1 + tanh(inner)
                    vDSP_vsadd(tPtr, 1, &one, tPtr, 1, vDSP_Length(n))
                    // tanhOut *= x
                    vDSP_vmul(tPtr, 1, xPtr, 1, tPtr, 1, vDSP_Length(n))
                    // x = 0.5 * tanhOut
                    vDSP_vsmul(tPtr, 1, &half, xPtr, 1, vDSP_Length(n))
                }
            }
        }
    }
}
