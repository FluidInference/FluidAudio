import Accelerate
@preconcurrency import CoreML
import Foundation

/// Native-Accelerate RNN-T predictor + joint forward.
///
/// Replaces `decoder.predict()` + `joint.predict()` (or the B1-fused
/// `decoder_joint.predict()`) for the inner RNN-T greedy loop. Both layers
/// stay in Swift with vDSP / cblas instead of paying the ~150-200us
/// per-token CoreML dispatch overhead.
///
/// Weights are loaded from a binary blob produced by
/// `extract_decoder_joint_weights.py`. Layout:
///   weights.bin  : concatenated float16 little-endian
///   weights_index.json : per-tensor offset + shape
///   parity_reference.json : fixed-input reference outputs for verification
///
/// All compute is upcast to float32 internally — vDSP/cblas paths are much
/// faster on M-series chips for float32 than float16, and the precision
/// matches CoreML's internal fp32 accumulation anyway.
public final class NativeRnntInner {

    // MARK: - Weight buffers (float32, kept resident in memory)
    private let embed: [Float]
    private let lstm0_W_ih: [Float]
    private let lstm0_W_hh: [Float]
    private let lstm0_bias: [Float]  // pre-summed b_ih + b_hh
    private let lstm1_W_ih: [Float]
    private let lstm1_W_hh: [Float]
    private let lstm1_bias: [Float]
    private let joint_enc_W: [Float]
    private let joint_enc_b: [Float]
    private let joint_pred_W: [Float]
    private let joint_pred_b: [Float]
    private let joint_out_W: [Float]
    private let joint_out_b: [Float]

    // MARK: - Dims
    public let vocab: Int
    public let hidden: Int = 640
    public let encoderDim: Int = 1024

    // MARK: - LSTM state (kept across calls; reset() to zero them)
    private var h0: [Float]
    private var c0: [Float]
    private var h1: [Float]
    private var c1: [Float]

    // MARK: - Scratch buffers
    private var gates: [Float]  // [4*hidden]
    private var encProj: [Float]  // [hidden]
    private var predProj: [Float]  // [hidden]
    private var combined: [Float]  // [hidden]
    private var logits: [Float]  // [vocab]
    private var h0_new: [Float]
    private var c0_new: [Float]
    private var h1_new: [Float]
    private var c1_new: [Float]
    // Persistent scratch for LSTM gate activations + tanh(c_new). Pre-allocated
    // once to avoid per-token allocation in the inner loop.
    private var sigScratch: [Float]  // [hidden]
    private var tanhCScratch: [Float]  // [hidden]

    /// Public state snapshots for compatibility with the existing CoreML
    /// MLMultiArray-based state-passing in the manager.
    public var hAsArray: [Float] { Array(h0) + Array(h1) }
    public var cAsArray: [Float] { Array(c0) + Array(c1) }

    // MARK: - Init

    /// Loads `weights.bin` + `weights_index.json` from `directory`.
    /// Returns nil if the directory lacks the expected files.
    public init?(directory: URL) {
        let indexURL = directory.appendingPathComponent("weights_index.json")
        let blobURL = directory.appendingPathComponent("weights.bin")
        guard let indexData = try? Data(contentsOf: indexURL),
            let blobData = try? Data(contentsOf: blobURL),
            let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
            let tensors = index["tensors"] as? [String: [String: Any]],
            let vocabSize = index["vocab_size"] as? Int
        else {
            return nil
        }
        self.vocab = vocabSize

        // Helper to slice a tensor as Float (upcast from stored float16)
        func loadF32(_ name: String) -> [Float] {
            guard let info = tensors[name],
                let offsetBytes = info["offset"] as? Int,
                let shape = info["shape"] as? [Int]
            else {
                fatalError("Missing tensor: \(name)")
            }
            let n = shape.reduce(1, *)
            var fp32 = [Float](repeating: 0, count: n)
            blobData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                let fp16Ptr = raw.baseAddress!.advanced(by: offsetBytes).assumingMemoryBound(to: UInt16.self)
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: fp16Ptr),
                    height: 1,
                    width: UInt(n),
                    rowBytes: n * 2
                )
                fp32.withUnsafeMutableBufferPointer { buf in
                    var dst = vImage_Buffer(
                        data: buf.baseAddress!,
                        height: 1,
                        width: UInt(n),
                        rowBytes: n * 4
                    )
                    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
                }
            }
            return fp32
        }

        self.embed = loadF32("decoder.embed.weight")
        self.lstm0_W_ih = loadF32("decoder.lstm.weight_ih_l0")
        self.lstm0_W_hh = loadF32("decoder.lstm.weight_hh_l0")
        let b0_ih = loadF32("decoder.lstm.bias_ih_l0")
        let b0_hh = loadF32("decoder.lstm.bias_hh_l0")
        var b0_sum = [Float](repeating: 0, count: b0_ih.count)
        vDSP_vadd(b0_ih, 1, b0_hh, 1, &b0_sum, 1, vDSP_Length(b0_ih.count))
        self.lstm0_bias = b0_sum

        self.lstm1_W_ih = loadF32("decoder.lstm.weight_ih_l1")
        self.lstm1_W_hh = loadF32("decoder.lstm.weight_hh_l1")
        let b1_ih = loadF32("decoder.lstm.bias_ih_l1")
        let b1_hh = loadF32("decoder.lstm.bias_hh_l1")
        var b1_sum = [Float](repeating: 0, count: b1_ih.count)
        vDSP_vadd(b1_ih, 1, b1_hh, 1, &b1_sum, 1, vDSP_Length(b1_ih.count))
        self.lstm1_bias = b1_sum

        self.joint_enc_W = loadF32("joint.enc.weight")
        self.joint_enc_b = loadF32("joint.enc.bias")
        self.joint_pred_W = loadF32("joint.pred.weight")
        self.joint_pred_b = loadF32("joint.pred.bias")
        self.joint_out_W = loadF32("joint.out.weight")
        self.joint_out_b = loadF32("joint.out.bias")

        self.h0 = [Float](repeating: 0, count: hidden)
        self.c0 = [Float](repeating: 0, count: hidden)
        self.h1 = [Float](repeating: 0, count: hidden)
        self.c1 = [Float](repeating: 0, count: hidden)
        self.h0_new = [Float](repeating: 0, count: hidden)
        self.c0_new = [Float](repeating: 0, count: hidden)
        self.h1_new = [Float](repeating: 0, count: hidden)
        self.c1_new = [Float](repeating: 0, count: hidden)
        self.gates = [Float](repeating: 0, count: 4 * hidden)
        self.encProj = [Float](repeating: 0, count: hidden)
        self.predProj = [Float](repeating: 0, count: hidden)
        self.combined = [Float](repeating: 0, count: hidden)
        self.logits = [Float](repeating: 0, count: vocab)
        self.sigScratch = [Float](repeating: 0, count: hidden)
        self.tanhCScratch = [Float](repeating: 0, count: hidden)
    }

    // MARK: - State management

    public func resetState() {
        for i in 0..<hidden {
            h0[i] = 0
            c0[i] = 0
            h1[i] = 0
            c1[i] = 0
        }
    }

    /// Restore LSTM state from MLMultiArrays produced by the standard
    /// CoreML decoder path. h, c are each [2, 1, hidden].
    internal func setState(h: MLMultiArray, c: MLMultiArray) {
        let hPtr = h.dataPointer.bindMemory(to: Float.self, capacity: 2 * hidden)
        let cPtr = c.dataPointer.bindMemory(to: Float.self, capacity: 2 * hidden)
        for i in 0..<hidden {
            h0[i] = hPtr[i]
            h1[i] = hPtr[hidden + i]
            c0[i] = cPtr[i]
            c1[i] = cPtr[hidden + i]
        }
    }

    /// Snapshot LSTM state into freshly-allocated MLMultiArrays for the
    /// async-pipeline path. Shape: [2, 1, hidden] each.
    internal func snapshotState() throws -> (h: MLMultiArray, c: MLMultiArray) {
        let hArr = try MLMultiArray(
            shape: [2, 1, NSNumber(value: hidden)], dataType: .float32)
        let cArr = try MLMultiArray(
            shape: [2, 1, NSNumber(value: hidden)], dataType: .float32)
        let hPtr = hArr.dataPointer.bindMemory(to: Float.self, capacity: 2 * hidden)
        let cPtr = cArr.dataPointer.bindMemory(to: Float.self, capacity: 2 * hidden)
        for i in 0..<hidden {
            hPtr[i] = h0[i]
            hPtr[hidden + i] = h1[i]
            cPtr[i] = c0[i]
            cPtr[hidden + i] = c1[i]
        }
        return (hArr, cArr)
    }

    // MARK: - Inner-loop step

    /// Single per-token forward: embed -> LSTM(2 layers) -> joint -> argmax.
    /// **Does NOT commit LSTM state** — caller must call `commitState()`
    /// after non-blank emissions. Blank emissions should NOT commit (RNN-T
    /// greedy semantics: state only advances on non-blank).
    /// encStep must point to `encoderDim` (1024) consecutive Float values
    /// (one encoder frame).
    /// Returns predicted token index.
    public func step(currentToken: Int32, encStep: UnsafePointer<Float>) -> Int {
        // 1. Embedding: x = embed[currentToken]  shape [hidden]
        let tokenIdx = Int(currentToken)
        let embedOffset = tokenIdx * hidden
        var x = [Float](repeating: 0, count: hidden)
        embed.withUnsafeBufferPointer { embPtr in
            memcpy(&x, embPtr.baseAddress!.advanced(by: embedOffset), hidden * MemoryLayout<Float>.stride)
        }

        // 2. LSTM layer 0: (x, h0, c0) -> (h0_new, c0_new)
        lstmCellForward(
            x: x,
            h: h0,
            c: c0,
            W_ih: lstm0_W_ih,
            W_hh: lstm0_W_hh,
            bias: lstm0_bias,
            outH: &h0_new,
            outC: &c0_new
        )
        // 3. LSTM layer 1: (h0_new, h1, c1) -> (h1_new, c1_new)
        lstmCellForward(
            x: h0_new,
            h: h1,
            c: c1,
            W_ih: lstm1_W_ih,
            W_hh: lstm1_W_hh,
            bias: lstm1_bias,
            outH: &h1_new,
            outC: &c1_new
        )

        // dec_out = h1_new
        // 4. Joint enc proj: encProj = joint_enc_W [hidden, 1024] @ encStep + joint_enc_b
        joint_enc_W.withUnsafeBufferPointer { wPtr in
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(hidden), Int32(encoderDim),
                1.0, wPtr.baseAddress, Int32(encoderDim),
                encStep, 1,
                0.0, &encProj, 1
            )
        }
        vDSP_vadd(encProj, 1, joint_enc_b, 1, &encProj, 1, vDSP_Length(hidden))

        // 5. Joint pred proj: predProj = joint_pred_W [hidden, hidden] @ dec_out + joint_pred_b
        joint_pred_W.withUnsafeBufferPointer { wPtr in
            h1_new.withUnsafeBufferPointer { hPtr in
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans,
                    Int32(hidden), Int32(hidden),
                    1.0, wPtr.baseAddress, Int32(hidden),
                    hPtr.baseAddress, 1,
                    0.0, &predProj, 1
                )
            }
        }
        vDSP_vadd(predProj, 1, joint_pred_b, 1, &predProj, 1, vDSP_Length(hidden))

        // 6. Combined = ReLU(encProj + predProj)
        vDSP_vadd(encProj, 1, predProj, 1, &combined, 1, vDSP_Length(hidden))
        var zero: Float = 0
        var maxF: Float = .greatestFiniteMagnitude
        vDSP_vclip(combined, 1, &zero, &maxF, &combined, 1, vDSP_Length(hidden))

        // 7. Joint out: logits = joint_out_W [vocab, hidden] @ combined + joint_out_b
        joint_out_W.withUnsafeBufferPointer { wPtr in
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(vocab), Int32(hidden),
                1.0, wPtr.baseAddress, Int32(hidden),
                combined, 1,
                0.0, &logits, 1
            )
        }
        vDSP_vadd(logits, 1, joint_out_b, 1, &logits, 1, vDSP_Length(vocab))

        // 8. Argmax over logits
        var maxVal: Float = -.greatestFiniteMagnitude
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(logits, 1, &maxVal, &maxIdx, vDSP_Length(vocab))

        // Do NOT commit state — caller will commitState() on non-blank only.

        return Int(maxIdx)
    }

    /// Speculative-blank helper: compute `encoder_proj = joint.enc(encoded)`
    /// for an entire chunk in ONE matmul, instead of redoing it per token.
    ///
    /// This bypasses the multi-output encoder approach (which broke ANE
    /// compilation): the standard encoder stays ANE-resident, and we do
    /// the small 1024→640 projection on CPU once per chunk via cblas_sgemm.
    ///
    /// Per-chunk cost: T_enc × 1024 × 640 ≈ 37M ops (at T=56), ~4ms on M-series
    /// CPU. Negligible vs the per-token decoder loop time.
    ///
    /// Inputs:
    ///   encoded: [encoderDim=1024, T_enc] floats (contiguous, caller-provided)
    ///   T_enc:   number of encoder frames in `encoded`
    ///   outBuf:  output buffer [T_enc, hidden=640] floats
    /// (caller manages memory; outBuf must be at least T_enc * 640 floats)
    public func computeEncoderProjBatch(
        encoded: UnsafePointer<Float>,
        T_enc: Int,
        outBuf: UnsafeMutablePointer<Float>
    ) {
        // joint_enc_W: [hidden=640, encoderDim=1024], row-major
        // Treat encoded as [encoderDim, T_enc] column-major (i.e. row-major [T_enc, encoderDim])
        // Want: outBuf[T_enc, hidden] = encoded[T_enc, encoderDim] @ joint_enc_W^T
        //
        // For BLAS: C[M, N] = alpha * A[M, K] * B[K, N] + beta * C
        //   M = T_enc, N = hidden, K = encoderDim
        //   A = encoded (treated as [T_enc, encoderDim] row-major) — leading dim K
        //   B = joint_enc_W^T (treated as [encoderDim, hidden] row-major) — leading dim N
        // cblas_sgemm with CblasNoTrans for A and CblasTrans for B:
        //   B stored as [hidden, encoderDim] row-major (leading dim = encoderDim)
        joint_enc_W.withUnsafeBufferPointer { wPtr in
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                Int32(T_enc), Int32(hidden), Int32(encoderDim),
                1.0,
                encoded, Int32(encoderDim),
                wPtr.baseAddress, Int32(encoderDim),
                0.0,
                outBuf, Int32(hidden)
            )
        }
        // Add joint_enc_b broadcast over T_enc rows
        joint_enc_b.withUnsafeBufferPointer { bPtr in
            for t in 0..<T_enc {
                vDSP_vadd(
                    outBuf.advanced(by: t * hidden), 1, bPtr.baseAddress!, 1, outBuf.advanced(by: t * hidden), 1,
                    vDSP_Length(hidden))
            }
        }
    }

    /// Hybrid path: run only the LSTM forward (embed + 2-layer LSTM),
    /// produce `dec_out` (= h1_new, [hidden=640]) WITHOUT running the
    /// native joint. Caller invokes CoreML joint with this dec_out.
    ///
    /// Same state semantics as `step`: state is NOT committed; caller
    /// must `commitState()` on non-blank emission only.
    ///
    /// Writes h1_new (the new layer-1 hidden, == dec_out for RNN-T) into
    /// the caller-provided `decOut` MLMultiArray (shape [1, hidden, 1]
    /// float32, matching CoreML joint's `decoder` input). Caller passes
    /// the same buffer every iter; LSTM advances on-blank-skipped.
    public func stepLSTMOnly(currentToken: Int32, decOut: MLMultiArray) {
        let tokenIdx = Int(currentToken)
        let embedOffset = tokenIdx * hidden
        var x = [Float](repeating: 0, count: hidden)
        embed.withUnsafeBufferPointer { embPtr in
            memcpy(&x, embPtr.baseAddress!.advanced(by: embedOffset), hidden * MemoryLayout<Float>.stride)
        }
        lstmCellForward(
            x: x, h: h0, c: c0,
            W_ih: lstm0_W_ih, W_hh: lstm0_W_hh, bias: lstm0_bias,
            outH: &h0_new, outC: &c0_new
        )
        lstmCellForward(
            x: h0_new, h: h1, c: c1,
            W_ih: lstm1_W_ih, W_hh: lstm1_W_hh, bias: lstm1_bias,
            outH: &h1_new, outC: &c1_new
        )
        let dstPtr = decOut.dataPointer.bindMemory(to: Float.self, capacity: hidden)
        h1_new.withUnsafeBufferPointer { srcPtr in
            memcpy(dstPtr, srcPtr.baseAddress!, hidden * MemoryLayout<Float>.stride)
        }
    }

    /// Commit the most-recently computed LSTM state. Caller should ONLY
    /// invoke this after a non-blank emission, mirroring CoreML RNN-T
    /// greedy semantics. State stays unchanged on blank.
    public func commitState() {
        swap(&h0, &h0_new)
        swap(&c0, &c0_new)
        swap(&h1, &h1_new)
        swap(&c1, &c1_new)
    }

    // MARK: - LSTM cell (single time step)

    /// PyTorch nn.LSTM gate order: (i, f, g, o) along the [4*hidden] axis.
    ///   gates = W_ih @ x + W_hh @ h + bias  (bias is pre-summed b_ih + b_hh)
    ///   i = sigmoid(gates[0:H])
    ///   f = sigmoid(gates[H:2H])
    ///   g = tanh(gates[2H:3H])
    ///   o = sigmoid(gates[3H:4H])
    ///   c_new = f * c + i * g
    ///   h_new = o * tanh(c_new)
    private func lstmCellForward(
        x: [Float],
        h: [Float],
        c: [Float],
        W_ih: [Float],
        W_hh: [Float],
        bias: [Float],
        outH: inout [Float],
        outC: inout [Float]
    ) {
        let H = hidden
        // gates = W_ih @ x  (W_ih is [4H, H], row-major)
        W_ih.withUnsafeBufferPointer { wPtr in
            x.withUnsafeBufferPointer { xPtr in
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans,
                    Int32(4 * H), Int32(H),
                    1.0, wPtr.baseAddress, Int32(H),
                    xPtr.baseAddress, 1,
                    0.0, &gates, 1
                )
            }
        }
        // gates += W_hh @ h
        W_hh.withUnsafeBufferPointer { wPtr in
            h.withUnsafeBufferPointer { hPtr in
                cblas_sgemv(
                    CblasRowMajor, CblasNoTrans,
                    Int32(4 * H), Int32(H),
                    1.0, wPtr.baseAddress, Int32(H),
                    hPtr.baseAddress, 1,
                    1.0, &gates, 1  // beta=1: accumulate
                )
            }
        }
        // gates += bias
        vDSP_vadd(gates, 1, bias, 1, &gates, 1, vDSP_Length(4 * H))

        // Apply gate nonlinearities IN-PLACE per gate slice
        // sigmoid via 1/(1+exp(-x))
        // For sigmoid in vDSP: negate, exp, add 1, reciprocal
        applySigmoid(buffer: &gates, offset: 0, length: H)  // i
        applySigmoid(buffer: &gates, offset: H, length: H)  // f
        applyTanh(buffer: &gates, offset: 2 * H, length: H)  // g
        applySigmoid(buffer: &gates, offset: 3 * H, length: H)  // o

        // c_new = f * c + i * g
        // Compute tmp = f * c
        gates.withUnsafeBufferPointer { gPtr in
            c.withUnsafeBufferPointer { cPtr in
                vDSP_vmul(
                    gPtr.baseAddress!.advanced(by: H), 1,  // f
                    cPtr.baseAddress!, 1,
                    &outC, 1,
                    vDSP_Length(H)
                )
            }
        }
        // outC += i * g
        gates.withUnsafeBufferPointer { gPtr in
            vDSP_vma(
                gPtr.baseAddress!, 1,  // i
                gPtr.baseAddress!.advanced(by: 2 * H), 1,  // g
                outC, 1,
                &outC, 1,
                vDSP_Length(H)
            )
        }

        // h_new = o * tanh(c_new) — use pre-allocated tanhCScratch
        var lenInt32 = Int32(H)
        tanhCScratch.withUnsafeMutableBufferPointer { tanhCPtr in
            outC.withUnsafeBufferPointer { outCPtr in
                vvtanhf(tanhCPtr.baseAddress!, outCPtr.baseAddress!, &lenInt32)
            }
        }
        // outH = o * tanhC
        gates.withUnsafeBufferPointer { gPtr in
            vDSP_vmul(
                gPtr.baseAddress!.advanced(by: 3 * H), 1,  // o
                tanhCScratch, 1,
                &outH, 1,
                vDSP_Length(H)
            )
        }
    }

    @inline(__always)
    private func applySigmoid(buffer: inout [Float], offset: Int, length: Int) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Uses pre-allocated sigScratch (length = hidden) to avoid per-call
        // allocations in the inner loop.
        buffer.withUnsafeBufferPointer { bPtr in
            var negOne: Float = -1
            sigScratch.withUnsafeMutableBufferPointer { scratchPtr in
                vDSP_vsmul(
                    bPtr.baseAddress!.advanced(by: offset), 1, &negOne, scratchPtr.baseAddress!, 1, vDSP_Length(length))
            }
        }
        var lenInt32 = Int32(length)
        sigScratch.withUnsafeMutableBufferPointer { scratchPtr in
            vvexpf(scratchPtr.baseAddress!, scratchPtr.baseAddress!, &lenInt32)
        }
        var one: Float = 1
        sigScratch.withUnsafeMutableBufferPointer { scratchPtr in
            vDSP_vsadd(scratchPtr.baseAddress!, 1, &one, scratchPtr.baseAddress!, 1, vDSP_Length(length))
        }
        sigScratch.withUnsafeBufferPointer { scratchPtr in
            buffer.withUnsafeMutableBufferPointer { bPtr in
                vvrecf(bPtr.baseAddress!.advanced(by: offset), scratchPtr.baseAddress!, &lenInt32)
            }
        }
    }

    @inline(__always)
    private func applyTanh(buffer: inout [Float], offset: Int, length: Int) {
        var lenInt32 = Int32(length)
        buffer.withUnsafeMutableBufferPointer { bPtr in
            vvtanhf(bPtr.baseAddress!.advanced(by: offset), bPtr.baseAddress!.advanced(by: offset), &lenInt32)
        }
    }
}
