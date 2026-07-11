import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Pure-logic unit tests for `PocketTtsVoiceCloner`'s pad/truncate and
/// frame-trim helpers. The full `cloneVoice(from:using:)` entry point
/// needs an `MLModel`, so these tests drive the smaller internal
/// helpers (`makeEncoderInputBuffer`, `usableFrameCount`) which the
/// production path delegates to.
final class PocketTtsVoiceClonerTests: XCTestCase {

    // MARK: - makeEncoderInputBuffer

    func testEncoderInputBufferPadsShorterAudio() {
        // 7.5 s of audio @ 24 kHz = 180_000 samples; encoder wants 240_000.
        let realCount = 180_000
        let input = (0..<realCount).map { Float($0 % 17) - 8 }
        let buffer = PocketTtsVoiceCloner.makeEncoderInputBuffer(input)

        XCTAssertEqual(
            buffer.count, PocketTtsVoiceCloner.encoderInputSamples,
            "Buffer must always be encoderInputSamples long")
        XCTAssertEqual(
            Array(buffer.prefix(realCount)), input,
            "Real samples must be copied verbatim into the prefix")
        XCTAssertTrue(
            buffer.dropFirst(realCount).allSatisfy { $0 == 0 },
            "Padding region must be zero-filled")
    }

    func testEncoderInputBufferTruncatesLongerAudio() {
        // 15 s @ 24 kHz = 360_000 samples; must be truncated to 240_000.
        let oversize = PocketTtsVoiceCloner.encoderInputSamples + 120_000
        let input = (0..<oversize).map { Float($0 % 23) - 11 }
        let buffer = PocketTtsVoiceCloner.makeEncoderInputBuffer(input)

        XCTAssertEqual(
            buffer.count, PocketTtsVoiceCloner.encoderInputSamples,
            "Buffer must always be encoderInputSamples long, never longer")
        XCTAssertEqual(
            buffer, Array(input.prefix(PocketTtsVoiceCloner.encoderInputSamples)),
            "Truncation must keep the leading samples")
    }

    func testEncoderInputBufferHandlesExactLength() {
        // Exactly 240_000 samples → no padding, no truncation.
        let input = (0..<PocketTtsVoiceCloner.encoderInputSamples).map { Float($0) * 1e-6 }
        let buffer = PocketTtsVoiceCloner.makeEncoderInputBuffer(input)

        XCTAssertEqual(buffer, input)
    }

    func testEncoderInputBufferHandlesEmptyInput() {
        // Defensive: empty input shouldn't crash, just produce all zeros.
        let buffer = PocketTtsVoiceCloner.makeEncoderInputBuffer([])

        XCTAssertEqual(buffer.count, PocketTtsVoiceCloner.encoderInputSamples)
        XCTAssertTrue(buffer.allSatisfy { $0 == 0 })
    }

    // MARK: - usableFrameCount

    func testUsableFrameCountRoundsPartialFrameUp() {
        // 7.5 s @ 24 kHz = 180_000 samples. 180_000 / 1920 = 93.75 → 94 frames
        // (ceiling). Encoder always emits 125 frames for the full 10 s window,
        // so we use the ceiling rather than the full output.
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: 180_000, availableFrames: 125)
        XCTAssertEqual(usable, 94)
    }

    func testUsableFrameCountCapsAtMaxVoiceFrames() {
        // Even with 10 s of real audio and a hypothetical bigger encoder
        // output, we never exceed `maxVoiceFrames` (KV cache budget).
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: PocketTtsVoiceCloner.encoderInputSamples,
            availableFrames: 200)
        XCTAssertEqual(usable, PocketTtsVoiceCloner.maxVoiceFrames)
    }

    func testUsableFrameCountCapsAtAvailableFrames() {
        // If the encoder somehow emits fewer frames than the real audio
        // implies, trust the encoder rather than over-reading its buffer.
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: PocketTtsVoiceCloner.encoderInputSamples,
            availableFrames: 80)
        XCTAssertEqual(usable, 80)
    }

    func testUsableFrameCountHandlesExactFrameBoundary() {
        // 95 * 1920 = 182_400 samples — clean multiple, no rounding needed.
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: 95 * PocketTtsVoiceCloner.frameSize,
            availableFrames: 125)
        XCTAssertEqual(usable, 95)
    }

    func testUsableFrameCountHandlesSubFrameAudio() {
        // < 1 frame of audio rounds up to 1 (the encoder still produces a
        // frame even for a tiny prefix). Below-minDurationSeconds inputs
        // are rejected upstream so this is mostly defensive.
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: 100, availableFrames: 125)
        XCTAssertEqual(usable, 1)
    }

    // MARK: - extractConditioning stride handling (FluidAudio #612)

    /// Build an MLMultiArray of shape `[1, frames, embDim]` whose logical
    /// element `[0, f, d]` holds `value(f, d)`. `framePad` extra elements are
    /// inserted (and poisoned with -99) between frames, so the array is
    /// non-contiguous: a naive contiguous read would pick up the padding
    /// instead of the next frame. `framePad == 0` yields a contiguous array.
    /// Returns the array plus the expected packed row-major extraction.
    private func makeConditioning(
        frames: Int, embDim: Int, framePad: Int,
        dataType: MLMultiArrayDataType,
        value: (Int, Int) -> Float
    ) -> (MLMultiArray, [Float]) {
        let frameStride = embDim + framePad
        let total = frames * frameStride
        let shape: [NSNumber] = [1, NSNumber(value: frames), NSNumber(value: embDim)]
        let strides: [NSNumber] = [
            NSNumber(value: total), NSNumber(value: frameStride), NSNumber(value: 1),
        ]
        var expected = [Float]()
        expected.reserveCapacity(frames * embDim)

        if dataType == .float16 {
            #if arch(arm64)
            let buf = UnsafeMutablePointer<Float16>.allocate(capacity: total)
            buf.initialize(repeating: Float16(-99), count: total)
            for f in 0..<frames {
                for d in 0..<embDim {
                    let v = value(f, d)
                    buf[f * frameStride + d] = Float16(v)
                    expected.append(Float(Float16(v)))  // round-trip through fp16
                }
            }
            let array = try! MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(buf),
                shape: shape, dataType: .float16, strides: strides,
                deallocator: { _ in buf.deallocate() })
            return (array, expected)
            #else
            fatalError("fp16 conditioning test requires arm64")
            #endif
        }

        let buf = UnsafeMutablePointer<Float>.allocate(capacity: total)
        buf.initialize(repeating: -99, count: total)
        for f in 0..<frames {
            for d in 0..<embDim {
                let v = value(f, d)
                buf[f * frameStride + d] = v
                expected.append(v)
            }
        }
        let array = try! MLMultiArray(
            dataPointer: UnsafeMutableRawPointer(buf),
            shape: shape, dataType: .float32, strides: strides,
            deallocator: { _ in buf.deallocate() })
        return (array, expected)
    }

    func testExtractConditioningContiguousFloat32() {
        let frames = 4
        let embDim = 8
        let (arr, expected) = makeConditioning(
            frames: frames, embDim: embDim, framePad: 0, dataType: .float32
        ) { f, d in Float(f * 100 + d) }
        let out = PocketTtsVoiceCloner.extractConditioning(arr, frames: frames, embDim: embDim)
        XCTAssertEqual(out, expected)
    }

    func testExtractConditioningStridedFloat32() {
        // framePad > 0: a naive contiguous read would interleave the -99
        // padding into the output. Stride-aware extraction must not.
        let frames = 5
        let embDim = 8
        let (arr, expected) = makeConditioning(
            frames: frames, embDim: embDim, framePad: 3, dataType: .float32
        ) { f, d in Float(f * 1000 + d) }
        let out = PocketTtsVoiceCloner.extractConditioning(arr, frames: frames, embDim: embDim)
        XCTAssertEqual(out, expected)
        XCTAssertFalse(out.contains(-99), "padding must not leak into the extraction")
    }

    func testExtractConditioningReadsLeadingFramesOnly() {
        // Encoder emits more frames than usable; extraction must read only the
        // leading `frames` rows, stride-correct.
        let embDim = 8
        let (arr, _) = makeConditioning(
            frames: 10, embDim: embDim, framePad: 2, dataType: .float32
        ) { f, d in Float(f * 1000 + d) }
        let out = PocketTtsVoiceCloner.extractConditioning(arr, frames: 3, embDim: embDim)
        XCTAssertEqual(out.count, 3 * embDim)
        for f in 0..<3 {
            for d in 0..<embDim {
                XCTAssertEqual(out[f * embDim + d], Float(f * 1000 + d))
            }
        }
    }

    #if arch(arm64)
    func testExtractConditioningContiguousFloat16() {
        let frames = 3
        let embDim = 8
        let (arr, expected) = makeConditioning(
            frames: frames, embDim: embDim, framePad: 0, dataType: .float16
        ) { f, d in Float(f) + Float(d) * 0.25 }  // fp16-exact
        let out = PocketTtsVoiceCloner.extractConditioning(arr, frames: frames, embDim: embDim)
        XCTAssertEqual(out, expected)
    }

    func testExtractConditioningStridedFloat16() {
        let frames = 4
        let embDim = 8
        let (arr, expected) = makeConditioning(
            frames: frames, embDim: embDim, framePad: 4, dataType: .float16
        ) { f, d in Float(f) + Float(d) * 0.5 }  // fp16-exact
        let out = PocketTtsVoiceCloner.extractConditioning(arr, frames: frames, embDim: embDim)
        XCTAssertEqual(out, expected)
        XCTAssertFalse(out.contains(-99), "padding must not leak into the extraction")
    }
    #endif

    // MARK: - reproject (#793 speaker projection)

    /// `reproject` must equal `conditioning · pinv · targetProjᵀ`. Validate the
    /// cblas matmul indexing against a naive triple-loop reference.
    func testReprojectMatchesNaiveReference() {
        let d = PocketTtsConstants.embeddingDim  // 1024
        let k = PocketTtsConstants.latentDim  // 32
        let frames = 3

        // Deterministic pseudo-random inputs.
        func seq(_ n: Int, _ salt: Int) -> [Float] {
            (0..<n).map { i in Float(((i * 1_103_515_245 + salt * 12345) % 2001) - 1000) / 1000.0 }
        }
        let cond = seq(frames * d, 1)
        let pinv = seq(d * k, 2)
        let targetProj = seq(d * k, 3)

        let out = PocketTtsVoiceCloner.reproject(
            conditioning: cond, frames: frames,
            projection: .init(encoderRecoverPinv: pinv, targetProj: targetProj))

        // Naive reference: latents[f,x] = Σ_j cond[f,j]·pinv[j,x];
        //                  ref[f,e]     = Σ_x latents[f,x]·targetProj[e,x].
        var ref = [Float](repeating: 0, count: frames * d)
        for f in 0..<frames {
            var latents = [Float](repeating: 0, count: k)
            for x in 0..<k {
                var acc: Float = 0
                for j in 0..<d { acc += cond[f * d + j] * pinv[j * k + x] }
                latents[x] = acc
            }
            for e in 0..<d {
                var acc: Float = 0
                for x in 0..<k { acc += latents[x] * targetProj[e * k + x] }
                ref[f * d + e] = acc
            }
        }

        XCTAssertEqual(out.count, frames * d)
        for i in 0..<out.count {
            XCTAssertEqual(out[i], ref[i], accuracy: 1e-3, "mismatch at \(i)")
        }
    }

    /// A projection whose recover-pinv and target are set so the second matmul
    /// inverts the first (`Σ_x pinv[j,x]·targetProj[e,x] = δ_{j,e}` on the used
    /// support) leaves the conditioning unchanged — the English (identity) case.
    func testReprojectIdentityWhenTargetInvertsRecover() {
        let d = PocketTtsConstants.embeddingDim
        let k = PocketTtsConstants.latentDim
        let frames = 2

        // pinv selects dims [0..k) into latents; targetProj scatters them back to
        // dims [0..k). Then reproject zeroes dims [k..d) and preserves [0..k).
        var pinv = [Float](repeating: 0, count: d * k)
        var targetProj = [Float](repeating: 0, count: d * k)
        for x in 0..<k {
            pinv[x * k + x] = 1  // pinv[j=x, x] = 1
            targetProj[x * k + x] = 1  // targetProj[e=x, x] = 1
        }
        let cond = (0..<frames * d).map { Float($0 % 13) - 6 }

        let out = PocketTtsVoiceCloner.reproject(
            conditioning: cond, frames: frames,
            projection: .init(encoderRecoverPinv: pinv, targetProj: targetProj))

        for f in 0..<frames {
            for e in 0..<d {
                let expected: Float = e < k ? cond[f * d + e] : 0
                XCTAssertEqual(out[f * d + e], expected, accuracy: 1e-4)
            }
        }
    }
}
