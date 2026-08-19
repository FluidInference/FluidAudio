import CoreML
import XCTest

@testable import FluidAudio

final class LogitsArgmaxTests: XCTestCase {

    /// Contiguous fp32 logits: argmax per frame matches a scalar reference.
    func testFloat32Contiguous() throws {
        let frames = 3
        let vocab = 5
        let logits = try MLMultiArray(shape: [1, frames as NSNumber, vocab as NSNumber], dataType: .float32)
        let values: [[Float]] = [
            [0.1, 0.9, -0.3, 0.2, 0.0],
            [-2.0, -1.0, -0.5, -3.0, -4.0],
            [7.0, 7.0, 8.0, 8.0, 1.0],  // tie on the max: first index wins
        ]
        for t in 0..<frames {
            for v in 0..<vocab { logits[[0, t as NSNumber, v as NSNumber]] = NSNumber(value: values[t][v]) }
        }
        XCTAssertEqual(LogitsArgmax.argmaxPerFrame(logits: logits, frames: frames), [1, 2, 2])
    }

    /// fp16 logits go through the vImage widening path.
    func testFloat16Conversion() throws {
        let frames = 2
        let vocab = 4
        let logits = try MLMultiArray(shape: [1, frames as NSNumber, vocab as NSNumber], dataType: .float16)
        let values: [[Float]] = [
            [0.25, -0.5, 3.0, 1.5],
            [-1.0, -0.25, -0.75, -0.125],
        ]
        for t in 0..<frames {
            for v in 0..<vocab { logits[[0, t as NSNumber, v as NSNumber]] = NSNumber(value: values[t][v]) }
        }
        XCTAssertEqual(LogitsArgmax.argmaxPerFrame(logits: logits, frames: frames), [2, 3])
    }

    /// Padded row stride (CoreML pads rows for ANE alignment, e.g. 8404 -> 8408):
    /// the padding slot holds a huge value that must never be selected.
    func testPaddedStrideIgnoresPadding() throws {
        let frames = 3
        let vocab = 3
        let stride = 4  // one padding slot per row
        var storage = [Float](repeating: 0, count: frames * stride)
        let rows: [[Float]] = [
            [0.5, 0.1, 0.2],
            [-1.0, -0.2, -0.6],
            [2.0, 9.0, 3.0],
        ]
        for t in 0..<frames {
            for v in 0..<vocab { storage[t * stride + v] = rows[t][v] }
            storage[t * stride + vocab] = 1e9  // poison the padding slot
        }
        let expected = [0, 1, 1]
        try storage.withUnsafeMutableBytes { buf in
            let logits = try MLMultiArray(
                dataPointer: buf.baseAddress!,
                shape: [1, frames as NSNumber, vocab as NSNumber],
                dataType: .float32,
                strides: [NSNumber(value: frames * stride), NSNumber(value: stride), 1])
            XCTAssertEqual(LogitsArgmax.argmaxPerFrame(logits: logits, frames: frames), expected)
        }
    }

    /// `frames` smaller than the tensor's frame dimension only scans that prefix.
    func testFramePrefix() throws {
        let logits = try MLMultiArray(shape: [1, 4, 2], dataType: .float32)
        for t in 0..<4 {
            logits[[0, t as NSNumber, 0]] = NSNumber(value: Float(t))
            logits[[0, t as NSNumber, 1]] = NSNumber(value: -Float(t))
        }
        XCTAssertEqual(LogitsArgmax.argmaxPerFrame(logits: logits, frames: 2).count, 2)
    }

    /// All-negative logits: `vDSP_maxvi` must still pick the true maximum
    /// (regression guard for an accumulator seeded with 0).
    func testAllNegativeLogits() throws {
        let logits = try MLMultiArray(shape: [1, 1, 3], dataType: .float32)
        logits[[0, 0, 0]] = -5.0
        logits[[0, 0, 1]] = -2.0
        logits[[0, 0, 2]] = -9.0
        XCTAssertEqual(LogitsArgmax.argmaxPerFrame(logits: logits, frames: 1), [1])
    }
}
