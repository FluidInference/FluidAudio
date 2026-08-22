import XCTest

@testable import FluidAudio

/// Uses hard-coded IEEE-754 binary16 bit patterns (not `Float16`) so the
/// test itself compiles on macOS x86_64.
final class Float16ConversionTests: XCTestCase {

    private let cases: [(bits: UInt16, value: Float)] = [
        (0x0000, 0.0),
        (0x3800, 0.5),
        (0x3C00, 1.0),
        (0xC000, -2.0),
        (0x4248, 3.140625),
        (0x7BFF, 65504.0),  // largest finite half
        (0x0001, Float(5.960464477539063e-08)),  // smallest subnormal
    ]

    func testToFloat32KnownBitPatterns() {
        let src = cases.map { $0.bits }
        var dst = [Float](repeating: .nan, count: src.count)
        src.withUnsafeBufferPointer { s in
            dst.withUnsafeMutableBufferPointer { d in
                Float16Conversion.toFloat32(src: s.baseAddress!, dst: d.baseAddress!, count: s.count)
            }
        }
        for (i, expected) in cases.map({ $0.value }).enumerated() {
            XCTAssertEqual(dst[i], expected, "index \(i)")
        }
    }

    func testFromFloat32KnownBitPatterns() {
        let src = cases.map { $0.value }
        var dst = [UInt16](repeating: 0xFFFF, count: src.count)
        src.withUnsafeBufferPointer { s in
            dst.withUnsafeMutableBufferPointer { d in
                Float16Conversion.fromFloat32(src: s.baseAddress!, dst: d.baseAddress!, count: s.count)
            }
        }
        for (i, expected) in cases.map({ $0.bits }).enumerated() {
            XCTAssertEqual(dst[i], expected, "index \(i)")
        }
    }

    func testRoundTripIsExactForHalfRepresentableValues() {
        let values: [Float] = [0.0, -0.125, 0.25, 1.5, -3.0, 1024.0, -65504.0]
        var bits = [UInt16](repeating: 0, count: values.count)
        var back = [Float](repeating: .nan, count: values.count)
        values.withUnsafeBufferPointer { s in
            bits.withUnsafeMutableBufferPointer { d in
                Float16Conversion.fromFloat32(src: s.baseAddress!, dst: d.baseAddress!, count: s.count)
            }
        }
        bits.withUnsafeBufferPointer { s in
            back.withUnsafeMutableBufferPointer { d in
                Float16Conversion.toFloat32(src: s.baseAddress!, dst: d.baseAddress!, count: s.count)
            }
        }
        XCTAssertEqual(back, values)
    }
}
