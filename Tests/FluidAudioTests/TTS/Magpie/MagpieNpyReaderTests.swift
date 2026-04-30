import XCTest

@testable import FluidAudio

final class NpyReaderTests: XCTestCase {

    func testParseTinyFloat32() throws {
        let data = makeNpyV1(
            header: "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
            body: floatBytes([1, 2, 3, 4, 5, 6]))

        let arr = try NpyReader.parse(data: data, sourceLabel: "tiny.npy")
        XCTAssertEqual(arr.shape, [2, 3])
        XCTAssertEqual(arr.data, [1, 2, 3, 4, 5, 6])
    }

    func testParseFloat16UpcastsToFloat32() throws {
        // 1.0 in IEEE 754 half is 0x3C00; 2.0 is 0x4000.
        var body = Data()
        body.append(contentsOf: [0x00, 0x3C, 0x00, 0x40])  // 1.0, 2.0 little-endian
        let data = makeNpyV1(
            header: "{'descr': '<f2', 'fortran_order': False, 'shape': (2,), }",
            body: body)

        let arr = try NpyReader.parse(data: data, sourceLabel: "tiny_fp16.npy")
        XCTAssertEqual(arr.shape, [2])
        XCTAssertEqual(arr.data, [1.0, 2.0])
    }

    func testBadMagicThrows() {
        var bogus = Data(repeating: 0, count: 32)
        bogus[0] = 0x00  // break magic
        XCTAssertThrowsError(try NpyReader.parse(data: bogus, sourceLabel: "bad.npy"))
    }

    // MARK: - helpers

    /// Build an NPY v1 blob with the given header dict literal and raw body bytes.
    private func makeNpyV1(header: String, body: Data) -> Data {
        var data = Data()
        // Magic + version 1.0
        data.append(contentsOf: [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])

        // Pad header so that `10 + len(header)` is a multiple of 64 (NumPy convention),
        // terminate with newline.
        var headerBytes = Array(header.utf8)
        let prelude = 10
        let padTo = ((prelude + headerBytes.count + 1 + 63) / 64) * 64
        let padLen = padTo - (prelude + headerBytes.count + 1)
        headerBytes.append(contentsOf: Array(repeating: UInt8(0x20), count: padLen))
        headerBytes.append(0x0A)

        let headerLen = UInt16(headerBytes.count)
        data.append(UInt8(headerLen & 0xFF))
        data.append(UInt8((headerLen >> 8) & 0xFF))
        data.append(contentsOf: headerBytes)
        data.append(body)
        return data
    }

    private func floatBytes(_ values: [Float]) -> Data {
        var data = Data()
        for v in values {
            var local = v
            withUnsafeBytes(of: &local) { data.append(contentsOf: $0) }
        }
        return data
    }
}
