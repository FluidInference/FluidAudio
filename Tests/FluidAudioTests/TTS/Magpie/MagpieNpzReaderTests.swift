import Compression
import Foundation
import XCTest

@testable import FluidAudio

/// Validates that `MagpieNpzReader` can decode both `np.savez` (STORE) and
/// `np.savez_compressed` (DEFLATE) archives. We synthesize the ZIP bytes in
/// the test so we don't need a Python/NumPy-emitted fixture in CI.
final class MagpieNpzReaderTests: XCTestCase {

    // MARK: - .npy synthesis

    /// Build a minimal NPY (v1.0) payload for a 1-D fp32 array with the given values.
    private func makeNpyFloat32(values: [Float]) -> Data {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(values.count),), }"
        return makeNpy(header: header, dtypeBytes: 4) {
            var data = Data()
            for v in values {
                var bits = v.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
            return data
        }
    }

    private func makeNpyInt32(values: [Int32], shape: [Int]) -> Data {
        let shapeStr =
            shape.count == 1
            ? "(\(shape[0]),)"
            : "(" + shape.map { String($0) }.joined(separator: ", ") + ")"
        let header = "{'descr': '<i4', 'fortran_order': False, 'shape': \(shapeStr), }"
        return makeNpy(header: header, dtypeBytes: 4) {
            var data = Data()
            for v in values {
                var bits = UInt32(bitPattern: v).littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
            return data
        }
    }

    private func makeNpy(
        header headerInner: String, dtypeBytes _: Int, body: () -> Data
    ) -> Data {
        // Pad header so total prefix is 64-byte aligned and ends with '\n'.
        let preamble = 10  // magic(6) + version(2) + headerLen(2)
        var header = headerInner
        while (preamble + header.count + 1) % 64 != 0 {
            header += " "
        }
        header += "\n"
        var data = Data()
        let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00]
        data.append(contentsOf: magic)
        let len = UInt16(header.count).littleEndian
        withUnsafeBytes(of: len) { data.append(contentsOf: $0) }
        data.append(header.data(using: .ascii)!)
        data.append(body())
        return data
    }

    // MARK: - ZIP synthesis (STORE only — used to test the parser)

    private func makeStoreZip(entries: [(name: String, payload: Data)]) -> Data {
        var archive = Data()
        var centralRecords: [Data] = []
        var localOffsets: [Int] = []

        for (name, payload) in entries {
            let nameBytes = name.data(using: .utf8)!
            let crc = crc32(payload)
            let localOffset = archive.count
            localOffsets.append(localOffset)

            // Local file header.
            archive.append(uint32: 0x0403_4b50)  // signature
            archive.append(uint16: 20)  // version needed
            archive.append(uint16: 0)  // flags
            archive.append(uint16: 0)  // method = STORE
            archive.append(uint16: 0)  // mod time
            archive.append(uint16: 0)  // mod date
            archive.append(uint32: crc)
            archive.append(uint32: UInt32(payload.count))  // compressed size
            archive.append(uint32: UInt32(payload.count))  // uncompressed size
            archive.append(uint16: UInt16(nameBytes.count))
            archive.append(uint16: 0)  // extra length
            archive.append(nameBytes)
            archive.append(payload)

            // Central directory record.
            var cd = Data()
            cd.append(uint32: 0x0201_4b50)
            cd.append(uint16: 20)  // version made by
            cd.append(uint16: 20)  // version needed
            cd.append(uint16: 0)  // flags
            cd.append(uint16: 0)  // method
            cd.append(uint16: 0)  // mod time
            cd.append(uint16: 0)  // mod date
            cd.append(uint32: crc)
            cd.append(uint32: UInt32(payload.count))
            cd.append(uint32: UInt32(payload.count))
            cd.append(uint16: UInt16(nameBytes.count))
            cd.append(uint16: 0)
            cd.append(uint16: 0)
            cd.append(uint16: 0)
            cd.append(uint16: 0)
            cd.append(uint32: 0)
            cd.append(uint32: UInt32(localOffset))
            cd.append(nameBytes)
            centralRecords.append(cd)
        }

        let cdOffset = archive.count
        for cd in centralRecords {
            archive.append(cd)
        }
        let cdSize = archive.count - cdOffset

        // EOCD.
        archive.append(uint32: 0x0605_4b50)
        archive.append(uint16: 0)  // disk #
        archive.append(uint16: 0)  // disk with CD
        archive.append(uint16: UInt16(entries.count))  // entries on this disk
        archive.append(uint16: UInt16(entries.count))  // total entries
        archive.append(uint32: UInt32(cdSize))
        archive.append(uint32: UInt32(cdOffset))
        archive.append(uint16: 0)  // comment length
        return archive
    }

    /// Standard CRC-32 (poly 0xEDB88320) — needed for valid ZIP central-dir entries.
    private func crc32(_ data: Data) -> UInt32 {
        var table = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            var c = UInt32(i)
            for _ in 0..<8 {
                c = (c & 1) != 0 ? (0xEDB8_8320 ^ (c >> 1)) : (c >> 1)
            }
            table[i] = c
        }
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in data {
            crc = table[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8)
        }
        return crc ^ 0xFFFF_FFFF
    }

    // MARK: - Tests

    func testReadStoreNpzWithFloat32AndInt32Members() throws {
        let floats: [Float] = [0.5, -1.25, 3.0, 7.125]
        let ints: [Int32] = [1, 2, 3, 4, 5, 6]
        let entries: [(name: String, payload: Data)] = [
            ("encoder_output.npy", makeNpyFloat32(values: floats)),
            ("token_ids.npy", makeNpyInt32(values: ints, shape: [6])),
        ]
        let zip = makeStoreZip(entries: entries)
        let parsed = try MagpieNpzReader.parse(archive: zip, sourceLabel: "synthetic.npz")

        XCTAssertEqual(Set(parsed.keys), Set(["encoder_output", "token_ids"]))

        let arrF = try XCTUnwrap(parsed["encoder_output"])
        XCTAssertEqual(arrF.shape, [4])
        XCTAssertEqual(arrF.data, floats)

        let arrI = try XCTUnwrap(parsed["token_ids"])
        XCTAssertEqual(arrI.shape, [6])
        XCTAssertEqual(arrI.data.map { Int32($0) }, ints)
    }

    func testReadStoreNpzWithMultiDimensionalShape() throws {
        let values: [Int32] = [1, 2, 3, 4, 5, 6]
        let entry = (name: "matrix.npy", payload: makeNpyInt32(values: values, shape: [2, 3]))
        let zip = makeStoreZip(entries: [entry])
        let parsed = try MagpieNpzReader.parse(archive: zip, sourceLabel: "synthetic.npz")
        let arr = try XCTUnwrap(parsed["matrix"])
        XCTAssertEqual(arr.shape, [2, 3])
        XCTAssertEqual(arr.data.map { Int32($0) }, values)
    }

    func testEmptyArchiveYieldsEmptyMap() throws {
        let zip = makeStoreZip(entries: [])
        let parsed = try MagpieNpzReader.parse(archive: zip, sourceLabel: "empty.npz")
        XCTAssertTrue(parsed.isEmpty)
    }

    func testTruncatedArchiveThrowsInvalidNpyFile() {
        let zip = Data([0x50, 0x4b, 0x05, 0x06])  // partial EOCD
        XCTAssertThrowsError(try MagpieNpzReader.parse(archive: zip, sourceLabel: "bad.npz")) {
            err in
            guard case MagpieError.invalidNpyFile = err else {
                XCTFail("Expected invalidNpyFile, got \(err)")
                return
            }
        }
    }
}

// MARK: - Data helpers

extension Data {
    fileprivate mutating func append(uint16 value: UInt16) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }

    fileprivate mutating func append(uint32 value: UInt32) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
}
