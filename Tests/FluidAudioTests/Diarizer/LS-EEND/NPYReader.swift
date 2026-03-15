import Foundation

@testable import FluidAudio

struct NPYFloatArray {
    let shape: [Int]
    let values: [Float]

    func matrix2D() throws -> LSEENDMatrix {
        guard shape.count == 2 else {
            throw NSError(
                domain: "NPYReader",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Expected a 2D array, received shape \(shape)."]
            )
        }
        return try LSEENDMatrix(rows: shape[0], columns: shape[1], values: values)
    }
}

enum NPYReader {
    static func loadFloatArray(from url: URL) throws -> NPYFloatArray {
        let data = try Data(contentsOf: url)
        guard data.count >= 10 else {
            throw NSError(
                domain: "NPYReader",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid npy file: \(url.path)"]
            )
        }

        let magic = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])
        guard data.prefix(6) == magic else {
            throw NSError(
                domain: "NPYReader",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported npy magic in \(url.path)"]
            )
        }

        let major = Int(data[6])
        let headerLength: Int
        let headerOffset: Int
        switch major {
        case 1:
            headerOffset = 10
            headerLength = Int(UInt16(data[8]) | (UInt16(data[9]) << 8))
        case 2, 3:
            headerOffset = 12
            headerLength =
                Int(UInt32(data[8]))
                | (Int(UInt32(data[9])) << 8)
                | (Int(UInt32(data[10])) << 16)
                | (Int(UInt32(data[11])) << 24)
        default:
            throw NSError(
                domain: "NPYReader",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported npy version \(major) in \(url.path)"]
            )
        }

        let payloadOffset = headerOffset + headerLength
        guard payloadOffset <= data.count else {
            throw NSError(
                domain: "NPYReader",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Corrupt npy header in \(url.path)"]
            )
        }

        let headerData = data.subdata(in: headerOffset..<payloadOffset)
        guard let header = String(data: headerData, encoding: .ascii) else {
            throw NSError(
                domain: "NPYReader",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Header is not ASCII in \(url.path)"]
            )
        }

        let descr = try capture(in: header, pattern: #"'descr':\s*'([^']+)'"#)
        guard descr == "<f4" || descr == "|f4" else {
            throw NSError(
                domain: "NPYReader",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "Only float32 npy files are supported, found \(descr)."]
            )
        }

        let fortranOrder = try capture(in: header, pattern: #"'fortran_order':\s*(False|True)"#)
        guard fortranOrder == "False" else {
            throw NSError(
                domain: "NPYReader",
                code: 8,
                userInfo: [NSLocalizedDescriptionKey: "Fortran-order npy files are not supported."]
            )
        }

        let shapeText = try capture(in: header, pattern: #"'shape':\s*\(([^)]*)\)"#)
        let shape =
            shapeText
            .split(separator: ",")
            .compactMap { component -> Int? in
                let trimmed = component.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? nil : Int(trimmed)
            }
        let elementCount = shape.reduce(1, *)
        let payload = data.suffix(from: payloadOffset)
        let expectedBytes = elementCount * MemoryLayout<Float>.size
        guard payload.count == expectedBytes else {
            throw NSError(
                domain: "NPYReader",
                code: 9,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Expected \(expectedBytes) payload bytes for shape \(shape), found \(payload.count)."
                ]
            )
        }

        var values = [Float](repeating: 0, count: elementCount)
        _ = values.withUnsafeMutableBytes { destination in
            payload.copyBytes(to: destination)
        }
        return NPYFloatArray(shape: shape, values: values)
    }

    private static func capture(in text: String, pattern: String) throws -> String {
        let expression = try NSRegularExpression(pattern: pattern)
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard
            let match = expression.firstMatch(in: text, range: range),
            match.numberOfRanges >= 2,
            let captureRange = Range(match.range(at: 1), in: text)
        else {
            throw NSError(
                domain: "NPYReader",
                code: 10,
                userInfo: [NSLocalizedDescriptionKey: "Could not parse npy header: \(text)"]
            )
        }
        return String(text[captureRange])
    }
}
