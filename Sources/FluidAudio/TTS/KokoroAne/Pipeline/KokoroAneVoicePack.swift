import Foundation

/// `[510, 256]` flat fp32 voice pack (e.g. `af_heart.bin`).
///
/// Indexed by phoneme-length bucket: `row = min(max(phonemeCount - 1, 0), 509)`
/// where `phonemeCount` is the raw phoneme-string length (BOS/EOS excluded).
/// Columns split into:
///   * `[0..<128]`   = `style_timbre` (fed into Noise + Vocoder)
///   * `[128..<256]` = `style_s`      (fed into PostAlbert + Prosody)
public struct KokoroAneVoicePack: Sendable {

    /// Row-major fp32 storage of length 510 * 256.
    public let storage: [Float]

    public init(storage: [Float]) throws {
        let expected = KokoroAneConstants.voicePackRows * KokoroAneConstants.voicePackCols
        guard storage.count == expected else {
            throw KokoroAneError.invalidVoicePack(
                "expected \(expected) fp32 elements, got \(storage.count)")
        }
        self.storage = storage
    }

    /// Load a flat fp32 binary file (`<voice>.bin`).
    public static func load(from url: URL) throws -> KokoroAneVoicePack {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw KokoroAneError.voicePackMissing(url)
        }
        let data = try Data(contentsOf: url)
        let elemSize = MemoryLayout<Float>.size
        guard data.count % elemSize == 0 else {
            throw KokoroAneError.invalidVoicePack(
                "file size \(data.count) is not a multiple of sizeof(Float)=\(elemSize)")
        }
        let count = data.count / elemSize
        var storage = [Float](repeating: 0, count: count)
        _ = storage.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst)
        }
        return try KokoroAneVoicePack(storage: storage)
    }

    /// Build a pack from a Kokoro-82M v1.0 JSON voice file (the
    /// `voices/<name>.json` layout hosted at the repo root): an object whose
    /// keys `"1"` … `"510"` hold the 256-float row for that phoneme count.
    /// Row `k` of the flat pack is key `"k+1"` — verified byte-exact against
    /// the shipped `af_heart.bin` (#896). Any extra keys (`"embedding"`) are
    /// ignored.
    public static func load(fromJSON data: Data) throws -> KokoroAneVoicePack {
        let rows = KokoroAneConstants.voicePackRows
        let cols = KokoroAneConstants.voicePackCols
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw KokoroAneError.invalidVoicePack("JSON voice pack is not an object")
        }
        var storage: [Float] = []
        storage.reserveCapacity(rows * cols)
        for row in 1...rows {
            guard let values = object[String(row)] as? [NSNumber], values.count == cols else {
                throw KokoroAneError.invalidVoicePack(
                    "JSON voice pack row \(row) missing or not \(cols) numbers")
            }
            for value in values {
                storage.append(value.floatValue)
            }
        }
        return try KokoroAneVoicePack(storage: storage)
    }

    /// Flat little-endian fp32 bytes in the `<voice>.bin` layout.
    public var binaryData: Data {
        storage.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    /// Pick the row that matches the phoneme-length bucket.
    /// Returns `(styleS, styleTimbre)` each of length 128.
    public func slice(for phonemeCount: Int) -> (styleS: [Float], styleTimbre: [Float]) {
        let cols = KokoroAneConstants.voicePackCols
        let row = max(min(phonemeCount - 1, KokoroAneConstants.voicePackRows - 1), 0)
        let base = row * cols
        let timbreRange = base..<(base + 128)
        let styleSRange = (base + 128)..<(base + cols)
        return (
            styleS: Array(storage[styleSRange]),
            styleTimbre: Array(storage[timbreRange])
        )
    }
}
