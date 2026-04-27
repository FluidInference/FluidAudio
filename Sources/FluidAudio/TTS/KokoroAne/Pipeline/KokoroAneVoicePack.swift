import Foundation

/// `[510, 256]` flat fp32 voice pack (e.g. `af_heart.bin`).
///
/// Indexed by phoneme-length bucket: `row = min(max(T_enc - 1, 0), 509)`.
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
