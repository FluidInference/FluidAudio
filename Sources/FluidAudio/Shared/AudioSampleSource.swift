import Foundation

public protocol AudioSampleSource: Sendable {
    var sampleCount: Int { get }
    func copySamples(
        into destination: UnsafeMutablePointer<Float>,
        offset: Int,
        count: Int
    ) throws
}

public struct ArrayAudioSampleSource: AudioSampleSource {
    private let samples: [Float]

    public init(samples: [Float]) {
        self.samples = samples
    }

    public var sampleCount: Int {
        samples.count
    }

    public func copySamples(
        into destination: UnsafeMutablePointer<Float>,
        offset: Int,
        count: Int
    ) throws {
        guard count > 0 else { return }
        guard !samples.isEmpty else { return }
        let clampedOffset = max(0, offset)
        guard clampedOffset < samples.count else { return }
        let available = min(samples.count - clampedOffset, count)
        samples.withUnsafeBufferPointer { pointer in
            destination.update(
                from: pointer.baseAddress!.advanced(by: clampedOffset),
                count: available
            )
        }
    }
}

public struct DiskBackedAudioSampleSource: AudioSampleSource {
    /// How the mapped bytes encode samples.
    enum Encoding: Sendable {
        /// Native float32, the converter's temporary file.
        case float32
        /// 16-bit little-endian PCM at the target rate, mono: the wav file
        /// itself, mapped in place; `dataOffset` is where its data chunk starts.
        case pcm16(dataOffset: Int)
    }

    private let mappedData: Data
    private let encoding: Encoding
    /// A temporary file to delete in `cleanup()`; nil when the source maps
    /// the caller's own file.
    private let temporaryFileURL: URL?

    public let sampleCount: Int

    init(mappedData: Data, fileURL: URL) {
        self.mappedData = mappedData
        self.encoding = .float32
        self.temporaryFileURL = fileURL
        self.sampleCount = mappedData.count / MemoryLayout<Float>.stride
    }

    init(pcm16MappedWav mappedData: Data, dataOffset: Int, sampleCount: Int) {
        self.mappedData = mappedData
        self.encoding = .pcm16(dataOffset: dataOffset)
        self.temporaryFileURL = nil
        self.sampleCount = sampleCount
    }

    public func copySamples(
        into destination: UnsafeMutablePointer<Float>,
        offset: Int,
        count: Int
    ) throws {
        guard count > 0 else { return }
        guard sampleCount > 0 else { return }
        let clampedOffset = max(0, offset)
        guard clampedOffset < sampleCount else { return }
        let available = min(sampleCount - clampedOffset, count)
        mappedData.withUnsafeBytes { rawBuffer in
            switch encoding {
            case .float32:
                let floatBuffer = rawBuffer.bindMemory(to: Float.self)
                destination.update(
                    from: floatBuffer.baseAddress!.advanced(by: clampedOffset),
                    count: available
                )
            case .pcm16(let dataOffset):
                // The data chunk may sit at an odd byte offset relative to the
                // page; load each sample unaligned.
                let base = rawBuffer.baseAddress!.advanced(by: dataOffset + clampedOffset * 2)
                let scale = Float(1) / 32768
                for index in 0..<available {
                    let sample = base.loadUnaligned(fromByteOffset: index * 2, as: Int16.self)
                    destination[index] = Float(Int16(littleEndian: sample)) * scale
                }
            }
        }
    }

    public func cleanup() {
        guard let temporaryFileURL else { return }
        do {
            try FileManager.default.removeItem(at: temporaryFileURL)
        } catch {
            // Silently ignore cleanup failures; temporary directory will be purged eventually.
        }
    }
}

/// The layout of a RIFF/WAVE file whose samples can be read in place: 16-bit
/// PCM, one channel, at the given sample rate. Anything else (other rates,
/// stereo, float, compressed, non-wav) returns nil and goes through the
/// converter instead.
struct PcmWavLayout: Equatable {
    let dataOffset: Int
    let sampleCount: Int

    static func parse(_ data: Data, sampleRate: Int) -> PcmWavLayout? {
        return data.withUnsafeBytes { raw -> PcmWavLayout? in
            guard raw.count >= 12, tag(raw, 0) == "RIFF", tag(raw, 8) == "WAVE" else { return nil }
            var formatOK = false
            var cursor = 12
            while cursor + 8 <= raw.count {
                let chunk = tag(raw, cursor)
                let size = Int(raw.loadUnaligned(fromByteOffset: cursor + 4, as: UInt32.self).littleEndian)
                let body = cursor + 8
                if chunk == "fmt " {
                    guard body + 16 <= raw.count else { return nil }
                    let formatTag = raw.loadUnaligned(fromByteOffset: body, as: UInt16.self).littleEndian
                    let channels = raw.loadUnaligned(fromByteOffset: body + 2, as: UInt16.self).littleEndian
                    let rate = raw.loadUnaligned(fromByteOffset: body + 4, as: UInt32.self).littleEndian
                    let bits = raw.loadUnaligned(fromByteOffset: body + 14, as: UInt16.self).littleEndian
                    // 1 = PCM; 0xFFFE = WAVE_FORMAT_EXTENSIBLE, whose sub-format GUID starts with the PCM tag.
                    var pcm = formatTag == 1
                    if formatTag == 0xFFFE, size >= 40, body + 26 <= raw.count {
                        pcm = raw.loadUnaligned(fromByteOffset: body + 24, as: UInt16.self).littleEndian == 1
                    }
                    formatOK = pcm && channels == 1 && Int(rate) == sampleRate && bits == 16
                } else if chunk == "data" {
                    guard formatOK else { return nil }
                    let available = min(size, raw.count - body)
                    return PcmWavLayout(dataOffset: body, sampleCount: available / 2)
                }
                cursor = body + size + (size & 1)
            }
            return nil
        }
    }

    private static func tag(_ raw: UnsafeRawBufferPointer, _ offset: Int) -> String {
        guard offset + 4 <= raw.count else { return "" }
        return String(decoding: raw[offset..<offset + 4], as: UTF8.self)
    }
}
