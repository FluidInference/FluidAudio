import Foundation
import XCTest

@testable import FluidAudio

/// 16 kHz mono 16-bit wavs are read in place by the disk-backed source; every
/// other layout goes through the converter.
final class PcmWavSampleSourceTests: XCTestCase {

    private func wav(
        samples: [Int16], rate: UInt32 = 16000, channels: UInt16 = 1, bits: UInt16 = 16, formatTag: UInt16 = 1
    )
        -> Data
    {
        var body = Data()
        for sample in samples {
            var le = sample.littleEndian
            body.append(Data(bytes: &le, count: 2))
        }
        var data = Data("RIFF".utf8)
        func u32(_ v: UInt32) {
            var le = v.littleEndian
            data.append(Data(bytes: &le, count: 4))
        }
        func u16(_ v: UInt16) {
            var le = v.littleEndian
            data.append(Data(bytes: &le, count: 2))
        }
        u32(UInt32(36 + body.count))
        data.append(Data("WAVE".utf8))
        data.append(Data("fmt ".utf8))
        u32(16)
        u16(formatTag)
        u16(channels)
        u32(rate)
        u32(rate * UInt32(channels) * UInt32(bits / 8))
        u16(channels * bits / 8)
        u16(bits)
        // A LIST chunk before the data chunk, as some encoders write.
        data.append(Data("LIST".utf8))
        u32(3)
        data.append(Data([1, 2, 3, 0]))  // odd-sized chunk is padded
        data.append(Data("data".utf8))
        u32(UInt32(body.count))
        data.append(body)
        return data
    }

    func testParsesPcm16MonoAtTargetRate() {
        let data = wav(samples: [0, 16384, -32768, 32767])
        let layout = PcmWavLayout.parse(data, sampleRate: 16000)
        XCTAssertEqual(layout?.sampleCount, 4)
        XCTAssertEqual(layout?.dataOffset, data.count - 8)
    }

    func testRejectsOtherLayouts() {
        XCTAssertNil(PcmWavLayout.parse(wav(samples: [0, 0], rate: 44100), sampleRate: 16000))
        XCTAssertNil(PcmWavLayout.parse(wav(samples: [0, 0], channels: 2), sampleRate: 16000))
        XCTAssertNil(PcmWavLayout.parse(wav(samples: [0, 0], bits: 24), sampleRate: 16000))
        XCTAssertNil(PcmWavLayout.parse(wav(samples: [0, 0], formatTag: 3), sampleRate: 16000))
        XCTAssertNil(PcmWavLayout.parse(Data("not a wav".utf8), sampleRate: 16000))
    }

    func testCopySamplesConvertsInPlace() throws {
        let data = wav(samples: [0, 16384, -32768, 32767, 8192])
        let layout = try XCTUnwrap(PcmWavLayout.parse(data, sampleRate: 16000))
        let source = DiskBackedAudioSampleSource(
            pcm16MappedWav: data, dataOffset: layout.dataOffset, sampleCount: layout.sampleCount)
        XCTAssertEqual(source.sampleCount, 5)
        var out = [Float](repeating: 9, count: 5)
        try out.withUnsafeMutableBufferPointer { try source.copySamples(into: $0.baseAddress!, offset: 1, count: 3) }
        XCTAssertEqual(out[0], 0.5)
        XCTAssertEqual(out[1], -1)
        XCTAssertEqual(out[2], 32767 / 32768, accuracy: 1e-6)
        XCTAssertEqual(out[3], 9)  // untouched beyond count
        // Past the end: only what is available.
        out = [Float](repeating: 9, count: 5)
        try out.withUnsafeMutableBufferPointer { try source.copySamples(into: $0.baseAddress!, offset: 4, count: 3) }
        XCTAssertEqual(out[0], 0.25)
        XCTAssertEqual(out[1], 9)
    }

    func testFactoryReadsPcmWavWithoutATemporaryFile() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("pcm-\(UUID().uuidString).wav")
        try wav(samples: Array(repeating: 1000, count: 16000)).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let (source, _) = try AudioSourceFactory().makeDiskBackedSource(from: url, targetSampleRate: 16000)
        XCTAssertEqual(source.sampleCount, 16000)
        var out = [Float](repeating: 0, count: 2)
        try out.withUnsafeMutableBufferPointer { try source.copySamples(into: $0.baseAddress!, offset: 0, count: 2) }
        XCTAssertEqual(out[0], 1000 / 32768, accuracy: 1e-6)
        source.cleanup()
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "cleanup must not delete the caller's file")
    }
}
