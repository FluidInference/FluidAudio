import AVFoundation
import FluidAudio
import Foundation

/// 16 kHz mono helpers shared by the file and live modes.
enum AudioFiles {
    static let sampleRate = Double(LocalVqeManager.sampleRate)

    /// Any format/rate → 16 kHz mono Float32.
    static func read(_ url: URL) throws -> [Float] {
        try AudioConverter().resampleAudioFile(url)
    }

    /// 16-bit PCM WAV at 16 kHz mono.
    static func write(_ samples: [Float], to url: URL) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(
            forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        let buffer = try pcmBuffer(samples)
        try file.write(from: buffer)
    }

    static func pcmBuffer(_ samples: [Float]) throws -> AVAudioPCMBuffer {
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(max(samples.count, 1)))
        else {
            throw DemoError.message("Could not allocate a 16 kHz mono buffer")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let channel = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        }
        return buffer
    }

    /// Peak envelope (max |x| per bin) for drawing; `bins` columns.
    static func peaks(_ samples: [Float], bins: Int) -> [Float] {
        guard !samples.isEmpty, bins > 0 else { return [] }
        let per = max(1, samples.count / bins)
        return stride(from: 0, to: samples.count, by: per).prefix(bins).map { start in
            samples[start..<min(samples.count, start + per)].reduce(0) { max($0, abs($1)) }
        }
    }

    static func rms(_ samples: ArraySlice<Float>) -> Float {
        guard !samples.isEmpty else { return 0 }
        return (samples.reduce(0) { $0 + $1 * $1 } / Float(samples.count)).squareRoot()
    }

    /// Level in dBFS, floored at -80.
    static func dbfs(_ rms: Float) -> Float {
        rms <= 0 ? -80 : max(-80, 20 * log10(rms))
    }
}

enum DemoError: Error, LocalizedError {
    case message(String)

    var errorDescription: String? {
        switch self {
        case .message(let text): return text
        }
    }
}
