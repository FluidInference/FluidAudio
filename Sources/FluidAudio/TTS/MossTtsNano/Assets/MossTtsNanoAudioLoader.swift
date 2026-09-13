@preconcurrency import AVFoundation
import Foundation
import os

/// Decodes any AVFoundation-readable file to 48 kHz stereo Float32 (mono is duplicated).
enum MossTtsNanoAudioLoader {

    static func loadStereo48k(url: URL) throws -> (left: [Float], right: [Float]) {
        do {
            let file = try AVAudioFile(forReading: url)
            guard
                let target = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32, sampleRate: Double(MossTtsNanoConstants.sampleRate),
                    channels: 2, interleaved: false)
            else { throw MossTtsNanoError.audioLoadFailed(path: url.path, underlying: "cannot build 48 kHz format") }
            let frameCount = AVAudioFrameCount(file.length)
            guard let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: max(frameCount, 1))
            else { throw MossTtsNanoError.audioLoadFailed(path: url.path, underlying: "cannot allocate buffer") }
            try file.read(into: input)

            guard let converter = AVAudioConverter(from: file.processingFormat, to: target) else {
                throw MossTtsNanoError.audioLoadFailed(
                    path: url.path, underlying: "no converter for \(file.processingFormat)")
            }
            let ratio = target.sampleRate / file.processingFormat.sampleRate
            let capacity = AVAudioFrameCount(Double(input.frameLength) * ratio) + 4096
            guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
                throw MossTtsNanoError.audioLoadFailed(path: url.path, underlying: "cannot allocate output buffer")
            }
            let provided = OSAllocatedUnfairLock(initialState: false)
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                let wasProvided = provided.withLock { state -> Bool in
                    if state { return true }
                    state = true
                    return false
                }
                if wasProvided {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                outStatus.pointee = .haveData
                return input
            }
            var conversionError: NSError?
            let status = converter.convert(to: output, error: &conversionError, withInputFrom: inputBlock)
            if status == .error || conversionError != nil {
                throw MossTtsNanoError.audioLoadFailed(
                    path: url.path, underlying: conversionError?.localizedDescription ?? "conversion failed")
            }
            let n = Int(output.frameLength)
            guard let channels = output.floatChannelData, n > 0 else {
                throw MossTtsNanoError.audioLoadFailed(path: url.path, underlying: "empty audio")
            }
            let left = Array(UnsafeBufferPointer(start: channels[0], count: n))
            let right = Array(UnsafeBufferPointer(start: channels[1], count: n))
            return (left, right)
        } catch let error as MossTtsNanoError {
            throw error
        } catch {
            throw MossTtsNanoError.audioLoadFailed(path: url.path, underlying: "\(error)")
        }
    }
}
