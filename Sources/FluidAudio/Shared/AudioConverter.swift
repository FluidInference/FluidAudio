@preconcurrency import AVFoundation
import Accelerate
import CoreMedia
import Foundation
import OSLog
import os

/// Converts audio buffers to the format required by ASR (16kHz, mono, Float32).
///
/// Implementation notes:
/// - Uses `AVAudioConverter` for all sample-rate, sample-format, and channel-count conversions.
/// - Avoids any manual resampling; only raw sample extraction occurs after conversion.
/// - Creates a new converter for each operation (stateless).
final public class AudioConverter: Sendable {
    private let logger = AppLogger(category: "AudioConverter")
    private let targetFormat: AVAudioFormat
    private let debug: Bool

    /// Public initializer so external modules (e.g. CLI) can construct the converter
    /// - Parameters:
    ///   - targetFormat: Target audio format
    ///   - debug: Whether to log debug messages
    public init(targetFormat: AVAudioFormat? = nil, debug: Bool = false) {
        self.debug = debug
        if let format = targetFormat {
            self.targetFormat = format
        } else {
            /// Most audio models expect this format.
            /// Target format for ASR, Speaker diarization model: 16kHz, mono
            self.targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16000,
                channels: 1,
                interleaved: false
            )!
        }
    }

    /// Public initializer so external modules (e.g. CLI) can construct the converter
    /// - Parameters:
    ///   - sampleRate: Target audio sample rate
    ///   - debug: Whether to log debug messages
    public init(sampleRate: Double, debug: Bool = false) {
        self.debug = debug
        self.targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!
    }

    // MARK: - Public Resampling Methods

    /// Resample a float array from one sample rate to the target sample rate.
    /// - Parameters:
    ///   - samples: Input audio samples (mono Float32)
    ///   - inputRate: Input sample rate (e.g., 48000)
    /// - Returns: Float array resampled to target sample rate
    public func resample(_ samples: [Float], from inputRate: Double) throws -> [Float] {
        guard !samples.isEmpty else { return [] }

        let outputRate = targetFormat.sampleRate

        // If already at target rate, return as-is
        if inputRate == outputRate {
            return samples
        }

        return try resampleWithAVAudio(samples, from: inputRate, to: outputRate)
    }

    /// Convert a standalone buffer to the target format.
    /// - Parameters:
    ///   - buffer: Input audio buffer (any format)
    /// - Returns: Float array at target sample rate (default 16kHz) mono
    public func resampleBuffer(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        let inputFormat = buffer.format

        // Fast path: if already in target format, just extract samples
        if isTargetFormat(inputFormat) {
            return extractFloatArray(from: buffer)
        }
        return try convertBuffer(buffer, to: targetFormat)
    }

    /// Convert an audio file to target sample rate (default 16kHz) mono Float32 samples.
    ///
    /// The file is converted as it is read: each chunk is mixed to mono and fed
    /// straight into one `AVAudioConverter`, which carries its filter state from
    /// chunk to chunk. The recording is never held at its source rate, so the
    /// peak is the output itself — about 1.5 GB for 6 h 30 min at 16 kHz,
    /// where reading it all first held the 48 kHz copy beside it (~6 GB).
    /// - Parameters:
    ///   - url: URL of the audio file to read
    /// - Returns: Float array at target sample rate mono
    public func resampleAudioFile(_ url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let chunkSize = max(4096, Int(format.sampleRate))
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(chunkSize)) else {
            throw AudioConverterError.failedToCreateBuffer
        }

        var readAny = false
        let nextChunk: () throws -> [Float]? = {
            guard audioFile.framePosition < audioFile.length else { return nil }
            let remaining = Int(audioFile.length - audioFile.framePosition)
            let framesToRead = AVAudioFrameCount(min(chunkSize, remaining))
            do {
                try audioFile.read(into: buffer, frameCount: framesToRead)
            } catch {
                // AVAudioFile.length is an ESTIMATE for packetized formats
                // (MP3/AAC): encoder delay/padding accounting can overshoot the
                // decodable stream by a few hundred frames, and reading the
                // phantom tail throws a contract-breaking nilError
                // (Foundation._GenericObjCError 0) instead of returning an
                // empty buffer. Everything decodable is already in hand —
                // treat a failed read after real audio as EOF rather than
                // failing the whole file. A throw on the FIRST read is a
                // genuinely unreadable file and still propagates.
                if !readAny { throw error }
                return nil
            }
            if buffer.frameLength == 0 { return nil }
            readAny = true
            return try self.extractMonoFloat32(from: buffer)
        }

        return try streamResample(
            from: format.sampleRate,
            to: targetFormat.sampleRate,
            expectedInputFrames: Int(audioFile.length),
            maxChunkFrames: chunkSize,
            nextChunk: nextChunk
        )
    }

    /// Convert an audio file path to target sample rate mono Float32 samples.
    /// - Parameters:
    ///   - path: File path of the audio file to read
    /// - Returns: Float array at target sample rate mono
    public func resampleAudioFile(path: String) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        return try resampleAudioFile(url)
    }

    // MARK: - Private Helpers

    /// Resample using AVAudioConverter with raw float arrays.
    ///
    /// The samples are fed to the converter in slices rather than as one
    /// buffer: an `AVAudioPCMBuffer` counts its bytes in a `UInt32`, so a single
    /// buffer for a whole recording cannot exceed 4 GiB — 6 h 12 min of mono
    /// at 48 kHz, 6 h 46 min at 44.1 kHz — and the allocation fails past that.
    /// One converter instance carries its filter state across the slices, so
    /// the output is the same as a single-buffer conversion.
    private func resampleWithAVAudio(
        _ samples: [Float], from inputRate: Double, to outputRate: Double
    ) throws -> [Float] {
        try resampleWithAVAudio(samples, from: inputRate, to: outputRate, sliceFrames: Self.resampleSliceFrames)
    }

    /// Frames per slice handed to the converter, and the most a single output
    /// buffer holds: 1 Mi frames is 4 MiB of Float32, well under the limit and
    /// large enough that the converter's per-call overhead does not show.
    static let resampleSliceFrames = 1 << 20

    /// `sliceFrames` is a parameter so a test can prove slicing does not change
    /// the result; production callers take the default.
    func resampleWithAVAudio(
        _ samples: [Float], from inputRate: Double, to outputRate: Double, sliceFrames: Int
    ) throws -> [Float] {
        let slice = max(1, min(sliceFrames, max(samples.count, 1)))
        var offset = 0
        let nextChunk: () throws -> [Float]? = {
            guard offset < samples.count else { return nil }
            let end = min(offset + slice, samples.count)
            defer { offset = end }
            return Array(samples[offset..<end])
        }
        return try streamResample(
            from: inputRate, to: outputRate,
            expectedInputFrames: samples.count, maxChunkFrames: slice,
            nextChunk: nextChunk
        )
    }

    /// Resample mono Float32 audio supplied a chunk at a time, through one
    /// converter, into fixed-size output buffers. `nextChunk` returns at most
    /// `maxChunkFrames` samples per call and nil at the end; an error it throws
    /// ends the stream and is rethrown. Nothing here is sized to the whole
    /// recording except the returned array.
    private func streamResample(
        from inputRate: Double, to outputRate: Double,
        expectedInputFrames: Int, maxChunkFrames: Int,
        nextChunk: @escaping () throws -> [Float]?
    ) throws -> [Float] {
        let ratio = outputRate / inputRate
        var aggregated: [Float] = []
        aggregated.reserveCapacity(Int((Double(expectedInputFrames) * ratio).rounded(.up)) + 4096)

        // Already at the target rate: the chunks are the output.
        if inputRate == outputRate {
            while let chunk = try nextChunk() { aggregated.append(contentsOf: chunk) }
            return aggregated
        }

        guard
            let inputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: inputRate,
                channels: 1,
                interleaved: false
            ),
            let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: outputRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw AudioConverterError.failedToCreateSourceFormat
        }
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioConverterError.failedToCreateConverter
        }
        configure(converter: converter)

        let capacity = max(1, maxChunkFrames)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(capacity))
        else {
            throw AudioConverterError.failedToCreateBuffer
        }
        let outputCapacity = AVAudioFrameCount((Double(capacity) * ratio).rounded(.up)) + 4096
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputCapacity) else {
            throw AudioConverterError.failedToCreateBuffer
        }

        // The converter pulls input synchronously inside convert(...); the
        // feed's state is only touched from there.
        let ended = OSAllocatedUnfairLock(initialState: false)
        let feedError = OSAllocatedUnfairLock<Error?>(initialState: nil)
        nonisolated(unsafe) let input = inputBuffer
        nonisolated(unsafe) let feed = nextChunk
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            if ended.withLock({ $0 }) {
                status.pointee = .endOfStream
                return nil
            }
            do {
                while let chunk = try feed() {
                    guard !chunk.isEmpty else { continue }
                    guard chunk.count <= capacity, let channel = input.floatChannelData else {
                        throw AudioConverterError.failedToCreateBuffer
                    }
                    chunk.withUnsafeBufferPointer { src in
                        memcpy(channel[0], src.baseAddress!, chunk.count * MemoryLayout<Float>.stride)
                    }
                    input.frameLength = AVAudioFrameCount(chunk.count)
                    status.pointee = .haveData
                    return input
                }
            } catch {
                feedError.withLock { $0 = error }
            }
            ended.withLock { $0 = true }
            status.pointee = .endOfStream
            return nil
        }

        var error: NSError?
        while true {
            outputBuffer.frameLength = 0
            let status = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
            if let failure = feedError.withLock({ $0 }) { throw failure }
            guard status != .error else { throw AudioConverterError.conversionFailed(error) }
            if outputBuffer.frameLength > 0 { aggregated.append(contentsOf: extractFloatArray(from: outputBuffer)) }
            if status == .endOfStream { break }
        }
        if debug {
            logger.debug(
                "Audio resample: \(expectedInputFrames) samples → \(aggregated.count) samples, ratio: \(Double(aggregated.count)/Double(max(expectedInputFrames, 1)))"
            )
        }
        return aggregated
    }

    /// Extract mono Float32 samples from a buffer (mixing channels if needed)
    private func extractMonoFloat32(from buffer: AVAudioPCMBuffer) throws -> [Float] {
        let format = buffer.format
        let frameCount = Int(buffer.frameLength)

        guard frameCount > 0 else { return [] }

        // If already mono Float32 non-interleaved, fast path
        if format.channelCount == 1 && format.commonFormat == .pcmFormatFloat32 && !format.isInterleaved {
            guard let channelData = buffer.floatChannelData else { return [] }
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        }

        // Need to convert to mono Float32
        // Create intermediate format (mono Float32 at original sample rate)
        guard
            let monoFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: format.sampleRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw AudioConverterError.failedToCreateSourceFormat
        }

        // Use AVAudioConverter for channel mixing (same sample rate, no resampling needed)
        guard let converter = AVAudioConverter(from: format, to: monoFormat) else {
            throw AudioConverterError.failedToCreateConverter
        }

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: buffer.frameCapacity) else {
            throw AudioConverterError.failedToCreateBuffer
        }

        let provided = OSAllocatedUnfairLock(initialState: false)
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            let wasProvided = provided.withLock { state -> Bool in
                if state { return true }
                state = true
                return false
            }
            if !wasProvided {
                status.pointee = .haveData
                return buffer
            } else {
                status.pointee = .endOfStream
                return nil
            }
        }

        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
        guard status != .error else {
            throw AudioConverterError.conversionFailed(error)
        }

        guard let channelData = outputBuffer.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: channelData[0], count: Int(outputBuffer.frameLength)))
    }

    /// Convert a CMSampleBuffer to the target format
    /// - Parameter sampleBuffer: Input CMSampleBuffer containing PCM data
    /// - Returns: Float array at 16kHz mono
    /// - Throws: `AudioConverterError.sampleBufferFormatMissing` (most likely caused by the sample buffer belonging
    ///  to a video frame)
    public func resampleSampleBuffer(_ sampleBuffer: CMSampleBuffer) throws -> [Float] {
        let buffer = try extractAVAudioPCMBuffer(from: sampleBuffer)
        return try convertBuffer(buffer, to: targetFormat)
    }

    /// Extract the `AVAudioPCMBuffer` from a `CMSampleBuffer`
    /// - Parameter sampleBuffer: Input CMSampleBuffer containing PCM data
    /// - Returns: An `AVAudioPCMBuffer`
    /// - Throws: `AudioConverterError.sampleBufferFormatMissing` (most likely caused by the sample buffer belonging
    ///  to a video frame)
    public func extractAVAudioPCMBuffer(from sampleBuffer: CMSampleBuffer) throws -> AVAudioPCMBuffer {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
            let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription)
        else {
            throw AudioConverterError.sampleBufferFormatMissing
        }

        guard let sourceFormat = AVAudioFormat(streamDescription: streamDescription) else {
            throw AudioConverterError.failedToCreateSourceFormat
        }

        let frameCount = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))

        guard let buffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            throw AudioConverterError.failedToCreateBuffer
        }
        buffer.frameLength = frameCount

        guard frameCount > 0 else {
            return buffer
        }

        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: buffer.mutableAudioBufferList
        )

        guard status == noErr else {
            throw AudioConverterError.sampleBufferCopyFailed(status)
        }

        return buffer
    }

    /// Convert a buffer to the target format.
    private func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> [Float] {
        try convertBuffer(buffer, to: format, outputSliceFrames: Self.resampleSliceFrames)
    }

    /// `outputSliceFrames` caps the one output buffer; a parameter so a test can
    /// prove a small cap gives the same samples. Production takes the default.
    func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat, outputSliceFrames: Int) throws -> [Float] {
        let inputFormat = buffer.format

        // For >2 channels, use manual linear resampling since AVAudioConverter has limitations
        if inputFormat.channelCount > 2 {
            return try linearResample(buffer, to: format)
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: format) else {
            throw AudioConverterError.failedToCreateConverter
        }
        configure(converter: converter)

        // One fixed-size output buffer, drained until end of stream. Sizing it
        // to the whole estimated output put it under the same UInt32 byte cap
        // as any single AVAudioPCMBuffer.
        let sampleRateRatio = format.sampleRate / inputFormat.sampleRate
        let estimatedOutputFrames = Int((Double(buffer.frameLength) * sampleRateRatio).rounded(.up))
        let outputCapacity = AVAudioFrameCount(max(4096, min(estimatedOutputFrames + 4096, outputSliceFrames)))
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: outputCapacity) else {
            throw AudioConverterError.failedToCreateBuffer
        }

        var aggregated: [Float] = []
        aggregated.reserveCapacity(estimatedOutputFrames)

        // AVAudioConverter consumes this input block synchronously within convert(...),
        // but Swift 6 rejects mutation of captured vars in this callback.
        let provided = OSAllocatedUnfairLock(initialState: false)
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            let wasProvided = provided.withLock { state -> Bool in
                if state { return true }
                state = true
                return false
            }
            if !wasProvided {
                status.pointee = .haveData
                return buffer
            } else {
                status.pointee = .endOfStream
                return nil
            }
        }

        var error: NSError?
        let inputSampleCount = Int(buffer.frameLength)

        while true {
            out.frameLength = 0
            let status = converter.convert(to: out, error: &error, withInputFrom: inputBlock)
            guard status != .error else { throw AudioConverterError.conversionFailed(error) }
            if out.frameLength > 0 { aggregated.append(contentsOf: extractFloatArray(from: out)) }
            if status == .endOfStream { break }
        }

        let outputSampleCount = aggregated.count
        if debug {
            logger.debug(
                "Audio conversion: \(inputSampleCount) samples → \(outputSampleCount) samples, ratio: \(Double(outputSampleCount)/Double(inputSampleCount))"
            )
        }

        return aggregated
    }

    private func configure(converter: AVAudioConverter) {
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
    }

    /// Check if a format already matches the target output format.
    private func isTargetFormat(_ format: AVAudioFormat) -> Bool {
        return format.sampleRate == targetFormat.sampleRate
            && format.channelCount == targetFormat.channelCount
            && format.commonFormat == targetFormat.commonFormat
            && format.isInterleaved == targetFormat.isInterleaved
    }

    /// Resample high channel count audio (>2 channels) using linear interpolation
    /// AVAudioConverter has limitations with >2 channels, so we handle it via a linear resample. Accuracy may not be as good as AVAudioConverter.
    /// But this is needed for applications like Safari on speaker mode, or for particular hardware devices.
    private func linearResample(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> [Float] {
        let inputFormat = buffer.format
        guard let channelData = buffer.floatChannelData else {
            throw AudioConverterError.failedToCreateBuffer
        }

        let inputFrameCount = Int(buffer.frameLength)
        let channelCount = Int(inputFormat.channelCount)

        // Step 1: Mix down to mono
        var monoSamples = [Float](repeating: 0, count: inputFrameCount)
        let channelWeight = 1.0 / Float(channelCount)

        for frame in 0..<inputFrameCount {
            var sum: Float = 0
            for channel in 0..<channelCount {
                sum += channelData[channel][frame]
            }
            monoSamples[frame] = sum * channelWeight
        }

        // Step 2: Resample if needed
        let inputSampleRate = inputFormat.sampleRate
        let targetSampleRate = format.sampleRate

        if inputSampleRate == targetSampleRate {
            return monoSamples
        }

        // Linear interpolation resampling
        let resampleRatio = inputSampleRate / targetSampleRate
        let outputFrameCount = Int(Double(inputFrameCount) / resampleRatio)
        var outputSamples = [Float](repeating: 0, count: outputFrameCount)

        for i in 0..<outputFrameCount {
            let sourceIndex = Double(i) * resampleRatio
            let index = Int(sourceIndex)
            let fraction = Float(sourceIndex - Double(index))

            if index < inputFrameCount - 1 {
                // Linear interpolation between samples
                outputSamples[i] = monoSamples[index] * (1.0 - fraction) + monoSamples[index + 1] * fraction
            } else if index < inputFrameCount {
                outputSamples[i] = monoSamples[index]
            }
        }

        if debug {
            logger.debug(
                "Manual resampling: \(channelCount) channels → mono, \(inputSampleRate)Hz → \(targetSampleRate)Hz"
            )
        }

        return outputSamples
    }

    /// Extract Float array from PCM buffer
    private func extractFloatArray(from buffer: AVAudioPCMBuffer) -> [Float] {
        // This function assumes mono, non-interleaved Float32 buffers.
        // All multi-channel or interleaved inputs should be converted via AVAudioConverter first.
        guard let channelData = buffer.floatChannelData else { return [] }

        let frameCount = Int(buffer.frameLength)
        if frameCount == 0 { return [] }

        // Enforce mono; converter guarantees this in normal flow.
        assert(buffer.format.channelCount == 1, "extractFloatArray expects mono buffers")

        // Fast copy using vDSP (equivalent to memcpy for contiguous Float32)
        let out = [Float](unsafeUninitializedCapacity: frameCount) { dest, initialized in
            vDSP_mmov(
                channelData[0],
                dest.baseAddress!,
                vDSP_Length(frameCount),
                1,
                1,
                1
            )
            initialized = frameCount
        }
        return out
    }

}

// MARK: - WAV Utilities (shared by TTS/ASR)
public enum AudioWAV {
    /// Convert float samples to 16-bit PCM mono WAV at the given sample rate.
    ///
    /// - Parameter normalize: when `true` (default) the samples are peak-scaled
    ///   to ±1.0 before quantization (consistent loudness). Pass `false` to
    ///   write the samples at their native level — used by backends whose
    ///   model output is already correctly leveled (e.g. KokoroAne, which
    ///   matches the PyTorch reference level) so the output isn't slammed to
    ///   0 dBFS. Out-of-range samples are still clamped to [-1, 1].
    public static func data(
        from samples: [Float], sampleRate: Double, normalize: Bool = true
    ) throws -> Data {
        let maxVal = samples.map { abs($0) }.max() ?? 1.0
        let norm = (normalize && maxVal > 0) ? samples.map { $0 / maxVal } : samples

        // Convert to 16-bit PCM
        var pcm = Data()
        pcm.reserveCapacity(norm.count * MemoryLayout<Int16>.size)
        for s in norm {
            let clipped = max(-1.0, min(1.0, s))
            let v = Int16(clipped * 32767)
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { pcm.append(contentsOf: $0) }
        }

        // Build WAV header
        var wav = Data()
        // RIFF header
        wav.append(contentsOf: "RIFF".data(using: .ascii)!)
        var fileSize = UInt32(36 + pcm.count).littleEndian
        withUnsafeBytes(of: &fileSize) { wav.append(contentsOf: $0) }
        wav.append(contentsOf: "WAVE".data(using: .ascii)!)

        // fmt chunk
        wav.append(contentsOf: "fmt ".data(using: .ascii)!)
        var subchunk1Size = UInt32(16).littleEndian  // PCM
        withUnsafeBytes(of: &subchunk1Size) { wav.append(contentsOf: $0) }
        var audioFormat = UInt16(1).littleEndian  // PCM
        withUnsafeBytes(of: &audioFormat) { wav.append(contentsOf: $0) }
        var numChannels = UInt16(1).littleEndian
        withUnsafeBytes(of: &numChannels) { wav.append(contentsOf: $0) }
        var sr = UInt32(sampleRate).littleEndian
        withUnsafeBytes(of: &sr) { wav.append(contentsOf: $0) }
        var byteRate = UInt32(sampleRate * 2).littleEndian  // 16-bit mono
        withUnsafeBytes(of: &byteRate) { wav.append(contentsOf: $0) }
        var blockAlign = UInt16(2).littleEndian
        withUnsafeBytes(of: &blockAlign) { wav.append(contentsOf: $0) }
        var bitsPerSample = UInt16(16).littleEndian
        withUnsafeBytes(of: &bitsPerSample) { wav.append(contentsOf: $0) }

        // data chunk
        wav.append(contentsOf: "data".data(using: .ascii)!)
        var dataSize = UInt32(pcm.count).littleEndian
        withUnsafeBytes(of: &dataSize) { wav.append(contentsOf: $0) }
        wav.append(pcm)

        return wav
    }
}

/// Errors that can occur during audio conversion
public enum AudioConverterError: LocalizedError {
    case failedToCreateConverter
    case failedToCreateBuffer
    case conversionFailed(Error?)
    case sampleBufferFormatMissing
    case failedToCreateSourceFormat
    case sampleBufferCopyFailed(OSStatus)

    public var errorDescription: String? {
        switch self {
        case .failedToCreateConverter:
            return "Failed to create audio converter"
        case .failedToCreateBuffer:
            return "Failed to create conversion buffer"
        case .conversionFailed(let error):
            return "Audio conversion failed: \(error?.localizedDescription ?? "Unknown error")"
        case .sampleBufferFormatMissing:
            return "Sample buffer is missing a valid audio format description."
        case .failedToCreateSourceFormat:
            // This edge case usually arises when a video sample is provided instead of an audio sample
            return "Failed to create a source audio format description for CMSampleBuffer."
        case .sampleBufferCopyFailed(let status):
            return "Failed to copy samples from CMSampleBuffer (status: \(status))"
        }
    }
}
