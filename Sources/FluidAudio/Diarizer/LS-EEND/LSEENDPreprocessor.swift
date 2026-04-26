import Foundation
import CoreML
import Accelerate

// MARK: - Feature Provider
public class LSEENDFeatureProvider {
    public struct Snapshot: ~Copyable {
        let state: LSEENDState
        let melQueue: StreamingChunkQueue
        let audioQueue: StreamingChunkQueue
        let cmnMean: [Float]
        let cmnCount: Int
        let decoderMaskEnd: Int
    }

    /// Number of mel chunks currently ready for `emitNextChunk()`.
    public var readyChunks: Int { lock.withLock { melQueue.readyChunks } }

    // MARK: Private Attributes

    private let melSpectrogram: AudioMelSpectrogram
    private let converter: AudioConverter
    private let input: LSEENDInput

    private var melQueue: StreamingChunkQueue
    private var audioQueue: StreamingChunkQueue

    private var cmnMean: [Float]
    private var cmnCount: Int

    private var isRightContextEmpty: Bool = true

    private var decoderMaskEnd: Int

    private let lock = NSLock()
    private let log10Scale: Float = 1.0 / log(10.0)
    private let decoderMask: [Float]

    /// Audio samples required past the last real sample to flush every
    /// buffered real frame through STFT + mel ±context + CNN right-lookahead.
    private let flushSampleCount: Int
    private let chunkFrames: Int
    private let nMels: Int

    // MARK: - Init
    public init(
        from metadata: borrowing LSEENDMetadata,
        restoringFrom snapshot: consuming Snapshot? = nil
    ) throws {
        self.nMels = metadata.nMels

        let contextMels = metadata.contextSize
        let contextSamples = metadata.nFFT / 2
        let chunkMels = metadata.subsampling * metadata.chunkSize
        let chunkSamples = metadata.hopLength * chunkMels
        let rightSamples = metadata.nFFT / 2 - metadata.hopLength

        // (mel ±context + CNN right-lookahead) mels × hop + STFT last-window halfNfft
        self.flushSampleCount =
            (contextMels + metadata.convDelay * metadata.subsampling) * metadata.hopLength
            + contextSamples

        self.chunkFrames = metadata.chunkSize

        var decoderMaskTemp = [Float](repeating: 1, count: metadata.convDelay + metadata.chunkSize)
        vDSP_vclr(&decoderMaskTemp, 1, vDSP_Length(metadata.convDelay))
        self.decoderMask = decoderMaskTemp

        // Initialize processors
        self.melSpectrogram = AudioMelSpectrogram(
            sampleRate: metadata.sampleRate,
            nMels: metadata.nMels,
            nFFT: metadata.nFFT,
            hopLength: metadata.hopLength,
            winLength: metadata.winLength,
            preemph: 0,
            padTo: 0,
            logFloor: 1e-10,
            logFloorMode: .clamped,
            windowPeriodic: true
        )

        self.converter = AudioConverter(sampleRate: Double(metadata.sampleRate))

        // Initialize state
        if let snapshot {
            self.melQueue = snapshot.melQueue
            self.audioQueue = snapshot.audioQueue
            self.cmnMean = snapshot.cmnMean
            self.cmnCount = snapshot.cmnCount
            self.decoderMaskEnd = snapshot.decoderMaskEnd
            self.input = try LSEENDInput(from: metadata, state: consume snapshot.state)
        } else {
            self.melQueue = StreamingChunkQueue(
                chunkLength: chunkMels,
                leftContextLength: contextMels,
                rightContextLength: contextMels + 1 - metadata.subsampling,
                stride: nMels
            )
            self.audioQueue = StreamingChunkQueue(
                chunkLength: chunkSamples,
                leftContextLength: contextSamples,
                rightContextLength: rightSamples,
                stride: 1
            )

            self.cmnMean = .init(repeating: 0, count: nMels)
            self.cmnCount = 0
            self.decoderMaskEnd = 0

            // Initialize preprocessor and converter
            self.input = try LSEENDInput(from: metadata)
        }
    }

    // MARK: - Push Audio

    /// Add audio to the processing queue
    /// - Parameters:
    ///   - samples: Audio samples to enqueue
    ///   - sourceSampleRate: Sample rate of audio input
    ///   - eagerPreprocessing: Whether to eagerly feed audio chunks to the mel spectrogram
    public func enqueueAudio<C: Collection>(
        _ samples: C,
        withSampleRate sourceSampleRate: Double? = nil,
        eagerPreprocessing: Bool = true
    ) throws where C.Element == Float {
        lock.lock()
        defer { lock.unlock() }

        if let sourceSampleRate {
            let array = (samples as? [Float]) ?? Array(samples)
            try audioQueue.append(converter.resample(array, from: sourceSampleRate))
        } else {
            audioQueue.append(samples)
        }

        if eagerPreprocessing {
            processAudioQueue()
        }
    }

    /// Resample and enqueue a full audio file.
    /// - Parameter url: Audio file to read.
    /// - Returns: Number of samples enqueued (at the model's sample rate).
    @discardableResult
    public func enqueueAudioFile(at url: URL) throws -> Int {
        let samples = try converter.resampleAudioFile(url)
        lock.lock()
        defer { lock.unlock() }
        audioQueue.append(samples)
        processAudioQueue()
        return samples.count
    }

    /// Add silence to push all queued audio out of the right context
    /// - Parameter flush Whether to flush the queued audio into the mel spectrogram preprocessor
    public func drainRightContextWithSilence(flush: Bool = true) throws {
        lock.lock()
        defer { lock.unlock() }

        // 1. Trailing silence covering STFT + mel ±context + CNN right-lookahead.
        audioQueue.append(repeatElement(0, count: flushSampleCount))

        // 2. Round up to the next audio-chunk boundary so popAllChunks consumes
        //    every real sample plus the silence we just pushed.
        let unread = audioQueue.unreadFloats
        let chunk = audioQueue.chunkFloats
        let ctx = audioQueue.contextFloats
        let overCtx = max(0, unread - ctx)
        let shortfall = (chunk - overCtx % chunk) % chunk
        if shortfall > 0 {
            audioQueue.append(repeatElement(0, count: shortfall))
        }

        // 3. Drain audioQueue → STFT → log10 → CMN → melQueue.
        if flush {
            processAudioQueue()
        }
    }

    // MARK: - Read Chunk

    /// Read the next chunk from the mel
    public func emitNextChunk() throws -> LSEENDInput? {
        lock.lock()
        defer { lock.unlock() }

        processAudioQueue()
        guard let rawChunk = melQueue.popNextChunk() else { return nil }

        // Advance decoder mask
        decoderMaskEnd = min(decoderMaskEnd + chunkFrames, decoderMask.count)

        try input.loadInputs(
            melFeatures: rawChunk,
            decoderMask: decoderMask[decoderMaskEnd - chunkFrames..<decoderMaskEnd],
            warmupFrames: min(decoderMask.count - decoderMaskEnd, chunkFrames)
        )

        return input
    }

    // MARK: - Snapshot and Rollback

    public func takeSnapshot() throws -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        let result = Snapshot(
            state: try input.state.copy(),
            melQueue: melQueue,
            audioQueue: audioQueue,
            cmnMean: cmnMean,
            cmnCount: cmnCount,
            decoderMaskEnd: decoderMaskEnd
        )
        return result
    }

    /// Rollback to a previous snapshot.
    /// - Parameters:
    ///   - snapshot Snapshot to revert to
    ///   - keepingState Whether to preserve the current recurrent state
    public func rollback(to snapshot: consuming Snapshot, keepingState: Bool = false) {
        lock.lock()
        defer { lock.unlock() }
        if !keepingState { self.input.state = snapshot.state }
        self.melQueue = snapshot.melQueue
        self.audioQueue = snapshot.audioQueue
        self.cmnMean = snapshot.cmnMean
        self.cmnCount = snapshot.cmnCount
        self.decoderMaskEnd = snapshot.decoderMaskEnd
    }

    /// Clear preprocessor buffers + model recurrence state + frame counter.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        vDSP.fill(&cmnMean, with: 0)
        cmnCount = 0
        decoderMaskEnd = 0
        audioQueue.reset()
        melQueue.reset()
        input.resetState()
    }

    // MARK: - Helpers

    private func processAudioQueue() {
        guard let audioChunk = audioQueue.popAllChunks() else { return }

        var (melFeats, melFrames, _) = melSpectrogram.computeFlatTransposed(
            audio: audioChunk,
            lastAudioSample: 0,
            paddingMode: .prePadded,
            expectedFrameCount: nil
        )

        // Rescale to use log10 instead of ln
        var scale = log10Scale
        vDSP_vsmul(melFeats, 1, &scale, &melFeats, 1, vDSP_Length(melFrames * nMels))

        // Cumulative mean normalization — sequential by definition.
        melFeats.withUnsafeMutableBufferPointer { melFeatsBuffer in
            let melFrameLength = vDSP_Length(nMels)
            guard let melBase = melFeatsBuffer.baseAddress else { return }

            for melFrame in stride(from: melBase, to: melBase + melFeatsBuffer.count, by: nMels) {
                // µ[k] = µ[k-1] + (mel[k] - µ[k-1]) * 1 / k
                cmnCount += 1
                var alpha = 1.0 / Float(cmnCount)
                vDSP_vintb(cmnMean, 1, melFrame, 1, &alpha, &cmnMean, 1, melFrameLength)
                // mel[k] <- mel[k] - µ[k]. vDSP_vsub(A,_,B,_,C,_,N) is C = B - A.
                vDSP_vsub(cmnMean, 1, melFrame, 1, melFrame, 1, melFrameLength)
            }
        }

        melQueue.append(consume melFeats)
    }
}

// MARK: - Streaming Chunk Queue

public struct StreamingChunkQueue {
    /// Stride between frames if features are n-dimensional arrays
    public let stride: Int

    /// Total context size in floats (`leftContextFloats + rightContextFloats`).
    public let contextFloats: Int

    /// Unpadded chunk size
    public let chunkFloats: Int

    /// Padded chunk size — width of a `popNextChunk` / `popAllChunks` slice.
    public let paddedChunkFloats: Int

    /// Whether the buffer is empty
    public var isEmpty: Bool { buffer.isEmpty }

    /// Number of unread floats
    public var unreadFloats: Int { buffer.count - head }

    /// Number of full chunks currently poppable via `popNextChunk` / `popAllChunks`.
    public var readyChunks: Int { max(0, (unreadFloats - contextFloats) / chunkFloats) }

    // MARK: - Private attributes

    /// Pre-pad width in floats — how many leading zeros are seeded at init
    private let leftContextFloats: Int

    /// Next index at which to start processing
    private var head: Int

    /// Data buffer
    private var buffer: [Float] = []

    /// Whether a chunk is ready
    public var hasChunk: Bool {
        buffer.count - head >= paddedChunkFloats
    }

    // MARK: - Init

    /// - Parameters:
    ///   - chunkLength Number of frames in a chunk
    ///   - leftContextLength Number of frames in a chunk's left context
    ///   - rightContextLength Number of frames in a chunk's right context (may be negative)
    ///   - stride Number of floats in a frame
    public init(
        chunkLength: Int,
        leftContextLength: Int,
        rightContextLength: Int,
        stride: Int
    ) {
        self.stride = stride
        self.chunkFloats = chunkLength * stride
        self.leftContextFloats = leftContextLength * stride
        self.contextFloats = (leftContextLength + rightContextLength) * stride
        self.paddedChunkFloats = chunkFloats + contextFloats

        self.head = 0

        self.buffer.reserveCapacity(paddedChunkFloats * 2)
        self.buffer.append(contentsOf: repeatElement(0, count: leftContextFloats))
    }

    // MARK: - Append and Pop

    public mutating func append<C: Collection>(_ newElements: C)
    where C.Element == Float {
        // Lazy trimming
        if buffer.count + newElements.count > buffer.capacity {
            buffer.removeFirst(head)
            head = 0
        }

        // Allow Swift to reserve more memory if needed after the trimming
        buffer.append(contentsOf: newElements)
    }

    /// Pop the last chunk
    public mutating func popNextChunk() -> ArraySlice<Float>? {
        guard hasChunk else { return nil }
        let result = buffer[head..<head + paddedChunkFloats]
        head += chunkFloats
        return result
    }

    /// Pop all available chunks as one buffer
    public mutating func popAllChunks() -> ArraySlice<Float>? {
        guard hasChunk else { return nil }
        let newHead = head + (buffer.count - head - contextFloats) / chunkFloats * chunkFloats
        let result = buffer[head..<newHead + contextFloats]
        head = newHead
        return result
    }

    /// Reset buffer
    public mutating func reset() {
        self.head = 0
        self.buffer.removeAll(keepingCapacity: true)
        self.buffer.append(contentsOf: repeatElement(0, count: leftContextFloats))
    }
}
