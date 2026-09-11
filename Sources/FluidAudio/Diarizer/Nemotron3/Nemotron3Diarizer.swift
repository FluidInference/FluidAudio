import Foundation

/// Streaming 8-speaker diarizer backed by NVIDIA's Nemotron 3 Diarization preview.
///
/// Processes audio in fixed 80 ms-frame chunks through the CoreML forward pass and applies
/// NeMo's async speaker-cache/FIFO update host-side. Output is per-frame speaker activity
/// probability at 10 ms resolution, speaker slots ordered by first arrival.
///
/// - Important: This class is **not** thread-safe.
public final class Nemotron3Diarizer {

    public let config: Nemotron3Config
    private let models: Nemotron3Models
    private let updater: Nemotron3StateUpdater
    private var state: Nemotron3StreamingState
    private let logger = AppLogger(category: "Nemotron3Diarizer")

    /// Wall-time breakdown of the last `processComplete` call, in seconds.
    public struct PipelineProfile: Sendable {
        public var melSeconds: Double = 0
        public var chunkSliceSeconds: Double = 0
        public var inferenceSeconds: Double = 0
        public var inputPrepSeconds: Double = 0
        public var predictSeconds: Double = 0
        public var readbackSeconds: Double = 0
        public var stateUpdateSeconds: Double = 0
        public var outputAppendSeconds: Double = 0
        public var totalSeconds: Double = 0
        public var chunkCount: Int = 0
        /// Chunks skipped by VAD gating (no speech in the chunk's core window).
        public var skippedChunks: Int = 0
    }

    /// Populated by `processComplete`; read after the call for stage-level analysis.
    public private(set) var lastProfile = PipelineProfile()

    public init(config: Nemotron3Config, models: Nemotron3Models) {
        self.config = config
        self.models = models
        self.updater = Nemotron3StateUpdater(config: config, silenceEmbedding: models.silenceEmbedding)
        self.state = Nemotron3StreamingState(config: config)
    }

    public func reset() {
        state = Nemotron3StreamingState(config: config)
    }

    /// Process a complete audio buffer (16 kHz mono) and return per-frame speaker
    /// probabilities at 10 ms resolution, [frames * 8] flattened.
    /// Process a complete audio buffer.
    ///
    /// - Parameters:
    ///   - audio: 16 kHz mono samples.
    ///   - speechMask: Optional per-10 ms-frame speech mask (e.g. from `VadManager`).
    ///     Chunks whose core window contains no `true` frame skip inference entirely and
    ///     emit zero probabilities; streaming state does not advance across them (the
    ///     skipped region behaves like a pause in the stream). Callers should pre-pad
    ///     speech regions (~1 s) to protect onsets/offsets.
    public func processComplete(
        _ audio: [Float], speechMask: [Bool]? = nil
    ) throws -> (probabilities: [Float], frameCount: Int) {
        reset()
        var profile = PipelineProfile()
        let t0 = Date()

        var tStage = Date()
        let mel = AudioMelSpectrogram()
        let (featSeq, featLength, featSeqLength) = mel.computeFlatTransposed(audio: audio)
        profile.melSeconds = Date().timeIntervalSince(tStage)

        var total = [Float]()
        total.reserveCapacity(featLength * config.numSpeakers)

        var loader = Nemotron3FeatureLoader(
            config: config, featSeq: featSeq, featLength: featLength, featSeqLength: featSeqLength)
        let sub = config.subsamplingFactor
        var coreStart = 0
        // Each chunk's prediction allocates IOSurface-backed output arrays; without a
        // per-iteration autorelease drain, long ANE-route runs exhaust the IOSurface
        // pool after a few thousand calls (issue #752 failure class).
        while try autoreleasepool(invoking: { () -> Bool in
            tStage = Date()
            guard let chunk = loader.next() else { return false }
            profile.chunkSliceSeconds += Date().timeIntervalSince(tStage)

            // VAD gate: emit zeros for speech-free chunks without running the model or
            // advancing state. Output frame count must match the normal path exactly.
            let coreEnd = min(coreStart + config.chunkLen * sub, featLength)
            if let speechMask {
                let lo = min(coreStart, speechMask.count)
                let hi = min(coreEnd, speechMask.count)
                let hasSpeech = lo < hi && speechMask[lo..<hi].contains(true)
                if !hasSpeech {
                    let lcEnc = (chunk.leftOffset + sub / 2) / sub
                    let rcEnc = (chunk.rightOffset + sub - 1) / sub
                    let encLen = (chunk.length + sub - 1) / sub
                    let chunkFrames = min(
                        max(encLen - lcEnc, 0), config.chunkEncFrames - lcEnc - rcEnc)
                    total.append(
                        contentsOf: repeatElement(
                            0, count: chunkFrames * config.upsampleFactor * config.numSpeakers))
                    profile.skippedChunks += 1
                    coreStart = coreEnd
                    return true
                }
            }
            coreStart = coreEnd

            tStage = Date()
            let out =
                config.splitGraph
                ? try models.runSplit(
                    chunk: chunk.features, chunkLength: chunk.length, state: state, config: config)
                : try models.run(
                    chunk: chunk.features, chunkLength: chunk.length, state: state, config: config)
            profile.inferenceSeconds += Date().timeIntervalSince(tStage)
            profile.inputPrepSeconds += out.inputPrepSeconds
            profile.predictSeconds += out.predictSeconds
            profile.readbackSeconds += out.readbackSeconds

            tStage = Date()
            let result = try updater.update(
                state: &state,
                chunkEmbeddings: out.chunkEmbeddings,
                chunkEncLength: out.chunkLength,
                predictions: out.predictions,
                highResPredictions: out.highResPredictions,
                lc: (chunk.leftOffset + sub / 2) / sub,
                rc: (chunk.rightOffset + sub - 1) / sub
            )
            profile.stateUpdateSeconds += Date().timeIntervalSince(tStage)

            tStage = Date()
            total.append(contentsOf: result.probabilities)
            profile.outputAppendSeconds += Date().timeIntervalSince(tStage)
            profile.chunkCount += 1
            return true
        }) {}

        // NeMo trims to ceil(mel_frames / output_subsampling_factor); output factor is 1 (10 ms).
        let outputFrames = min(featSeqLength, total.count / config.numSpeakers)
        profile.totalSeconds = Date().timeIntervalSince(t0)
        lastProfile = profile
        return (Array(total[0..<(outputFrames * config.numSpeakers)]), outputFrames)
    }

    /// Run one streaming step from raw mel features.
    ///
    /// - Parameters:
    ///   - chunkFeatures: Mel features [frames * 128] for lc+core+rc mel frames.
    ///   - chunkMelLength: Valid mel frames in `chunkFeatures`.
    ///   - leftOffsetMel: Mel frames of left context included at the start.
    ///   - rightOffsetMel: Mel frames of right context included at the end.
    public func step(
        chunkFeatures: [Float],
        chunkMelLength: Int,
        leftOffsetMel: Int,
        rightOffsetMel: Int
    ) throws -> Nemotron3ChunkResult {
        let out =
            config.splitGraph
            ? try models.runSplit(
                chunk: chunkFeatures, chunkLength: chunkMelLength, state: state, config: config)
            : try models.run(
                chunk: chunkFeatures, chunkLength: chunkMelLength, state: state, config: config)
        let sub = config.subsamplingFactor
        let lcEnc = (leftOffsetMel + sub / 2) / sub  // round()
        let rcEnc = (rightOffsetMel + sub - 1) / sub  // ceil()
        return try updater.update(
            state: &state,
            chunkEmbeddings: out.chunkEmbeddings,
            chunkEncLength: out.chunkLength,
            predictions: out.predictions,
            highResPredictions: out.highResPredictions,
            lc: lcEnc,
            rc: rcEnc
        )
    }

    /// Convert frame probabilities into arrival-ordered speaker segments.
    public static func segments(
        probabilities: [Float], frameCount: Int, numSpeakers: Int = 8,
        threshold: Float = 0.5, frameSeconds: Float = 0.01, minDurationSeconds: Float = 0.2
    ) -> [Nemotron3Segment] {
        var result: [Nemotron3Segment] = []
        for spk in 0..<numSpeakers {
            var start: Int? = nil
            for frame in 0...frameCount {
                let active = frame < frameCount && probabilities[frame * numSpeakers + spk] > threshold
                if active, start == nil {
                    start = frame
                } else if !active, let s0 = start {
                    let dur = Float(frame - s0) * frameSeconds
                    if dur >= minDurationSeconds {
                        result.append(
                            Nemotron3Segment(
                                speakerIndex: spk,
                                startSeconds: Float(s0) * frameSeconds,
                                endSeconds: Float(frame) * frameSeconds))
                    }
                    start = nil
                }
            }
        }
        return result.sorted { $0.startSeconds < $1.startSeconds }
    }
}

// MARK: - Feature Loader

/// Chunk iterator over a mel feature sequence, mirroring NeMo's `streaming_feat_loader`:
/// fixed core stride, left context of 0 (all preview profiles), right context shrinking at
/// the tail so trailing audio is still emitted.
public struct Nemotron3FeatureLoader {
    private let lcMel: Int
    private let rcMel: Int
    private let coreMel: Int
    private let melFeatures: Int
    private let capacityMel: Int

    private let featSeq: [Float]
    private let featLength: Int
    private let featSeqLength: Int

    private var startFeat = 0

    public init(config: Nemotron3Config, featSeq: [Float], featLength: Int, featSeqLength: Int) {
        self.lcMel = config.chunkLeftContext * config.subsamplingFactor
        self.rcMel = config.chunkRightContext * config.subsamplingFactor
        self.coreMel = config.chunkLen * config.subsamplingFactor
        self.melFeatures = config.melFeatures
        self.capacityMel = config.chunkMelFrames
        self.featSeq = featSeq
        self.featLength = featLength
        self.featSeqLength = featSeqLength
    }

    public mutating func next() -> (features: [Float], length: Int, leftOffset: Int, rightOffset: Int)? {
        guard startFeat < featLength else { return nil }
        let leftOffset = min(lcMel, startFeat)
        let endFeat = min(startFeat + coreMel, featLength)
        let rightOffset = min(rcMel, featLength - endFeat)

        let startIdx = (startFeat - leftOffset) * melFeatures
        let endIdx = (endFeat + rightOffset) * melFeatures
        var features = Array(featSeq[startIdx..<endIdx])
        // Zero-pad to the model's fixed mel capacity.
        if features.count < capacityMel * melFeatures {
            features.append(contentsOf: repeatElement(0, count: capacityMel * melFeatures - features.count))
        }
        let frames = endFeat + rightOffset - (startFeat - leftOffset)
        let length = max(min(featSeqLength - startFeat + leftOffset, frames), 0)

        startFeat = endFeat
        return (features, length, leftOffset, rightOffset)
    }
}
