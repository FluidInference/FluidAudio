@preconcurrency import CoreML
import Foundation

/// Word → speaker attribution from Nemotron-3-Diarization's per-frame activity.
///
/// The diarizer emits 10 ms probabilities for 8 speaker slots. They are averaged to the encoder's 80 ms
/// frames (a short last group is averaged over the frames it has), aligned to the encoder frame count
/// (repeating the last frame if short), slots with under 2 s of total activity are dropped, and each
/// word takes the most likely active speaker at its frame, searching up to ~1 s either side, else the
/// previous word's speaker.
enum NemotronTranscribeLiteSpeakers {

    /// `[frames10ms * speakers]` frame-major → `[frames80ms * speakers]`.
    static func downsample(_ probs: [Float], frames: Int, speakers: Int, factor: Int) -> [Float] {
        let groups = NemotronTranscribeLiteManager.ceilDiv(frames, factor)
        var out = [Float](repeating: 0, count: groups * speakers)
        for g in 0..<groups {
            let lo = g * factor
            let hi = min(lo + factor, frames)
            for s in 0..<speakers {
                var sum: Float = 0
                for f in lo..<hi { sum += probs[f * speakers + s] }
                out[g * speakers + s] = sum / Float(hi - lo)
            }
        }
        return out
    }

    /// Truncate or extend (repeating the last frame) to `frames`.
    static func align(_ probs: [Float], speakers: Int, to frames: Int) -> [Float] {
        let have = probs.count / speakers
        guard have > 0 else { return [Float](repeating: 0, count: frames * speakers) }
        if have >= frames { return Array(probs[0..<(frames * speakers)]) }
        var out = probs
        let last = Array(probs[((have - 1) * speakers)..<(have * speakers)])
        for _ in have..<frames { out += last }
        return out
    }

    /// Zero the slots active (p > threshold) in fewer than `minFrames` frames.
    static func dropShortSpeakers(_ probs: [Float], speakers: Int, threshold: Float, minFrames: Int) -> [Float] {
        let frames = probs.count / speakers
        var active = [Int](repeating: 0, count: speakers)
        for f in 0..<frames {
            for s in 0..<speakers where probs[f * speakers + s] > threshold { active[s] += 1 }
        }
        var out = probs
        for s in 0..<speakers where active[s] < minFrames {
            for f in 0..<frames { out[f * speakers + s] = 0 }
        }
        return out
    }

    /// Speaker slot per word start time.
    static func assign(
        starts: [Double], probs: [Float], speakers: Int, threshold: Float, searchFrames: Int, frameSeconds: Double
    ) -> [Int] {
        let frames = probs.count / speakers
        guard frames > 0 else { return starts.map { _ in 0 } }
        var result: [Int] = []
        var last: Int?
        for t in starts {
            let f = min(Int(t / frameSeconds), frames - 1)
            var candidate: Int?
            search: for d in 0...searchFrames {
                for g in [f - d, f + d] where g >= 0 && g < frames {
                    var best = -1
                    var bestValue: Float = 0
                    for s in 0..<speakers {
                        let p = probs[g * speakers + s]
                        if p > threshold && (best < 0 || p > bestValue) {
                            best = s
                            bestValue = p
                        }
                    }
                    if best >= 0 {
                        candidate = best
                        break search
                    }
                }
            }
            last = candidate ?? last ?? 0
            result.append(last ?? 0)
        }
        return result
    }

    /// Diarizer probabilities (10 ms, frame-major) → one speaker slot per word.
    static func speakers(
        for words: [NemotronTranscribeLiteWord], probabilities: [Float], diarizerFrames: Int,
        encoderFrames: Int, speakers: Int = 8
    ) -> [Int] {
        let c = NemotronTranscribeLiteConfig.self
        let p80 = downsample(probabilities, frames: diarizerFrames, speakers: speakers, factor: c.stack)
        let aligned = align(p80, speakers: speakers, to: encoderFrames)
        let kept = dropShortSpeakers(
            aligned, speakers: speakers, threshold: c.speakerActivityThreshold, minFrames: c.minSpeakerFrames)
        return assign(
            starts: words.map(\.start), probs: kept, speakers: speakers, threshold: c.speakerActivityThreshold,
            searchFrames: c.speakerSearchFrames, frameSeconds: c.encoderFrameSeconds)
    }
}

#if Diarizer
extension NemotronTranscribeLiteManager {

    /// Transcribe with speakers: Nemotron-3-Diarization runs concurrently (by default on the Neural
    /// Engine) while the encoder and CTC head run, then each word gets the speaker active at its start.
    ///
    /// The diarizer is loaded inside the call (from the model cache), as its own task. The first
    /// Neural Engine load compiles it, which can take over a minute; run once on a short clip to warm up.
    public func transcribeWithSpeakers(
        audio: [Float], diarizerComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> NemotronTranscribeLiteResult {
        let t0 = Date()
        async let diarization = Self.diarize(audio: audio, computeUnits: diarizerComputeUnits)
        let asr = try transcribe(audio: audio)
        let (probabilities, diarizerFrames, diarizerSeconds) = try await diarization

        let encoderFrames = Self.ceilDiv(
            audio.count / NemotronTranscribeLiteConfig.hopLength, NemotronTranscribeLiteConfig.stack)
        let slots = NemotronTranscribeLiteSpeakers.speakers(
            for: asr.words, probabilities: probabilities, diarizerFrames: diarizerFrames,
            encoderFrames: encoderFrames)
        let words = zip(asr.words, slots).map {
            NemotronTranscribeLiteWord(text: $0.text, start: $0.start, speaker: $1)
        }
        return NemotronTranscribeLiteResult(
            words: words, durationSeconds: asr.durationSeconds,
            timings: .init(
                encoderSeconds: asr.timings.encoderSeconds, ctcSeconds: asr.timings.ctcSeconds,
                diarizerSeconds: diarizerSeconds, totalSeconds: Date().timeIntervalSince(t0)))
    }

    /// Load Nemotron-3-Diarization (offline preset) and run it over the whole recording.
    private static func diarize(
        audio: [Float], computeUnits: MLComputeUnits
    ) async throws -> ([Float], Int, Double) {
        let t0 = Date()
        let config = Nemotron3Config.offline
        let models = try await Nemotron3Models.loadFromHuggingFace(config: config, computeUnits: computeUnits)
        let (probabilities, frames) = try Nemotron3Diarizer(config: config, models: models).processComplete(audio)
        return (probabilities, frames, Date().timeIntervalSince(t0))
    }
}
#endif
