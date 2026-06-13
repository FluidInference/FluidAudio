import Testing

@testable import FluidAudio

/// Tests for the pure window/frame bookkeeping behind Parakeet Unified:
/// the streaming `UnifiedStreamingWindower` and the offline `UnifiedBatchLayout`.
/// Mirrors the Python reference loop validated against NeMo in mobius
/// models/stt/parakeet-unified-en-0.6b/coreml.
struct UnifiedWindowingTests {

    private let config = UnifiedConfig()  // [70, 13, 13], 1280 samples/frame

    /// Encoder frames for a given valid sample count (ceil(samples/1280) capped at window).
    private func encoderLength(forBufferSamples samples: Int) -> Int {
        min((samples + config.frameSamples - 1) / config.frameSamples, 96)
    }

    @Test
    func firstWindowRequiresChunkPlusRightContext() {
        var windower = UnifiedStreamingWindower(config: config)
        // chunk+right = 26 frames = 33280 samples of initial latency
        #expect(windower.nextWindow(totalSamples: 33279, isFinal: false) == nil)
        let plan = windower.nextWindow(totalSamples: 33280, isFinal: false)
        #expect(plan != nil)
        #expect(plan?.bufferStart == 0)
        #expect(plan?.bufferEnd == 33280)
        #expect(plan?.holdbackFrames == config.rightFrames)
    }

    @Test
    func firstWindowDecodesOnlyChunkFrames() {
        var windower = UnifiedStreamingWindower(config: config)
        let plan = windower.nextWindow(totalSamples: 33280, isFinal: false)!
        // 26 valid frames, right context (13) held back → decode chunk frames 0..<13
        let range = windower.decodeRange(encoderLength: 26, plan: plan)
        #expect(range == 0..<13)
        #expect(windower.decodedFrames == 13)
    }

    @Test
    func steadyStateAdvancesByOneChunkPerStep() {
        var windower = UnifiedStreamingWindower(config: config)
        let total = 16 * config.chunkSamples + config.rightSamples

        var decodedRanges: [Range<Int>] = []
        while let plan = windower.nextWindow(totalSamples: total, isFinal: false) {
            let bufferSamples = plan.bufferEnd - plan.bufferStart
            #expect(bufferSamples <= config.windowSamples)
            let encLen = encoderLength(forBufferSamples: bufferSamples)
            if let range = windower.decodeRange(encoderLength: encLen, plan: plan) {
                #expect(range.count == config.chunkFrames)
                decodedRanges.append(range)
            }
        }
        // 16 chunks fed; the final right context stays undecoded until finish.
        #expect(windower.decodedFrames == 16 * config.chunkFrames)
        #expect(decodedRanges.count == 16)
        // Global decode positions are contiguous: each step continues where the
        // previous one ended, independent of how the window slid.
        #expect(windower.consumedSamples == total)
    }

    @Test
    func finalFlushDecodesHeldBackRightContext() {
        var windower = UnifiedStreamingWindower(config: config)
        let total = 4 * config.chunkSamples + config.rightSamples

        while let plan = windower.nextWindow(totalSamples: total, isFinal: false) {
            let encLen = encoderLength(forBufferSamples: plan.bufferEnd - plan.bufferStart)
            _ = windower.decodeRange(encoderLength: encLen, plan: plan)
        }
        #expect(windower.decodedFrames == 4 * config.chunkFrames)

        // Final flush: the leftover right context becomes decodable (holdback 0).
        let plan = windower.nextWindow(totalSamples: total, isFinal: true)
        #expect(plan != nil)
        #expect(plan?.holdbackFrames == 0)
        let encLen = encoderLength(forBufferSamples: plan!.bufferEnd - plan!.bufferStart)
        let range = windower.decodeRange(encoderLength: encLen, plan: plan!)
        #expect(range?.count == config.rightFrames)
        #expect(windower.decodedFrames == 4 * config.chunkFrames + config.rightFrames)
        // Nothing left afterwards.
        #expect(windower.nextWindow(totalSamples: total, isFinal: true) == nil)
    }

    @Test
    func unalignedFinalBufferNeverExceedsWindow() {
        // Regression: with a total that is not a multiple of the frame size,
        // the last buffer start must round UP to a frame boundary, otherwise
        // the buffer exceeds the fixed encoder window.
        var windower = UnifiedStreamingWindower(config: config)
        let total = 123_440  // > windowSamples (122880), not frame-aligned

        var sawFinal = false
        while let plan = windower.nextWindow(totalSamples: total, isFinal: true) {
            let bufferSamples = plan.bufferEnd - plan.bufferStart
            #expect(bufferSamples <= config.windowSamples)
            #expect(plan.bufferStart % config.frameSamples == 0)
            let encLen = encoderLength(forBufferSamples: bufferSamples)
            _ = windower.decodeRange(encoderLength: encLen, plan: plan)
            if plan.bufferEnd == total { sawFinal = true }
        }
        #expect(sawFinal)
        #expect(windower.consumedSamples == total)
    }

    @Test
    func shortFinalOnlyAudioIsFlushedInOneWindow() {
        // Audio shorter than chunk+right: nothing during streaming, one final window.
        var windower = UnifiedStreamingWindower(config: config)
        let total = 20_000

        #expect(windower.nextWindow(totalSamples: total, isFinal: false) == nil)
        let plan = windower.nextWindow(totalSamples: total, isFinal: true)
        #expect(plan?.bufferStart == 0)
        #expect(plan?.bufferEnd == total)
        #expect(plan?.holdbackFrames == 0)
        let encLen = encoderLength(forBufferSamples: total)
        let range = windower.decodeRange(encoderLength: encLen, plan: plan!)
        #expect(range == 0..<encLen)
    }

    @Test
    func finalFlushEmitsAtMostOnceEvenIfDecodeFallsShort() {
        // Regression: termination must not depend on re-deriving the encoder's
        // exact length formula. If the final window's reported encoder length
        // yields fewer frames than a ceil(samples/frame) estimate, nextWindow
        // must still return nil afterwards instead of looping forever.
        var windower = UnifiedStreamingWindower(config: config)
        let total = 960_006  // ceil(total/1280) = 751, but encoder yields 750

        var plans = 0
        while let plan = windower.nextWindow(totalSamples: total, isFinal: true) {
            plans += 1
            // Simulate the model reporting one frame fewer than the estimate.
            let bufferSamples = plan.bufferEnd - plan.bufferStart
            let encLen = min(bufferSamples / config.frameSamples, 96)
            _ = windower.decodeRange(encoderLength: encLen, plan: plan)
            #expect(plans < 100, "final flush loops forever")
            if plans >= 100 { break }
        }
        #expect(windower.nextWindow(totalSamples: total, isFinal: true) == nil)
    }

    @Test
    func resetClearsProgress() {
        var windower = UnifiedStreamingWindower(config: config)
        let plan = windower.nextWindow(totalSamples: 50_000, isFinal: false)!
        _ = windower.decodeRange(encoderLength: 26, plan: plan)
        #expect(windower.consumedSamples > 0)

        windower.reset()
        #expect(windower.consumedSamples == 0)
        #expect(windower.decodedFrames == 0)
        // Behaves like a fresh stream again.
        #expect(windower.nextWindow(totalSamples: 33_279, isFinal: false) == nil)
    }

    @Test
    func batchLayoutQuantities() {
        let layout = UnifiedBatchLayout(config: config)
        #expect(layout.windowSamples == 240_000)  // 15 s
        #expect(layout.chunkSamples % config.frameSamples == 0)
        #expect(layout.chunkSamples <= layout.windowSamples)
        #expect(layout.overlapSamples == 32_000)  // 2 s, frame-aligned (25 frames)
        #expect(layout.strideSamples == layout.chunkSamples - layout.overlapSamples)
    }

    @Test
    func batchChunkStartsCoverAllAudio() {
        let layout = UnifiedBatchLayout(config: config)

        // Short audio: single window.
        #expect(layout.chunkStarts(totalSamples: 100_000) == [0])
        #expect(layout.chunkStarts(totalSamples: 0) == [])

        // Long audio: frame-aligned starts, full coverage, 2 s overlap.
        let total = 60 * 16_000
        let starts = layout.chunkStarts(totalSamples: total)
        #expect(starts.first == 0)
        for start in starts {
            #expect(start % config.frameSamples == 0)
        }
        for (a, b) in zip(starts, starts.dropFirst()) {
            #expect(b - a == layout.strideSamples)
            // Adjacent windows overlap.
            #expect(a + layout.chunkSamples > b)
        }
        // The last window reaches the end of the audio.
        #expect(starts.last! + layout.chunkSamples >= total)
        // No window is fully contained in the previous one (each adds samples).
        for (a, b) in zip(starts, starts.dropFirst()) {
            #expect(b + layout.overlapSamples < total || b == starts.last!)
            _ = a
        }
    }

    @Test
    func configDerivedQuantities() {
        #expect(config.windowSamples == 122_880)  // 96 frames × 1280
        #expect(config.chunkSamples == 16_640)
        #expect(config.latencyMs == 2080)
        #expect(config.contextSuffix == "70_13_13")
    }
}
