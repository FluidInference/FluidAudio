import Foundation

/// Clean single-speaker region selection for the TitaNet `.cleanRegion` pooling
/// path (PIPELINE_SPEC §3.2). Operates purely on the 100 fps mel-frame grid, so
/// it is unit-testable without any CoreML model.
///
/// Given a per-mel-frame clean weight for ONE speaker (overlap already excluded
/// upstream), it returns the ordered mel-frame indices to pack contiguously into
/// the encoder input — or `nil` to skip the speaker in this window.
enum CleanRegionSelector {

    /// - Parameters:
    ///   - cleanMaskMel: per-mel-frame clean weight (~0..1), length = mel frames.
    ///   - collarLead / collarTrail: frames eroded inward from each contiguous
    ///     clean run's start / end (co-articulation, reverb tail, interferer onset).
    ///   - regionMin: minimum run length (frames) AFTER erosion to keep it.
    ///   - embedMin: skip the speaker (return `nil`) if fewer than this many clean
    ///     frames survive.
    ///   - embedTarget: accumulate clean frames in time order up to this many. A
    ///     single contiguous kept run ≥ target is embedded IN FULL (maximizes
    ///     clean seconds, avoids a splice), per spec §3.4/§5.
    /// - Returns: ascending mel-frame indices to pack, or `nil` to skip.
    static func selectCleanMelFrames(
        cleanMaskMel: [Float],
        collarLead: Int,
        collarTrail: Int,
        regionMin: Int,
        embedMin: Int,
        embedTarget: Int
    ) -> [Int]? {
        let n = cleanMaskMel.count
        guard n > 0 else { return nil }

        // 1. Binarize (>0.5) → contiguous clean runs [start, end).
        var runs: [(start: Int, end: Int)] = []
        var i = 0
        while i < n {
            if cleanMaskMel[i] > 0.5 {
                var j = i + 1
                while j < n, cleanMaskMel[j] > 0.5 { j += 1 }
                runs.append((i, j))
                i = j
            } else {
                i += 1
            }
        }

        // 2. Collar-erode inward; keep runs still ≥ regionMin.
        let minRun = max(1, regionMin)
        var kept: [(start: Int, end: Int)] = []
        for run in runs {
            let s = run.start + max(0, collarLead)
            let e = run.end - max(0, collarTrail)
            if e - s >= minRun { kept.append((start: s, end: e)) }
        }
        guard !kept.isEmpty else { return nil }

        // 3a. A single contiguous kept run ≥ target → embed it in full.
        if let longest = kept.max(by: { ($0.end - $0.start) < ($1.end - $1.start) }),
            longest.end - longest.start >= embedTarget
        {
            return Array(longest.start..<longest.end)  // ≥ embedTarget ≥ embedMin
        }

        // 3b. Otherwise concatenate kept runs in time order up to the target.
        var frames: [Int] = []
        frames.reserveCapacity(min(embedTarget, n))
        outer: for run in kept {
            for f in run.start..<run.end {
                frames.append(f)
                if frames.count >= embedTarget { break outer }
            }
        }

        // 4. Skip if too little clean speech accumulated.
        return frames.count >= embedMin ? frames : nil
    }

    /// Clean single-speaker WAVEFORM regions for the `.cleanWaveform` pooling path
    /// (the reference-faithful fix). Unlike `selectCleanMelFrames` this works at
    /// SAMPLE resolution and returns sample ranges to crop from the window audio,
    /// so each crop can be re-featurized on its own. Mirrors the Python reference
    /// `extract_clean_region_waveforms` / `build_wavecrop.clean_crops`.
    ///
    /// - Parameters:
    ///   - weights: per-frame speaker posteriors `[frameCount][speakerCount]`.
    ///   - speaker: the speaker column to extract.
    ///   - availableSamples: window audio length (≤ 160k; the crop source).
    ///   - regionMinSamples / targetSamples / embedMinSamples: min kept run,
    ///     accumulation target, and skip floor — all in samples.
    ///   - threshold: hard active/overlap threshold (reference uses 0.5, NOT the
    ///     soft 1e-3 overlap threshold of the masked path).
    /// - Returns: `(regions, firstFrame, lastFrame)` where regions are ascending,
    ///   non-overlapping sample ranges `[start, end)`, and first/last frame index
    ///   the extractor maps to segment timing — or `nil` to skip the speaker.
    static func selectCleanSampleRegions(
        weights: [[Float]],
        speaker: Int,
        availableSamples: Int,
        regionMinSamples: Int,
        targetSamples: Int,
        embedMinSamples: Int,
        threshold: Float
    ) -> (regions: [(start: Int, end: Int)], firstFrame: Int, lastFrame: Int)? {
        let frameCount = weights.count
        guard frameCount > 0, availableSamples > 0,
            let speakerCount = weights.first?.count, speaker < speakerCount
        else { return nil }

        // clean[f] = speaker active AND no overlap (≥2 speakers over threshold).
        var clean = [Bool](repeating: false, count: frameCount)
        var firstFrame = -1
        var lastFrame = -1
        for f in 0..<frameCount {
            let row = weights[f]
            guard row[speaker] > threshold else { continue }
            var active = 0
            for value in row where value > threshold {
                active += 1
                if active > 1 { break }
            }
            if active < 2 {
                clean[f] = true
                if firstFrame < 0 { firstFrame = f }
                lastFrame = f
            }
        }
        guard firstFrame >= 0 else { return nil }

        let spf = Double(availableSamples) / Double(frameCount)
        var regions: [(start: Int, end: Int)] = []
        var total = 0
        var f = 0
        outer: while f < frameCount {
            if clean[f] {
                var j = f + 1
                while j < frameCount, clean[j] { j += 1 }
                let s0 = min(availableSamples, max(0, Int((Double(f) * spf).rounded())))
                let s1 = min(availableSamples, max(0, Int((Double(j) * spf).rounded())))
                if s1 - s0 >= regionMinSamples {
                    regions.append((start: s0, end: s1))
                    total += s1 - s0
                    if total >= targetSamples { break outer }
                }
                f = j
            } else {
                f += 1
            }
        }

        if total < embedMinSamples { return nil }
        return (regions, firstFrame, lastFrame)
    }

    /// Concatenates per-crop mel-major `[melBins, frames_i]` buffers in the feature
    /// (frame) domain into a fresh `[melBins, dstFrames]` buffer, zero-padded past
    /// the total. Returns the packed buffer and the number of frames actually
    /// written (`min(sum frames_i, dstFrames)`), which the mask spans. Pure/testable.
    static func packConcatFeatures(
        columns: [(feats: [Float], frames: Int)], melBins: Int, dstFrames: Int
    ) -> (feats: [Float], usedFrames: Int) {
        var dst = [Float](repeating: 0, count: melBins * dstFrames)
        let total = columns.reduce(0) { $0 + $1.frames }
        let used = min(total, dstFrames)
        for m in 0..<melBins {
            var pos = 0
            let dstRow = m * dstFrames
            for col in columns {
                if pos >= used { break }
                let srcRow = m * col.frames
                var t = 0
                while t < col.frames, pos < used {
                    dst[dstRow + pos] = col.feats[srcRow + t]
                    t += 1
                    pos += 1
                }
            }
        }
        return (dst, used)
    }

    /// Packs the `frames` columns of a C-contiguous row-major `[melBins, srcFrames]`
    /// buffer into a fresh `[melBins, dstFrames]` buffer, zero-padded past the
    /// selected count. Pure/testable core of the extractor's mel-frame packer.
    static func packColumns(
        src: [Float], melBins: Int, srcFrames: Int, dstFrames: Int, frames: [Int]
    ) -> [Float] {
        var dst = [Float](repeating: 0, count: melBins * dstFrames)
        let count = min(frames.count, dstFrames)
        for c in 0..<melBins {
            let srcRow = c * srcFrames
            let dstRow = c * dstFrames
            for p in 0..<count { dst[dstRow + p] = src[srcRow + frames[p]] }
        }
        return dst
    }
}
