import Foundation

private let lseendLogConversionFactor = Float(1.0 / Foundation.log(10.0))

public struct LSEENDFeatureConfig: Sendable, Hashable {
    public let sampleRate: Int
    public let winLength: Int
    public let hopLength: Int
    public let nFFT: Int
    public let nMels: Int
    public let contextRecp: Int
    public let subsampling: Int
    public let inputDim: Int

    public init(metadata: LSEENDModelMetadata) {
        sampleRate = metadata.resolvedSampleRate
        winLength = metadata.resolvedWinLength
        hopLength = metadata.resolvedHopLength
        nFFT = metadata.resolvedFFTSize
        nMels = metadata.resolvedMelCount
        contextRecp = metadata.resolvedContextRecp
        subsampling = metadata.resolvedSubsampling
        inputDim = metadata.inputDim
    }

    public var stableBlockSize: Int {
        hopLength * subsampling
    }
}

struct LSEENDMelSpectrogramKey: Hashable {
    let sampleRate: Int
    let nMels: Int
    let nFFT: Int
    let hopLength: Int
    let winLength: Int
    let preemphBits: UInt32
    let logFloorBits: UInt32
    let logFloorModeRawValue: Int
    let windowPeriodic: Bool
    let padTo: Int

    init(
        sampleRate: Int,
        nMels: Int,
        nFFT: Int,
        hopLength: Int,
        winLength: Int,
        preemph: Float,
        logFloor: Float,
        logFloorMode: NeMoMelSpectrogram.LogFloorMode,
        windowPeriodic: Bool,
        padTo: Int
    ) {
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        preemphBits = preemph.bitPattern
        logFloorBits = logFloor.bitPattern
        logFloorModeRawValue = logFloorMode == .additive ? 0 : 1
        self.windowPeriodic = windowPeriodic
        self.padTo = padTo
    }
}

final class LSEENDMelSpectrogramStore {
    static let shared = LSEENDMelSpectrogramStore()

    private let lock = NSLock()
    private var cache: [LSEENDMelSpectrogramKey: NeMoMelSpectrogram] = [:]

    private init() {}

    func spectrogram(for config: LSEENDFeatureConfig) -> NeMoMelSpectrogram {
        let key = LSEENDMelSpectrogramKey(
            sampleRate: config.sampleRate,
            nMels: config.nMels,
            nFFT: config.nFFT,
            hopLength: config.hopLength,
            winLength: config.winLength,
            preemph: 0,
            logFloor: 1e-10,
            logFloorMode: .clamped,
            windowPeriodic: true,
            padTo: 1
        )

        lock.lock()
        defer { lock.unlock() }

        if let cached = cache[key] {
            return cached
        }

        let created = NeMoMelSpectrogram(
            sampleRate: config.sampleRate,
            nMels: config.nMels,
            nFFT: config.nFFT,
            hopLength: config.hopLength,
            winLength: config.winLength,
            preemph: 0,
            padTo: 1,
            logFloor: 1e-10,
            logFloorMode: .clamped,
            windowPeriodic: true
        )
        cache[key] = created
        return created
    }
}

final class LSEENDOfflineFeatureExtractorStore {
    static let shared = LSEENDOfflineFeatureExtractorStore()

    private let lock = NSLock()
    private var cache: [LSEENDFeatureConfig: LSEENDOfflineFeatureExtractor] = [:]

    private init() {}

    func extractor(for metadata: LSEENDModelMetadata, spectrogram: NeMoMelSpectrogram) -> LSEENDOfflineFeatureExtractor {
        let config = LSEENDFeatureConfig(metadata: metadata)

        lock.lock()
        defer { lock.unlock() }

        if let cached = cache[config] {
            return cached
        }

        let created = LSEENDOfflineFeatureExtractor(metadata: metadata, spectrogram: spectrogram)
        cache[config] = created
        return created
    }
}

public final class LSEENDOfflineFeatureExtractor {
    private let config: LSEENDFeatureConfig
    private let spectrogram: NeMoMelSpectrogram

    public init(metadata: LSEENDModelMetadata, spectrogram: NeMoMelSpectrogram? = nil) {
        let featureConfig = LSEENDFeatureConfig(metadata: metadata)
        config = featureConfig
        self.spectrogram = spectrogram ?? LSEENDMelSpectrogramStore.shared.spectrogram(for: featureConfig)
    }

    public func extractFeatures(audio: [Float]) throws -> LSEENDMatrix {
        let usableSamples = (audio.count / config.stableBlockSize) * config.stableBlockSize
        guard usableSamples > 0 else {
            return .empty(columns: config.inputDim)
        }
        let trimmedAudio = Array(audio.prefix(usableSamples))
        let stftFrameCount = max(0, usableSamples / config.hopLength - 1)
        guard stftFrameCount > 0 else {
            return .empty(columns: config.inputDim)
        }
        let mel = spectrogram.computeFlatTransposed(
            audio: trimmedAudio,
            lastAudioSample: 0,
            paddingMode: .center,
            expectedFrameCount: stftFrameCount
        ).mel
        let normalized = Self.applyLogMelCumMeanNormalization(
            mel,
            rowCount: stftFrameCount,
            nMels: config.nMels,
            frameStart: 0,
            cumulativeFeatureSum: nil
        )
        let base = LSEENDMatrix(validatingRows: stftFrameCount, columns: config.nMels, values: normalized.values)
        return Self.spliceAndSubsample(
            baseFeatures: base,
            contextSize: config.contextRecp,
            subsampling: config.subsampling
        )
    }

    fileprivate static func applyLogMelCumMeanNormalization(
        _ mel: [Float],
        rowCount: Int,
        nMels: Int,
        frameStart: Int,
        cumulativeFeatureSum: [Double]?
    ) -> (values: [Float], cumulativeFeatureSum: [Double]) {
        var cumulative = cumulativeFeatureSum ?? [Double](repeating: 0, count: nMels)
        var output = [Float](repeating: 0, count: mel.count)
        for rowIndex in 0..<rowCount {
            let count = Double(frameStart + rowIndex + 1)
            let rowOffset = rowIndex * nMels
            for melIndex in 0..<nMels {
                let log10Mel = mel[rowOffset + melIndex] * lseendLogConversionFactor
                cumulative[melIndex] += Double(log10Mel)
                output[rowOffset + melIndex] = log10Mel - Float(cumulative[melIndex] / count)
            }
        }
        return (output, cumulative)
    }

    fileprivate static func spliceAndSubsample(
        baseFeatures: LSEENDMatrix,
        contextSize: Int,
        subsampling: Int
    ) -> LSEENDMatrix {
        guard baseFeatures.rows > 0 else {
            return .empty(columns: baseFeatures.columns * ((2 * contextSize) + 1))
        }
        let outputRows = (baseFeatures.rows + subsampling - 1) / subsampling
        let outputColumns = baseFeatures.columns * ((2 * contextSize) + 1)
        var output = [Float](repeating: 0, count: outputRows * outputColumns)
        for outputRow in 0..<outputRows {
            let center = outputRow * subsampling
            let destinationRowOffset = outputRow * outputColumns
            for contextOffset in -contextSize...contextSize {
                let sourceRow = center + contextOffset
                guard sourceRow >= 0, sourceRow < baseFeatures.rows else {
                    continue
                }
                let destinationOffset = destinationRowOffset + (contextOffset + contextSize) * baseFeatures.columns
                let sourceOffset = sourceRow * baseFeatures.columns
                for featureIndex in 0..<baseFeatures.columns {
                    output[destinationOffset + featureIndex] = baseFeatures.values[sourceOffset + featureIndex]
                }
            }
        }
        return LSEENDMatrix(validatingRows: outputRows, columns: outputColumns, values: output)
    }
}

public final class LSEENDStreamingFeatureExtractor {
    private let config: LSEENDFeatureConfig
    private let spectrogram: NeMoMelSpectrogram

    private var audioBuffer: [Float] = []
    private var audioStartSample = 0
    private var totalSamples = 0

    private var nextSTFTFrame = 0
    private var nextModelFrame = 0

    private var baseFeatureStart = 0
    private var baseFeatureBuffer: [Float] = []
    private var baseFeatureRows = 0
    private var cumulativeFeatureSum: [Double]

    public init(metadata: LSEENDModelMetadata, spectrogram: NeMoMelSpectrogram? = nil) {
        let featureConfig = LSEENDFeatureConfig(metadata: metadata)
        config = featureConfig
        self.spectrogram = spectrogram ?? LSEENDMelSpectrogramStore.shared.spectrogram(for: featureConfig)
        cumulativeFeatureSum = [Double](repeating: 0, count: featureConfig.nMels)
    }

    public func pushAudio(_ chunk: [Float]) throws -> LSEENDMatrix {
        guard !chunk.isEmpty else {
            return .empty(columns: config.inputDim)
        }
        audioBuffer.append(contentsOf: chunk)
        totalSamples += chunk.count
        try appendSTFTFrames(targetFrameCount: stableSTFTFrameCount(), allowRightPad: false, effectiveTotalSamples: nil)
        return try emitModelFrames(final: false, totalSTFTFrames: nil)
    }

    public func finalize() throws -> LSEENDMatrix {
        let usableSamples = usableSampleCount(totalSamples)
        let totalSTFTFrames = offlineSTFTFrameCount(usableSamples)
        try appendSTFTFrames(
            targetFrameCount: totalSTFTFrames,
            allowRightPad: true,
            effectiveTotalSamples: usableSamples
        )
        return try emitModelFrames(final: true, totalSTFTFrames: totalSTFTFrames)
    }

    private func usableSampleCount(_ sampleCount: Int) -> Int {
        (sampleCount / config.stableBlockSize) * config.stableBlockSize
    }

    private func stableSTFTFrameCount() -> Int {
        let leftPad = config.nFFT / 2
        guard totalSamples > leftPad else {
            return 0
        }
        return max(0, ((totalSamples - leftPad) / config.hopLength) + 1)
    }

    private func offlineSTFTFrameCount(_ usableSamples: Int) -> Int {
        guard usableSamples > 0 else {
            return 0
        }
        return max(0, usableSamples / config.hopLength - 1)
    }

    private func totalModelFrameCount(_ totalSTFTFrames: Int) -> Int {
        guard totalSTFTFrames > 0 else {
            return 0
        }
        return (totalSTFTFrames + config.subsampling - 1) / config.subsampling
    }

    private func appendSTFTFrames(
        targetFrameCount: Int,
        allowRightPad: Bool,
        effectiveTotalSamples: Int?
    ) throws {
        guard targetFrameCount > nextSTFTFrame else {
            return
        }
        let frameStart = nextSTFTFrame
        let frameStop = targetFrameCount
        let expectedFrames = frameStop - frameStart
        let segment = try stftSegment(
            frameStart: frameStart,
            frameStop: frameStop,
            allowRightPad: allowRightPad,
            effectiveTotalSamples: effectiveTotalSamples
        )
        let mel = spectrogram.computeFlatTransposed(
            audio: segment,
            lastAudioSample: 0,
            paddingMode: .prePadded,
            expectedFrameCount: expectedFrames
        ).mel
        let normalized = LSEENDOfflineFeatureExtractor.applyLogMelCumMeanNormalization(
            mel,
            rowCount: expectedFrames,
            nMels: config.nMels,
            frameStart: frameStart,
            cumulativeFeatureSum: cumulativeFeatureSum
        )
        cumulativeFeatureSum = normalized.cumulativeFeatureSum
        baseFeatureBuffer.append(contentsOf: normalized.values)
        baseFeatureRows += expectedFrames
        nextSTFTFrame = frameStop
        dropConsumedAudio()
    }

    private func stftSegment(
        frameStart: Int,
        frameStop: Int,
        allowRightPad: Bool,
        effectiveTotalSamples: Int?
    ) throws -> [Float] {
        guard frameStop > frameStart else {
            return []
        }
        let leftPad = config.nFFT / 2
        let total = effectiveTotalSamples ?? totalSamples
        let globalStart = frameStart * config.hopLength - leftPad
        let globalStop = (frameStop - 1) * config.hopLength - leftPad + config.nFFT

        let prefixCount = max(0, -globalStart)
        let suffixCount = allowRightPad ? max(0, globalStop - total) : 0
        let rawStart = max(0, globalStart)
        let rawStop = min(total, globalStop)
        guard rawStart >= audioStartSample else {
            throw LSEENDError.unsupportedAudio(
                "Audio buffer underflow. Need sample \(rawStart) but buffer starts at \(audioStartSample)."
            )
        }
        let localStart = rawStart - audioStartSample
        let localStop = rawStop - audioStartSample
        var segment = [Float](repeating: 0, count: prefixCount + (localStop - localStart) + suffixCount)
        let coreCount = max(0, localStop - localStart)
        if coreCount > 0 {
            for index in 0..<coreCount {
                segment[prefixCount + index] = audioBuffer[localStart + index]
            }
        }
        return segment
    }

    private func emitModelFrames(final: Bool, totalSTFTFrames: Int?) throws -> LSEENDMatrix {
        var output = [Float]()
        let latestFrame = nextSTFTFrame - 1
        let totalModelFrames = final ? self.totalModelFrameCount(totalSTFTFrames ?? 0) : nil

        while true {
            let centerIndex = nextModelFrame * config.subsampling
            let maxIndex: Int
            if final {
                guard let totalModelFrames, let totalSTFTFrames else { break }
                if nextModelFrame >= totalModelFrames {
                    break
                }
                maxIndex = totalSTFTFrames - 1
            } else {
                if centerIndex + config.contextRecp > latestFrame {
                    break
                }
                maxIndex = latestFrame
            }
            output.append(contentsOf: try spliceFrame(centerIndex: centerIndex, maxIndex: maxIndex))
            nextModelFrame += 1
            dropConsumedBaseFeatures()
        }

        let outputRows = output.isEmpty ? 0 : output.count / config.inputDim
        return LSEENDMatrix(validatingRows: outputRows, columns: config.inputDim, values: output)
    }

    private func spliceFrame(centerIndex: Int, maxIndex: Int) throws -> [Float] {
        var frame = [Float](repeating: 0, count: config.inputDim)
        for frameIndex in (centerIndex - config.contextRecp)...(centerIndex + config.contextRecp) {
            let destinationBase = (frameIndex - (centerIndex - config.contextRecp)) * config.nMels
            guard frameIndex >= 0, frameIndex <= maxIndex else {
                continue
            }
            let localIndex = frameIndex - baseFeatureStart
            guard localIndex >= 0, localIndex < baseFeatureRows else {
                throw LSEENDError.unsupportedAudio(
                    "Feature buffer underflow. Need frame \(frameIndex), buffer covers [\(baseFeatureStart), \(baseFeatureStart + baseFeatureRows - 1)]."
                )
            }
            let sourceBase = localIndex * config.nMels
            for melIndex in 0..<config.nMels {
                frame[destinationBase + melIndex] = baseFeatureBuffer[sourceBase + melIndex]
            }
        }
        return frame
    }

    private func dropConsumedAudio() {
        let leftPad = config.nFFT / 2
        let keepFrom = max(0, nextSTFTFrame * config.hopLength - leftPad)
        let dropCount = keepFrom - audioStartSample
        guard dropCount > 0 else {
            return
        }
        audioBuffer.removeFirst(dropCount)
        audioStartSample += dropCount
    }

    private func dropConsumedBaseFeatures() {
        let keepFrom = max(0, nextModelFrame * config.subsampling - config.contextRecp)
        let dropRows = keepFrom - baseFeatureStart
        guard dropRows > 0 else {
            return
        }
        let dropCount = dropRows * config.nMels
        baseFeatureBuffer.removeFirst(dropCount)
        baseFeatureRows -= dropRows
        baseFeatureStart += dropRows
    }
}
