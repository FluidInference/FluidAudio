import CoreML
import Foundation

internal struct TdtDecoderV2 {

    private let config: ASRConfig

    init(config: ASRConfig) {
        self.config = Self.adaptConfigForV2(config)
    }

    func decodeWithTimings(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        actualAudioFrames: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0
    ) async throws -> TdtHypothesis {
        let decoder = TdtDecoderV3(config: config)
        return try await decoder.decodeWithTimings(
            encoderOutput: encoderOutput,
            encoderSequenceLength: encoderSequenceLength,
            actualAudioFrames: actualAudioFrames,
            decoderModel: decoderModel,
            jointModel: jointModel,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )
    }

    func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        decoderState: inout TdtDecoderState
    ) async throws -> [Int] {
        let hypothesis = try await decodeWithTimings(
            encoderOutput: encoderOutput,
            encoderSequenceLength: encoderSequenceLength,
            actualAudioFrames: encoderSequenceLength,
            decoderModel: decoderModel,
            jointModel: jointModel,
            decoderState: &decoderState
        )
        return hypothesis.ySequence
    }

    private static func adaptConfigForV2(_ config: ASRConfig) -> ASRConfig {
        let tdt = config.tdtConfig
        guard tdt.blankId != 1024 else { return config }

        let adaptedTdt = TdtConfig(
            includeTokenDuration: tdt.includeTokenDuration,
            maxSymbolsPerStep: tdt.maxSymbolsPerStep,
            durationBins: tdt.durationBins,
            blankId: 1024,
            boundarySearchFrames: tdt.boundarySearchFrames,
            maxTokensPerChunk: tdt.maxTokensPerChunk,
            consecutiveBlankLimit: tdt.consecutiveBlankLimit
        )

        return ASRConfig(sampleRate: config.sampleRate, tdtConfig: adaptedTdt)
    }
}
