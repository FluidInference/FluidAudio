@preconcurrency import CoreML
import Foundation

extension AsrManager {

    /// How the model is told where the audio ends inside the padded input.
    ///
    /// The empty set is the normal path. The flags are recovery perturbations
    /// for a window the model decodes to nothing although it carries speech
    /// (#909): Parakeet TDT v3 has input cuts on which the joint never beats
    /// blank — the fp16 MLX port blanks on the same cuts, so this is the model,
    /// not quantization — and the output is on a knife edge with respect to the
    /// length inputs. Declaring the zero padding valid to the encoder
    /// (`encoderFull`), or to the preprocessor (`preprocessorFull`), or
    /// declaring the audio 0.2 s shorter with that tail silenced (`trimmedTail`),
    /// each flip cuts the others do not, while a good cut is never touched (the
    /// ladder runs only after an empty decode, and the decoder still stops at
    /// the real frames).
    struct InferenceLengthPolicy: OptionSet, CustomStringConvertible {
        let rawValue: Int
        static let encoderFull = InferenceLengthPolicy(rawValue: 1)
        static let preprocessorFull = InferenceLengthPolicy(rawValue: 2)
        static let trimmedTail = InferenceLengthPolicy(rawValue: 4)
        var description: String {
            var parts: [String] = []
            if contains(.encoderFull) { parts.append("encoderFull") }
            if contains(.preprocessorFull) { parts.append("preprocessorFull") }
            if contains(.trimmedTail) { parts.append("trimmedTail") }
            return parts.isEmpty ? "actual" : parts.joined(separator: "+")
        }
    }

    /// Audio declared absent by `.trimmedTail` (0.2 s).
    static let trimmedTailSamples = ASRConstants.sampleRate / 5
    /// The recovery ladder, in the order it is tried. Trimming costs the final
    /// 0.2 s (covered by the next window's overlap everywhere but at the very
    /// end of a stream), so it comes last, alone and combined.
    static let emptyDecodeRecoveryPolicies: [InferenceLengthPolicy] = [
        .encoderFull, .preprocessorFull, .trimmedTail, [.trimmedTail, .encoderFull], [.trimmedTail, .preprocessorFull],
    ]

    /// Minimum audio behind an empty decode before recovery is attempted (2 s),
    /// and the RMS below which a window counts as silence (about −50 dBFS).
    static let emptyDecodeRecoveryMinimumSamples = 2 * ASRConstants.sampleRate
    static let emptyDecodeRecoveryMinimumRMS: Float = 0.003

    /// Whether an empty decode of `samples[0..<actualLength]` deserves a retry:
    /// enough audio, and not silence. Pure.
    static func shouldRecoverEmptyDecode(samples: [Float], actualLength: Int) -> Bool {
        let length = min(actualLength, samples.count)
        guard length >= emptyDecodeRecoveryMinimumSamples else { return false }
        var energy: Double = 0
        samples.withUnsafeBufferPointer { buffer in
            for index in 0..<length { energy += Double(buffer[index] * buffer[index]) }
        }
        return (energy / Double(length)).squareRoot() >= Double(emptyDecodeRecoveryMinimumRMS)
    }

    internal func executeMLInferenceWithTimings(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        actualAudioFrames: Int? = nil,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0,
        language: Language? = nil,
        emitTokensAfterGlobalFrame: Int? = nil,
        initialTimeIndexOverride: Int? = nil
    ) async throws -> (hypothesis: TdtHypothesis, encoderSequenceLength: Int) {
        let audioLength = originalLength ?? paddedAudio.count
        let recoverable = Self.shouldRecoverEmptyDecode(samples: paddedAudio, actualLength: audioLength)
        // The decode mutates the state; keep a copy so a retry starts where the
        // first attempt did.
        let entryState = recoverable ? try TdtDecoderState(from: decoderState) : nil

        var result = try await runInference(
            paddedAudio, originalLength: originalLength, actualAudioFrames: actualAudioFrames,
            lengthPolicy: [], decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment, isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset, language: language,
            emitTokensAfterGlobalFrame: emitTokensAfterGlobalFrame,
            initialTimeIndexOverride: initialTimeIndexOverride)
        guard result.hypothesis.ySequence.isEmpty, let entryState else { return result }

        for policy in Self.emptyDecodeRecoveryPolicies {
            var retryState = try TdtDecoderState(from: entryState)
            let retry = try await runInference(
                paddedAudio, originalLength: originalLength, actualAudioFrames: actualAudioFrames,
                lengthPolicy: policy, decoderState: &retryState,
                contextFrameAdjustment: contextFrameAdjustment, isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset, language: language,
                emitTokensAfterGlobalFrame: emitTokensAfterGlobalFrame,
                initialTimeIndexOverride: initialTimeIndexOverride)
            guard !retry.hypothesis.ySequence.isEmpty else { continue }
            logger.info(
                "Empty decode of \(String(format: "%.1f", Double(audioLength) / Double(ASRConstants.sampleRate))) s of speech recovered with the \(String(describing: policy)) length policy (#909): \(retry.hypothesis.ySequence.count) tokens"
            )
            decoderState = retryState
            result = retry
            break
        }
        return result
    }

    private func runInference(
        _ paddedAudio: [Float],
        originalLength: Int?,
        actualAudioFrames: Int?,
        lengthPolicy: InferenceLengthPolicy,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int,
        isLastChunk: Bool,
        globalFrameOffset: Int,
        language: Language?,
        emitTokensAfterGlobalFrame: Int?,
        initialTimeIndexOverride: Int?
    ) async throws -> (hypothesis: TdtHypothesis, encoderSequenceLength: Int) {
        let fullLength = originalLength ?? paddedAudio.count
        // `.trimmedTail`: declare the audio 0.2 s shorter on a frame boundary and
        // silence the trimmed samples too — the declared length alone leaves
        // their mel frames in the input, and the cut does not flip.
        let trimmedLength =
            max(Self.emptyDecodeRecoveryMinimumSamples, fullLength - Self.trimmedTailSamples)
            / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
        let effectiveLength = lengthPolicy.contains(.trimmedTail) ? trimmedLength : fullLength
        var inputAudio = paddedAudio
        if lengthPolicy.contains(.trimmedTail), effectiveLength < inputAudio.count {
            for index in effectiveLength..<min(inputAudio.count, fullLength) {
                inputAudio[index] = 0
            }
        }
        let declaredLength: Int? = lengthPolicy.contains(.preprocessorFull) ? nil : effectiveLength
        let preprocessorInput = try await preparePreprocessorInput(inputAudio, actualLength: declaredLength)
        let preprocessorAudioArray = preprocessorInput.featureValue(for: "audio_signal")?.multiArrayValue
        do {
            guard let preprocessorModel = preprocessorModel else {
                throw ASRError.notInitialized
            }
            try Task.checkCancellation()
            let preprocessorOutput = try await preprocessorModel.compatPrediction(
                from: preprocessorInput,
                options: predictionOptions
            )
            let encoderOutputProvider: MLFeatureProvider
            if let encoderModel = encoderModel {
                // Split frontend: run separate encoder
                var encoderInput = try prepareEncoderInput(
                    encoder: encoderModel,
                    preprocessorOutput: preprocessorOutput,
                    originalInput: preprocessorInput
                )
                if lengthPolicy.contains(.encoderFull) {
                    encoderInput = try Self.declaringFullMelLength(encoderInput)
                }
                try Task.checkCancellation()
                encoderOutputProvider = try await encoderModel.compatPrediction(
                    from: encoderInput,
                    options: predictionOptions
                )
            } else {
                // Fused frontend: preprocessor output already contains encoder features
                encoderOutputProvider = preprocessorOutput
            }
            let rawEncoderOutput = try extractFeatureValue(
                from: encoderOutputProvider, key: "encoder", errorMessage: "Invalid encoder output")
            let encoderLength = try extractFeatureValue(
                from: encoderOutputProvider, key: "encoder_length",
                errorMessage: "Invalid encoder output length")
            let encoderSequenceLength = encoderLength[0].intValue
            // Calculate actual audio frames if not provided using shared constants
            let actualFrames =
                lengthPolicy.contains(.trimmedTail)
                ? ASRConstants.calculateEncoderFrames(from: effectiveLength)
                : actualAudioFrames ?? ASRConstants.calculateEncoderFrames(from: fullLength)
            let hypothesis = try await tdtDecodeWithTimings(
                encoderOutput: rawEncoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualFrames,
                originalAudioSamples: inputAudio,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset,
                language: language,
                emitTokensAfterGlobalFrame: emitTokensAfterGlobalFrame,
                initialTimeIndexOverride: initialTimeIndexOverride
            )
            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }
            return (hypothesis, encoderSequenceLength)
        } catch {
            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }
            throw error
        }
    }

    /// The encoder input with `mel_length` set to the padded mel frame count,
    /// so the encoder treats the zero padding as valid audio (the
    /// `.encoderFull` recovery policy). Inputs without `mel`/`mel_length`
    /// (fused frontends) are returned unchanged.
    static func declaringFullMelLength(_ input: MLFeatureProvider) throws -> MLFeatureProvider {
        guard let mel = input.featureValue(for: "mel")?.multiArrayValue,
            let melLength = input.featureValue(for: "mel_length")?.multiArrayValue, mel.shape.count == 3
        else { return input }
        let full = try MLMultiArray(shape: melLength.shape, dataType: melLength.dataType)
        full[0] = NSNumber(value: mel.shape[2].intValue)
        var features: [String: MLFeatureValue] = [:]
        for name in input.featureNames {
            features[name] = input.featureValue(for: name)
        }
        features["mel_length"] = MLFeatureValue(multiArray: full)
        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private func prepareEncoderInput(
        encoder: MLModel,
        preprocessorOutput: MLFeatureProvider,
        originalInput: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        let inputDescriptions = encoder.modelDescription.inputDescriptionsByName

        let missingNames = inputDescriptions.keys.filter { name in
            preprocessorOutput.featureValue(for: name) == nil
        }

        if missingNames.isEmpty {
            return preprocessorOutput
        }

        var features: [String: MLFeatureValue] = [:]

        for name in inputDescriptions.keys {
            if let value = preprocessorOutput.featureValue(for: name) {
                features[name] = value
                continue
            }

            if let fallback = originalInput.featureValue(for: name) {
                features[name] = fallback
                continue
            }

            let availableInputs = preprocessorOutput.featureNames.sorted().joined(separator: ", ")
            let fallbackInputs = originalInput.featureNames.sorted().joined(separator: ", ")
            throw ASRError.processingFailed(
                "Missing required encoder input: \(name). Available inputs: \(availableInputs), "
                    + "fallback inputs: \(fallbackInputs)"
            )
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// Align audio samples to encoder frame boundaries by zero-padding to the next frame boundary.
    /// Returns the aligned samples and the frame-aligned length.
    /// - Parameters:
    ///   - audioSamples: Raw audio samples
    ///   - allowAlignment: When false, skip alignment (e.g. when previous context exists)
    nonisolated internal func frameAlignedAudio(
        _ audioSamples: [Float], allowAlignment: Bool = true
    ) -> (samples: [Float], frameAlignedLength: Int) {
        let originalLength = audioSamples.count
        let frameAlignedCandidate =
            ((originalLength + ASRConstants.samplesPerEncoderFrame - 1)
                / ASRConstants.samplesPerEncoderFrame) * ASRConstants.samplesPerEncoderFrame
        if allowAlignment && frameAlignedCandidate > originalLength
            && frameAlignedCandidate <= ASRConstants.maxModelSamples
        {
            let aligned = audioSamples + Array(repeating: 0, count: frameAlignedCandidate - originalLength)
            return (aligned, frameAlignedCandidate)
        }
        return (audioSamples, originalLength)
    }

    nonisolated internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
    }

}
