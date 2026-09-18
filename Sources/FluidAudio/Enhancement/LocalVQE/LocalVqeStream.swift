@preconcurrency import CoreML
import Foundation
import OSLog

/// Stateful hop-by-hop LocalVQE session.
///
/// Feed equal-length mic and far-end reference audio (16 kHz mono Float32)
/// in any buffer size; the stream assembles whole model calls internally and
/// returns enhanced samples as they complete. Output is sample-aligned with
/// input (sample `i` out corresponds to sample `i` in) but is delivered one
/// hop (256 samples, 16 ms) later than the input that produced it, plus
/// whatever remains buffered toward the next call. Call `flush()` at the end
/// of a clip to drain the delay line so the total output length equals the
/// total input length.
///
/// The Core ML model carries every recurrent state (conv histories, delay
/// windows, S4D bottleneck, overlap-add tail) as explicit `in_*`/`out_*`
/// tensors; the stream just passes each call's outputs back in as the next
/// call's inputs. Create one stream per audio channel pair; a stream is not
/// reusable across unrelated clips without `reset()`.
///
/// Streams created from one `LocalVqeManager` share its `MLModel`. Inference
/// goes through Core ML's async prediction API, which Apple documents as
/// thread-safe (WWDC23 10049), so independent streams may run concurrently;
/// the synchronous API would require serializing every call on the model.
public actor LocalVqeStream {

    private static let logger = AppLogger(category: "LocalVqeStream")
    private static let stateInputPrefix = "in_"
    private static let stateOutputPrefix = "out_"

    private let model: MLModel
    /// Samples consumed (and produced) per Core ML call.
    public let samplesPerCall: Int

    private let micInput: MLMultiArray
    private let refInput: MLMultiArray
    private let stateNames: [String]
    private let stateShapes: [String: [NSNumber]]
    private var states: [String: MLMultiArray] = [:]

    private var pendingMic: [Float] = []
    private var pendingRef: [Float] = []
    /// The first emitted hop covers t < 0 of the input and is discarded so
    /// that cumulative output stays aligned with cumulative input.
    private var leadingSamplesToDrop = LocalVqeManager.hopSize
    private var samplesIn = 0
    private var samplesOut = 0

    init(model: MLModel, samplesPerCall: Int) throws {
        self.model = model
        self.samplesPerCall = samplesPerCall

        let desc = model.modelDescription
        guard let micDesc = desc.inputDescriptionsByName["mic"]?.multiArrayConstraint,
            let refDesc = desc.inputDescriptionsByName["ref"]?.multiArrayConstraint
        else {
            throw LocalVqeError.modelLoadingFailed("model lacks 'mic'/'ref' inputs")
        }
        let expected = micDesc.shape.map { $0.intValue }.reduce(1, *)
        guard expected == samplesPerCall, refDesc.shape == micDesc.shape else {
            throw LocalVqeError.modelLoadingFailed(
                "model consumes \(expected) samples per call, expected \(samplesPerCall)")
        }
        micInput = try MLMultiArray(shape: micDesc.shape, dataType: .float32)
        refInput = try MLMultiArray(shape: refDesc.shape, dataType: .float32)

        var names: [String] = []
        var shapes: [String: [NSNumber]] = [:]
        for (inputName, inputDesc) in desc.inputDescriptionsByName where inputName.hasPrefix(Self.stateInputPrefix) {
            guard let constraint = inputDesc.multiArrayConstraint else { continue }
            let name = String(inputName.dropFirst(Self.stateInputPrefix.count))
            guard desc.outputDescriptionsByName[Self.stateOutputPrefix + name] != nil else {
                throw LocalVqeError.modelLoadingFailed("state '\(name)' has no matching output")
            }
            names.append(name)
            shapes[name] = constraint.shape
        }
        guard !names.isEmpty else {
            throw LocalVqeError.modelLoadingFailed("model exposes no in_*/out_* state tensors")
        }
        stateNames = names.sorted()
        stateShapes = shapes
        states = try Self.zeroStates(names: stateNames, shapes: stateShapes)
    }

    /// Number of state tensors the model carries between calls.
    public var stateCount: Int { stateNames.count }

    /// Clear all recurrent state and buffered audio; the next call starts a new clip.
    public func reset() throws {
        try resetStates()
        pendingMic.removeAll(keepingCapacity: true)
        pendingRef.removeAll(keepingCapacity: true)
        leadingSamplesToDrop = LocalVqeManager.hopSize
        samplesIn = 0
        samplesOut = 0
    }

    private func resetStates() throws {
        states = try Self.zeroStates(names: stateNames, shapes: stateShapes)
    }

    private static func zeroStates(names: [String], shapes: [String: [NSNumber]]) throws -> [String: MLMultiArray] {
        var fresh: [String: MLMultiArray] = [:]
        for name in names {
            guard let shape = shapes[name] else { continue }
            let array = try MLMultiArray(shape: shape, dataType: .float32)
            array.withUnsafeMutableBytes { ptr, _ in
                ptr.initializeMemory(as: UInt8.self, repeating: 0)
            }
            fresh[name] = array
        }
        return fresh
    }

    /// Push audio and return every enhanced sample that completed.
    ///
    /// `mic` and `reference` must have the same length; both may be any
    /// length (including zero) — partial calls are buffered until enough
    /// samples arrive.
    public func enhance(mic: [Float], reference: [Float]) async throws -> [Float] {
        guard mic.count == reference.count else {
            throw LocalVqeError.lengthMismatch(mic: mic.count, reference: reference.count)
        }
        pendingMic.append(contentsOf: mic)
        pendingRef.append(contentsOf: reference)
        samplesIn += mic.count

        var out: [Float] = []
        var offset = 0
        while pendingMic.count - offset >= samplesPerCall {
            let hop = try await runCall(
                mic: pendingMic[offset..<offset + samplesPerCall],
                reference: pendingRef[offset..<offset + samplesPerCall])
            out.append(contentsOf: hop)
            offset += samplesPerCall
        }
        if offset > 0 {
            pendingMic.removeFirst(offset)
            pendingRef.removeFirst(offset)
        }
        return emit(out)
    }

    /// Drain the delay line by feeding silence, returning the remaining
    /// samples so that total output length equals total input length.
    /// Ends the current clip: the stream is reset afterwards.
    public func flush() async throws -> [Float] {
        let outstanding = samplesIn - samplesOut
        guard outstanding > 0 else {
            try reset()
            return []
        }
        // Zeros needed to complete every outstanding sample, rounded up to whole calls.
        let needed = pendingMic.count + LocalVqeManager.hopSize
        let padded = ((needed + samplesPerCall - 1) / samplesPerCall) * samplesPerCall
        let zeros = [Float](repeating: 0, count: padded - pendingMic.count)
        pendingMic.append(contentsOf: zeros)
        pendingRef.append(contentsOf: zeros)

        var out: [Float] = []
        var offset = 0
        while pendingMic.count - offset >= samplesPerCall {
            let hop = try await runCall(
                mic: pendingMic[offset..<offset + samplesPerCall],
                reference: pendingRef[offset..<offset + samplesPerCall])
            out.append(contentsOf: hop)
            offset += samplesPerCall
        }
        let emitted = emit(out)
        let tail = Array(emitted.prefix(outstanding))
        try reset()
        return tail
    }

    /// Apply the leading-hop drop and account for emitted samples.
    private func emit(_ samples: [Float]) -> [Float] {
        var result = samples
        if leadingSamplesToDrop > 0 {
            let drop = min(leadingSamplesToDrop, result.count)
            result.removeFirst(drop)
            leadingSamplesToDrop -= drop
        }
        samplesOut += result.count
        return result
    }

    private func runCall(mic: ArraySlice<Float>, reference: ArraySlice<Float>) async throws -> [Float] {
        micInput.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            _ = buf.initialize(from: mic)
        }
        refInput.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            _ = buf.initialize(from: reference)
        }
        var features: [String: Any] = ["mic": micInput, "ref": refInput]
        for name in stateNames {
            features[Self.stateInputPrefix + name] = states[name]
        }

        let output: MLFeatureProvider
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: features)
            output = try await model.compatPrediction(from: provider, options: MLPredictionOptions())
        } catch {
            throw LocalVqeError.modelProcessingFailed(error.localizedDescription)
        }

        for name in stateNames {
            guard let next = output.featureValue(for: Self.stateOutputPrefix + name)?.multiArrayValue else {
                throw LocalVqeError.modelProcessingFailed("missing state output '\(name)'")
            }
            states[name] = next
        }
        guard let enhanced = output.featureValue(for: "enhanced")?.multiArrayValue else {
            throw LocalVqeError.modelProcessingFailed("missing 'enhanced' output")
        }
        return Self.floats(from: enhanced, count: samplesPerCall)
    }

    private static func floats(from array: MLMultiArray, count: Int) -> [Float] {
        var result = [Float](repeating: 0, count: count)
        switch array.dataType {
        case .float32:
            array.withUnsafeBufferPointer(ofType: Float.self) { buf in
                for i in 0..<min(count, buf.count) { result[i] = buf[i] }
            }
        default:
            for i in 0..<min(count, array.count) { result[i] = array[i].floatValue }
        }
        return result
    }
}
