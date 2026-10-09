#if TTS
@preconcurrency import CoreML
import Foundation

/// ANE-v3 runtime. Owned and called synchronously by KokoroAneModelStore.
/// Supply the selected language's pinned base assets,
/// export-fast-pipeline assets, and the masked split-decoder assets separately.
/// Uses original models for requests outside available static buckets.
/// A pipeline serializes calls; create separate instances for independent streams.
final class KokoroAneV3Pipeline {
    struct Result: Sendable {
        let samples: [Float]
        let acousticFrames: Int
        let durationFrames: [Int32]
        let stageMilliseconds: [String: Double]
        let usedFastVocoder: Bool
    }
    private let models: [String: MLModel]
    private let albertBuckets: [Int]
    private let decoderBuckets: [Int]
    private let source: KokoroAneV3NativeSource
    private let lock = NSLock()
    private typealias Arrays = KokoroAneV3Arrays

    init(baseDirectory: URL, fastDirectory: URL, decoderDirectory: URL) throws {
        source = try KokoroAneV3NativeSource(parametersURL: fastDirectory.appendingPathComponent("native-source.json"))
        func open(
            _ directory: URL, _ name: String, _ units: MLComputeUnits, lowPrecision: Bool = false
        ) throws -> MLModel {
            let compiled = directory.appendingPathComponent(name + ".mlmodelc")
            let package = directory.appendingPathComponent(name + ".mlpackage")
            let configuration = MLModelConfiguration()
            configuration.computeUnits = units
            configuration.allowLowPrecisionAccumulationOnGPU = lowPrecision
            if FileManager.default.fileExists(atPath: compiled.path) {
                return try MLModel(contentsOf: compiled, configuration: configuration)
            }
            guard FileManager.default.fileExists(atPath: package.path) else {
                throw KokoroAneV3Error.missingModel(name)
            }
            let temporary = try MLModel.compileModel(at: package)
            defer { try? FileManager.default.removeItem(at: temporary) }
            let staging = compiled.appendingPathExtension(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: staging) }
            try FileManager.default.copyItem(at: temporary, to: staging)
            do {
                try FileManager.default.moveItem(at: staging, to: compiled)
            } catch {
                guard FileManager.default.fileExists(atPath: compiled.path) else { throw error }
            }
            return try MLModel(contentsOf: compiled, configuration: configuration)
        }
        func buckets(_ directory: URL, _ prefix: String) throws -> [Int] {
            let names = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            return Array(
                Set(
                    names.compactMap { url -> Int? in
                        guard ["mlpackage", "mlmodelc"].contains(url.pathExtension),
                            url.lastPathComponent.hasPrefix(prefix)
                        else { return nil }
                        return Int(url.deletingPathExtension().lastPathComponent.dropFirst(prefix.count))
                    })
            ).sorted()
        }
        albertBuckets = try buckets(fastDirectory, "KokoroAlbert_")
        decoderBuckets = try buckets(decoderDirectory, "KokoroDecoderPre_")
        guard !albertBuckets.isEmpty, !decoderBuckets.isEmpty else {
            throw KokoroAneV3Error.missingModel("Static buckets")
        }
        var loaded: [String: MLModel] = [:]
        for stage in ["Albert", "PostAlbert", "Alignment", "Prosody", "Noise", "Vocoder", "Tail"] {
            let suffix = ["Noise", "Prosody", "Tail"].contains(stage) ? "_v2" : ""
            let units: MLComputeUnits = ["Noise", "Tail"].contains(stage) ? .cpuAndGPU : .cpuAndNeuralEngine
            loaded[stage] = try open(baseDirectory, "Kokoro" + stage + suffix, units, lowPrecision: true)
        }
        for bucket in albertBuckets {
            loaded["Albert_\(bucket)"] = try open(fastDirectory, "KokoroAlbert_\(bucket)", .cpuAndNeuralEngine)
        }
        for bucket in decoderBuckets {
            loaded["Decoder_\(bucket)"] = try open(decoderDirectory, "KokoroDecoderPre_\(bucket)", .cpuAndNeuralEngine)
        }
        loaded["SourceGenerator"] = try open(fastDirectory, "KokoroSourceGenerator", .cpuAndGPU)
        models = loaded
    }

    func synthesize(inputIds: [Int32], style: [Float], speed: Float = 1) throws -> Result {
        guard (2...512).contains(inputIds.count), inputIds.allSatisfy({ (0..<178).contains($0) }),
            style.count == 256, style.allSatisfy(\.isFinite), speed.isFinite, speed > 0,
            (Float(0x1p-24)...65504).contains(speed)
        else {
            throw KokoroAneV3Error.invalidInput(
                "Expected 2...512 valid tokens, finite 256-value style and positive speed")
        }
        lock.lock()
        defer { lock.unlock() }
        return try autoreleasepool { try run(inputIds: inputIds, style: style, speed: speed) }
    }

    private func run(inputIds: [Int32], style voice: [Float], speed: Float) throws -> Result {
        var timings: [String: Double] = [:]
        func time() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
        func predict(_ stage: String, _ input: [String: MLMultiArray]) throws -> MLFeatureProvider {
            guard let model = models[stage] else { throw KokoroAneV3Error.missingModel(stage) }
            let start = time()
            var prepared = input
            for (name, value) in input {
                let expected = model.modelDescription.inputDescriptionsByName[name]?.multiArrayConstraint?.dataType
                if expected == .float32 && value.dataType != .float32 { prepared[name] = try Self.single(value) }
                if expected == .float16 && value.dataType != .float16 { prepared[name] = try Self.half(value) }
            }
            try Task.checkCancellation()
            let output = try model.prediction(
                from: MLDictionaryFeatureProvider(dictionary: prepared.mapValues { MLFeatureValue(multiArray: $0) }))
            timings[stage] = time() - start
            return output
        }
        let count = inputIds.count
        let ids = try Arrays.int32Array(shape: [1, count], from: inputIds)
        let mask = try Arrays.attentionMask(length: count)
        let style = try Arrays.float16Array(shape: [1, 128], from: Array(voice.suffix(128)))
        let timbre = try Arrays.float16Array(shape: [1, 128], from: Array(voice.prefix(128)))
        let rate = try Arrays.float16Array(shape: [1], from: [speed])
        let bert: MLMultiArray
        if let bucket = albertBuckets.first(where: { $0 >= count }) {
            let paddedIds = try Arrays.int32Array(
                shape: [1, bucket], from: inputIds + [Int32](repeating: 0, count: bucket - count))
            let paddedMask = try Arrays.int32Array(shape: [1, bucket], from: (0..<bucket).map { $0 < count ? 1 : 0 })
            let output = try predict("Albert_\(bucket)", ["input_ids": paddedIds, "attention_mask": paddedMask])
            // Crop padding before any bidirectional LSTM: padding must never change its recurrence.
            let array = try Self.array(output, "bert_dur")
            bert = try Arrays.float16Array(
                shape: [1, count, 768], from: Array(Arrays.readFloats(array).prefix(count * 768)))
        } else {
            bert = try Self.half(Self.array(predict("Albert", ["input_ids": ids, "attention_mask": mask]), "bert_dur"))
        }
        let post = try predict(
            "PostAlbert", ["bert_dur": bert, "input_ids": ids, "attention_mask": mask, "style_s": style, "speed": rate])
        let values = try Arrays.readFloats(Self.array(post, "duration"))
        guard values.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 2000 }) else {
            throw KokoroAneV3Error.invalidInput("Invalid predicted durations")
        }
        let durations = values.map { max(1, Int32($0.rounded())) }
        let frames = durations.reduce(0) { $0 + Int($1) }
        guard frames >= 2, frames <= 2000 else { throw KokoroAneV3Error.invalidInput("Audio exceeds 2000-frame limit") }
        let alignment = try predict(
            "Alignment",
            [
                "pred_dur": Arrays.int32Array(shape: [1, count], from: durations),
                "d": Self.half(Self.array(post, "d")), "t_en": Self.half(Self.array(post, "t_en")),
            ])
        let asr = try Self.half(Self.array(alignment, "asr"))
        let prosody = try predict("Prosody", ["en": Self.half(Self.array(alignment, "en")), "style_s": style])
        let f0 = try Self.array(prosody, "F0")
        let noise = try Self.array(prosody, "N")
        let f16 = try Self.half(f0)
        let n16 = try Self.half(noise)
        var xPre: MLMultiArray?
        var generatedAudio: MLMultiArray?
        let bucket = decoderBuckets.first(where: { $0 >= frames })
        if let bucket {
            let start = time()
            let har = try source.build(f0: Arrays.readFloats(f0))
            let spectrum = try Arrays.float32Array(shape: [1, 22, frames * 120 + 1], from: har)
            timings["NativeSource"] = time() - start
            let valid = (0..<bucket).map { Float($0 < frames ? 1 : 0) }
            let decoded = try predict(
                "Decoder_\(bucket)",
                [
                    "asr": Self.resizeTime(asr, to: bucket), "F0_curve": Self.resizeTime(f16, to: bucket * 2),
                    "N_pred": Self.resizeTime(n16, to: bucket * 2), "style_timbre": timbre,
                    "mask": Arrays.float16Array(shape: [1, 1, bucket], from: valid),
                ])
            let features = try Self.resizeTime(Self.array(decoded, "features"), to: frames * 2)
            var feed = ["features": features, "har": spectrum, "style_timbre": timbre]
            if models["SourceGenerator"]?.modelDescription.inputDescriptionsByName["ref_s"] != nil {
                feed = [
                    "x_pre": features, "har": spectrum,
                    "ref_s": try Arrays.float32Array(shape: [1, 256], from: voice),
                    "mask": try Arrays.float32Array(
                        shape: [1, 1, frames * 2], from: [Float](repeating: 1, count: frames * 2)),
                    "mask_x10": try Arrays.float32Array(
                        shape: [1, 1, frames * 20], from: [Float](repeating: 1, count: frames * 20)),
                    "mask_x60": try Arrays.float32Array(
                        shape: [1, 1, frames * 120 + 1], from: [Float](repeating: 1, count: frames * 120 + 1)),
                ]
            }
            if models["SourceGenerator"]?.modelDescription.inputDescriptionsByName["weights_x10"] != nil {
                let rowWeights =
                    models["SourceGenerator"]?.modelDescription.inputDescriptionsByName["weights_x10"]?
                    .multiArrayConstraint?.shape[1].intValue == 1
                feed["weights_x10"] = try Arrays.float16Array(
                    shape: rowWeights ? [1, 1, frames * 20] : [1, frames * 20, 1],
                    from: [Float](repeating: 1 / 1024, count: frames * 20))
                feed["weights_x60"] = try Arrays.float16Array(
                    shape: rowWeights ? [1, 1, frames * 120 + 1] : [1, frames * 120 + 1, 1],
                    from: [Float](repeating: 1 / 1024, count: frames * 120 + 1))
            }
            let generated = try predict("SourceGenerator", feed)
            if let waveform = generated.featureValue(for: "audio")?.multiArrayValue
                ?? generated.featureValue(for: "waveform")?.multiArrayValue
            {
                generatedAudio = waveform
            } else {
                xPre = try Self.array(generated, "x_pre")
            }
        } else {
            let timbre32 = try Arrays.float32Array(shape: [1, 128], from: Array(voice.prefix(128)))
            let source = try predict("Noise", ["F0_curve": Self.single(f0), "style_timbre": timbre32])
            let generated = try predict(
                "Vocoder",
                [
                    "asr": asr, "F0_curve": f16, "N_pred": n16, "style_timbre": timbre,
                    "x_source_0": Self.half(Self.array(source, "x_source_0")),
                    "x_source_1": Self.half(Self.array(source, "x_source_1")),
                ])
            xPre = try Self.array(generated, "x_pre")
        }
        let audio: [Float]
        if let generatedAudio {
            audio = Arrays.readFloats(generatedAudio)
        } else if let xPre {
            let tail = try predict("Tail", ["x_pre": Self.single(xPre)])
            audio = try Arrays.readFloats(Self.array(tail, "audio"))
        } else {
            throw KokoroAneV3Error.missingOutput("audio/x_pre")
        }
        guard audio.count == frames * 600, audio.allSatisfy(\.isFinite) else {
            throw KokoroAneV3Error.invalidInput("Invalid audio output")
        }
        return Result(
            samples: audio, acousticFrames: frames, durationFrames: durations, stageMilliseconds: timings,
            usedFastVocoder: bucket != nil)
    }
    private static func array(_ output: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
        guard let value = output.featureValue(for: name)?.multiArrayValue else {
            throw KokoroAneV3Error.missingOutput(name)
        }
        return value
    }
    private static func half(_ a: MLMultiArray) throws -> MLMultiArray {
        try Arrays.float16Array(shape: a.shape.map(\.intValue), from: a)
    }
    private static func single(_ a: MLMultiArray) throws -> MLMultiArray {
        try Arrays.float32Array(shape: a.shape.map(\.intValue), from: a)
    }
    static func resizeTime(_ a: MLMultiArray, to target: Int) throws -> MLMultiArray {
        let shape = a.shape.map(\.intValue)
        let strides = a.strides.map(\.intValue)
        guard (2...3).contains(shape.count), shape[0] == 1, a.dataType == .float16, target > 0 else {
            throw KokoroAneV3Error.invalidInput("Invalid resize shape")
        }
        var newShape = shape
        newShape[newShape.count - 1] = target
        let result = try MLMultiArray(shape: newShape.map(NSNumber.init), dataType: .float16)
        memset(result.dataPointer, 0, result.count * 2)
        let rows = shape.count == 3 ? shape[1] : 1
        let source = a.dataPointer.assumingMemoryBound(to: UInt16.self)
        let dest = result.dataPointer.assumingMemoryBound(to: UInt16.self)
        let rowStride = shape.count == 3 ? strides[1] : 0
        for row in 0..<rows {
            for t in 0..<min(shape[shape.count - 1], target) {
                dest[row * target + t] = source[row * rowStride + t * strides[strides.count - 1]]
            }
        }
        return result
    }
}
#endif
