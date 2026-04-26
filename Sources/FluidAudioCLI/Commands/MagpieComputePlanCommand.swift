#if os(macOS)
@preconcurrency import CoreML
import FluidAudio
import Foundation

/// Per-model compute-device probe via timing (`MLComputePlan` crashes on
/// Magpie's scatter-heavy graphs). Runs N forward passes of each .mlmodelc
/// under cpuOnly / cpuAndGPU / cpuAndNeuralEngine and compares wall time.
/// The fastest config indicates which compute device the runtime actually
/// chose. ANE usage is inferred by `cpuAndNeuralEngine` being meaningfully
/// faster than `cpuOnly`; if same speed → ANE fell back to CPU.
public enum MagpieComputePlanCommand {

    public static func run(arguments: [String]) async {
        let cacheDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/fluidaudio/Models/magpie-tts")

        let models: [(String, String, () throws -> [String: MLFeatureValue])] = [
            ("text_encoder", "text_encoder.mlmodelc", makeTextEncoderInputs),
            ("decoder_prefill", "decoder_prefill.mlmodelc", makePrefillInputs),
            ("decoder_step", "decoder_step.mlmodelc", makeDecoderStepInputs),
            ("nanocodec_decoder", "nanocodec_decoder.mlmodelc", makeNanocodecInputs),
        ]

        let configs: [(String, MLComputeUnits)] = [
            ("CPU", .cpuOnly),
            ("CPU+GPU", .cpuAndGPU),
            ("CPU+ANE", .cpuAndNeuralEngine),
        ]

        let warmup = 1
        let iters = 3

        print(
            "Model                CPU only   CPU+GPU    CPU+ANE    ANE actually used?")
        print(String(repeating: "-", count: 78))
        for (name, file, makeInputs) in models {
            let url = cacheDir.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else {
                print("\(name.padding(toLength: 20, withPad: " ", startingAt: 0))NOT FOUND")
                continue
            }
            var times: [String: Double] = [:]
            for (label, units) in configs {
                let cfg = MLModelConfiguration()
                cfg.computeUnits = units
                do {
                    let model = try MLModel(contentsOf: url, configuration: cfg)
                    let provider = try MLDictionaryFeatureProvider(dictionary: try makeInputs())
                    for _ in 0..<warmup { _ = try await model.prediction(from: provider) }
                    let t0 = Date()
                    for _ in 0..<iters { _ = try await model.prediction(from: provider) }
                    let dt = Date().timeIntervalSince(t0) / Double(iters)
                    times[label] = dt
                } catch {
                    times[label] = -1
                }
            }
            let cpu = times["CPU"] ?? -1
            let gpu = times["CPU+GPU"] ?? -1
            let ane = times["CPU+ANE"] ?? -1
            let cellW = 11
            let aneVerdict: String = {
                guard cpu > 0, ane > 0 else { return "n/a" }
                let ratio = cpu / ane
                if ratio > 1.3 { return "yes (\(String(format: "%.1f", ratio))× vs CPU)" }
                if ratio < 0.85 { return "no — slower than CPU" }
                return "no — same as CPU (fallback)"
            }()
            func fmt(_ t: Double) -> String {
                if t < 0 { return "ERR".padding(toLength: cellW, withPad: " ", startingAt: 0) }
                return String(format: "%.0fms", t * 1000)
                    .padding(toLength: cellW, withPad: " ", startingAt: 0)
            }
            print(
                "\(name.padding(toLength: 20, withPad: " ", startingAt: 0))"
                    + "\(fmt(cpu))\(fmt(gpu))\(fmt(ane))\(aneVerdict)"
            )
        }
    }

    // MARK: - Dummy inputs that match each model's I/O signature

    private static func makeTextEncoderInputs() throws -> [String: MLFeatureValue] {
        let tokens = try MLMultiArray(shape: [1, 256], dataType: .int32)
        memset(tokens.dataPointer, 0, tokens.count * MemoryLayout<Int32>.size)
        let mask = try MLMultiArray(shape: [1, 256], dataType: .float32)
        memset(mask.dataPointer, 0, mask.count * MemoryLayout<Float>.size)
        return [
            "text_tokens": MLFeatureValue(multiArray: tokens),
            "text_mask": MLFeatureValue(multiArray: mask),
        ]
    }

    private static func makePrefillInputs() throws -> [String: MLFeatureValue] {
        let audioEmbed = try MLMultiArray(shape: [1, 110, 768], dataType: .float32)
        memset(audioEmbed.dataPointer, 0, audioEmbed.count * MemoryLayout<Float>.size)
        let encOut = try MLMultiArray(shape: [1, 256, 768], dataType: .float32)
        memset(encOut.dataPointer, 0, encOut.count * MemoryLayout<Float>.size)
        let encMask = try MLMultiArray(shape: [1, 256], dataType: .float32)
        memset(encMask.dataPointer, 0, encMask.count * MemoryLayout<Float>.size)
        return [
            "audio_embed": MLFeatureValue(multiArray: audioEmbed),
            "encoder_output": MLFeatureValue(multiArray: encOut),
            "encoder_mask": MLFeatureValue(multiArray: encMask),
        ]
    }

    private static func makeDecoderStepInputs() throws -> [String: MLFeatureValue] {
        let audioEmbed = try MLMultiArray(shape: [1, 1, 768], dataType: .float32)
        memset(audioEmbed.dataPointer, 0, audioEmbed.count * MemoryLayout<Float>.size)
        let encOut = try MLMultiArray(shape: [1, 256, 768], dataType: .float32)
        memset(encOut.dataPointer, 0, encOut.count * MemoryLayout<Float>.size)
        let encMask = try MLMultiArray(shape: [1, 256], dataType: .float32)
        memset(encMask.dataPointer, 0, encMask.count * MemoryLayout<Float>.size)
        var inputs: [String: MLFeatureValue] = [
            "audio_embed": MLFeatureValue(multiArray: audioEmbed),
            "encoder_output": MLFeatureValue(multiArray: encOut),
            "encoder_mask": MLFeatureValue(multiArray: encMask),
        ]
        for i in 0..<12 {
            let k = try MLMultiArray(shape: [1, 512, 12, 64], dataType: .float16)
            memset(k.dataPointer, 0, k.count * 2)
            let v = try MLMultiArray(shape: [1, 512, 12, 64], dataType: .float16)
            memset(v.dataPointer, 0, v.count * 2)
            let pos = try MLMultiArray(shape: [1], dataType: .float16)
            memset(pos.dataPointer, 0, 2)
            inputs["cache_k\(i)"] = MLFeatureValue(multiArray: k)
            inputs["cache_v\(i)"] = MLFeatureValue(multiArray: v)
            inputs["position\(i)"] = MLFeatureValue(multiArray: pos)
        }
        return inputs
    }

    private static func makeNanocodecInputs() throws -> [String: MLFeatureValue] {
        // (8, 24) typical token count; nanocodec accepts variable length but needs
        // a sane shape. Use 24 frames as a representative count.
        let codes = try MLMultiArray(shape: [1, 8, 24], dataType: .int32)
        memset(codes.dataPointer, 0, codes.count * MemoryLayout<Int32>.size)
        return ["audio_codes": MLFeatureValue(multiArray: codes)]
    }
}
#endif
