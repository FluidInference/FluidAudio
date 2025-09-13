import CoreML
import Foundation
import OSLog

@available(macOS 13.0, *)
public enum TTSHarness {

    /// Minimal, self-contained CoreML TTS harness (Swift version of main4.py)
    public static func run(arguments: [String]) async {
        // Defaults
        var textArg: String? = nil
        var voice = "af_heart"
        var output = "tts_harness_output.wav"
        var modelPath: String? = nil
        var debug = false

        // Parse args: allow first positional as text
        var i = 0
        while i < arguments.count {
            let a = arguments[i]
            switch a {
            case "--voice", "-v":
                if i + 1 < arguments.count { voice = arguments[i + 1]; i += 1 }
            case "--output", "-o":
                if i + 1 < arguments.count { output = arguments[i + 1]; i += 1 }
            case "--model", "-m":
                if i + 1 < arguments.count { modelPath = arguments[i + 1]; i += 1 }
            case "--debug":
                debug = true
            default:
                if textArg == nil { textArg = a }
            }
            i += 1
        }

        let text = textArg ?? "Hello, this is FluidAudio."

        do {
            // Locate model
            let modelURL = try await resolveModelURL(explicit: modelPath)
            if debug { print("Model: \(modelURL.path)") }

            // Load resources
            let (vocab, w2p) = try loadResources()

            // Text -> phonemes
            var phonemes = textToPhonemes(text, using: w2p)
            if vocab["a"] != nil { phonemes.insert("a", at: 0) } // language token
            if debug {
                print("Phonemes(\(phonemes.count)): \(phonemes.prefix(32).joined(separator: " "))")
            }

            // Phonemes -> ids, with BOS/EOS zeros
            var ids: [Int32] = [0]
            for p in phonemes { if let id = vocab[p] { ids.append(id) } }
            ids.append(0)
            if debug { print("Token IDs(\(ids.count)): \(ids.prefix(32))") }

            // Load model (compile if mlpackage)
            let model = try await loadModel(from: modelURL)

            // Determine token length from model description
            let targetTokens = tokenLength(from: model) ?? 139
            if debug { print("targetTokens=\(targetTokens)") }

            // Build inputs
            let trueLen = min(ids.count, targetTokens)
            let inputIds = try MLMultiArray(shape: [1, NSNumber(value: targetTokens)], dataType: .int32)
            for (idx, v) in ids.prefix(trueLen).enumerated() { inputIds[idx] = NSNumber(value: v) }
            for idx in trueLen..<targetTokens { inputIds[idx] = 0 }

            let attention = try MLMultiArray(shape: [1, NSNumber(value: targetTokens)], dataType: .int32)
            for idx in 0..<targetTokens { attention[idx] = NSNumber(value: idx < trueLen ? 1 : 0) }
            if debug { print("maskOn=\((0..<targetTokens).reduce(0){ $0 + attention[$1].intValue })") }

            let ref = try loadVoiceEmbeddingJSON(voice: voice, phonemeCount: phonemes.count)
            if debug {
                let norm = sqrt((0..<ref.count).reduce(0.0) { $0 + Double(truncating: ref[$1]) * Double(truncating: ref[$1]) })
                print(String(format: "ref_s norm=%.3f", norm))
            }

            let phases = try MLMultiArray(shape: [1, 9], dataType: .float32)
            for idx in 0..<9 { phases[idx] = 0 }

            // Input names and presence
            let inputs = model.modelDescription.inputDescriptionsByName
            var dict: [String: MLFeatureValue] = [:]
            dict["input_ids"] = MLFeatureValue(multiArray: inputIds)
            if inputs["attention_mask"] != nil { dict["attention_mask"] = MLFeatureValue(multiArray: attention) }
            dict["ref_s"] = MLFeatureValue(multiArray: ref)
            if inputs["random_phases"] != nil { dict["random_phases"] = MLFeatureValue(multiArray: phases) }

            let provider = try MLDictionaryFeatureProvider(dictionary: dict)
            let out = try model.prediction(from: provider)

            // Extract audio and optional length
            guard let audioArr = out.featureValue(for: "audio")?.multiArrayValue else {
                throw NSError(domain: "TTSHarness", code: 2, userInfo: [NSLocalizedDescriptionKey: "No 'audio' output in model outputs: \(Array(out.featureNames))"]) }

            var effective = audioArr.count
            if let fv = out.featureValue(for: "audio_length_samples") {
                if let a = fv.multiArrayValue, a.count > 0 { effective = max(0, a[0].intValue) }
                else if fv.type == .int64 { effective = max(0, Int(fv.int64Value)) }
                else if fv.type == .double { effective = max(0, Int(fv.doubleValue)) }
            }
            effective = min(effective, audioArr.count)

            // Convert to Float samples
            var samples: [Float] = []
            samples.reserveCapacity(effective)
            for idx in 0..<effective { samples.append(audioArr[idx].floatValue) }

            // Normalize
            if let m = samples.map({ abs($0) }).max(), m > 0 { samples = samples.map { $0 / m } }

            // Write WAV
            let outURL = URL(fileURLWithPath: output)
            try writeWav(samples: samples, sampleRate: 24000, to: outURL)
            print("Saved: \(outURL.path) (\(String(format: "%.2f", Double(samples.count) / 24000.0))s)")
        } catch {
            print("TTSHarness error: \(error.localizedDescription)")
        }
    }

    // MARK: - Helpers

    private static func resolveModelURL(explicit: String?) async throws -> URL {
        let fm = FileManager.default
        if let p = explicit {
            let u = URL(fileURLWithPath: p)
            if fm.fileExists(atPath: u.path) { return u }
            throw NSError(domain: "TTSHarness", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not found at \(u.path)"])
        }
        // Prefer local mlpackage, then cache mlmodelc
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let localPkg = cwd.appendingPathComponent("kokoro_completev20.mlpackage")
        if fm.fileExists(atPath: localPkg.path) { return localPkg }
        let cache = try cacheDir().appendingPathComponent("Models/kokoro/kokoro_completev20.mlmodelc")
        return cache
    }

    private static func loadModel(from url: URL) async throws -> MLModel {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        if url.pathExtension == "mlpackage" {
            let compiled = try await MLModel.compileModel(at: url)
            return try MLModel(contentsOf: compiled, configuration: cfg)
        } else {
            return try MLModel(contentsOf: url, configuration: cfg)
        }
    }

    private static func tokenLength(from model: MLModel) -> Int? {
        if let c = model.modelDescription.inputDescriptionsByName["input_ids"]?.multiArrayConstraint {
            let shape = c.shape
            if shape.count >= 2 { return shape.last?.intValue }
        }
        return nil
    }

    private static func cacheDir() throws -> URL {
        #if os(macOS)
        let base = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache")
        #else
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        #endif
        let dir = base.appendingPathComponent("fluidaudio")
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    private static func loadResources() throws -> ([String: Int32], [String: [String]]) {
        let cache = try cacheDir().appendingPathComponent("Models/kokoro")
        let fm = FileManager.default
        let vocabURL = fm.fileExists(atPath: cache.appendingPathComponent("vocab_index.json").path)
            ? cache.appendingPathComponent("vocab_index.json")
            : URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("vocab_index.json")
        let wpURL = fm.fileExists(atPath: cache.appendingPathComponent("word_phonemes.json").path)
            ? cache.appendingPathComponent("word_phonemes.json")
            : URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("word_phonemes.json")

        let vocabData = try Data(contentsOf: vocabURL)
        let vocabJSON = try JSONSerialization.jsonObject(with: vocabData) as! [String: Any]
        let vocabDict = vocabJSON["vocab"] as? [String: Any] ?? [:]
        var vocab: [String: Int32] = [:]
        for (k, v) in vocabDict { if let vi = v as? Int { vocab[k] = Int32(vi) } else if let vd = v as? Double { vocab[k] = Int32(vd) } }

        let wpData = try Data(contentsOf: wpURL)
        let wpJSON = try JSONSerialization.jsonObject(with: wpData) as! [String: Any]
        guard let w2p = wpJSON["word_to_phonemes"] as? [String: [String]] else {
            throw NSError(domain: "TTSHarness", code: 3, userInfo: [NSLocalizedDescriptionKey: "Invalid word_phonemes.json format"])
        }
        return (vocab, w2p)
    }

    private static func textToPhonemes(_ text: String, using w2p: [String: [String]]) -> [String] {
        let parts = text.lowercased().split(separator: " ")
        var out: [String] = []
        for w in parts {
            let clean = w.filter { $0.isLetter || $0.isNumber }
            if let arr = w2p[String(clean)] {
                out.append(contentsOf: arr)
                out.append(" ")
            }
        }
        if out.last == " " { out.removeLast() }
        return out
    }

    private static func loadVoiceEmbeddingJSON(voice: String, phonemeCount: Int) throws -> MLMultiArray {
        let voicesDir = try cacheDir().appendingPathComponent("Models/kokoro/voices")
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        // Try multiple candidates (prefer local repo files first)
        let candidates: [URL] = [
            cwd.appendingPathComponent("voices/\(voice).json"),
            cwd.appendingPathComponent("\(voice).json"),
            voicesDir.appendingPathComponent("\(voice).json"),
        ]
        guard let jsonURL = candidates.first(where: { fm.fileExists(atPath: $0.path) }) else {
            throw NSError(
                domain: "TTSHarness",
                code: 100,
                userInfo: [NSLocalizedDescriptionKey: "Voice embedding JSON not found for \(voice). Checked: \(candidates.map { $0.path })"]
            )
        }

        let data = try Data(contentsOf: jsonURL)
        let jsonAny = try JSONSerialization.jsonObject(with: data)
        var vec: [Float]? = nil

        func parseArray(_ any: Any) -> [Float]? {
            if let ds = any as? [Double] { return ds.map { Float($0) } }
            if let fs = any as? [Float] { return fs }
            if let ns = any as? [NSNumber] { return ns.map { $0.floatValue } }
            if let arr = any as? [Any] {
                var out: [Float] = []
                out.reserveCapacity(arr.count)
                for v in arr {
                    if let n = v as? NSNumber { out.append(n.floatValue) }
                    else if let d = v as? Double { out.append(Float(d)) }
                    else if let f = v as? Float { out.append(f) }
                    else { return nil }
                }
                return out
            }
            return nil
        }

        if let arr = parseArray(jsonAny) { vec = arr }
        else if let dict = jsonAny as? [String: Any] {
            if let embed = dict["embedding"], let arr = parseArray(embed) { vec = arr }
            else if let byVoice = dict[voice], let arr = parseArray(byVoice) { vec = arr }
            else {
                let numericKeys = dict.keys.compactMap { Int($0) }.sorted()
                var chosen: [Float]? = nil
                if let exact = dict["\(phonemeCount)"] { chosen = parseArray(exact) }
                else if let k = numericKeys.last(where: { $0 <= phonemeCount }), let cand = dict["\(k)"] { chosen = parseArray(cand) }
                if let c = chosen { vec = c }
                else if let any = dict.values.first { vec = parseArray(any) }
            }
        }

        let dim = 256
        let arr = try MLMultiArray(shape: [1, NSNumber(value: dim)], dataType: .float32)
        guard let vec = vec, vec.count == dim else {
            throw NSError(
                domain: "TTSHarness",
                code: 101,
                userInfo: [NSLocalizedDescriptionKey: "Invalid or missing voice embedding for \(voice) at \(jsonURL.path) (expected 256 floats, got \(vec?.count ?? -1))"]
            )
        }
        let values = vec
        for i in 0..<dim { arr[i] = NSNumber(value: values[i]) }
        return arr
    }

    private static func writeWav(samples: [Float], sampleRate: Int, to url: URL) throws {
        var pcmData = Data()
        pcmData.reserveCapacity(samples.count * 2)
        for s in samples {
            let clipped = max(-1.0, min(1.0, s))
            let i16 = Int16(clipped * 32767)
            pcmData.append(contentsOf: withUnsafeBytes(of: i16.littleEndian) { Array($0) })
        }

        var wav = Data()
        wav.append("RIFF".data(using: .ascii)!)
        let fileSize = UInt32(36 + pcmData.count)
        wav.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        wav.append("WAVE".data(using: .ascii)!)
        wav.append("fmt ".data(using: .ascii)!)
        wav.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        wav.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) }) // PCM
        wav.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) }) // mono
        wav.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        wav.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate * 2).littleEndian) { Array($0) })
        wav.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) }) // block align
        wav.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) }) // bits per sample
        wav.append("data".data(using: .ascii)!)
        wav.append(contentsOf: withUnsafeBytes(of: UInt32(pcmData.count).littleEndian) { Array($0) })
        wav.append(pcmData)

        try wav.write(to: url)
    }
}
