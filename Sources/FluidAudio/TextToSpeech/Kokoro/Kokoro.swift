import CoreML
import Foundation
import OSLog

/// Direct TTS implementation using pre-computed phoneme dictionary and unified CoreML model
/// This follows the main.py approach - no pipeline, just dictionary lookup + model inference
@available(macOS 13.0, iOS 16.0, *)
public struct Kokoro {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroDirect")

    // Phoneme dictionary loaded from kokoro_word_phonemes_full.json
    private static var phonemeDictionary: [String: [String]] = [:]
    private static var isDictionaryLoaded = false
    private static var activeLanguageCode: String? = nil

    // Voice embeddings loaded from JSON
    private static var voiceEmbeddings: [String: [Int: [Float]]] = [:]

    /// Load the comprehensive phoneme dictionary
    public static func loadPhonemeDictionary(for languageCode: String?) throws {
        let normalized = languageCode?.lowercased()
        if isDictionaryLoaded, normalized == activeLanguageCode { return }

        let dictionary = try KokoroModel.phonemeDictionary(for: normalized)
        phonemeDictionary = dictionary
        activeLanguageCode = normalized
        isDictionaryLoaded = true
        logger.info(
            "Loaded \(phonemeDictionary.count) words from phoneme dictionary (lang=\(normalized ?? "default"))")
    }

    /// Convert text to phonemes using dictionary lookup (like main.py)
    public static func textToPhonemes(_ text: String) throws -> [String] {
        try loadPhonemeDictionary(for: activeLanguageCode)

        let words = text.lowercased().split(separator: " ")
        var allPhonemes: [String] = []

        for word in words {
            // Remove punctuation
            let cleanWord = String(word).filter { $0.isLetter || $0.isNumber }

            if let phonemes = phonemeDictionary[cleanWord] {
                allPhonemes.append(contentsOf: phonemes)
                allPhonemes.append(" ")  // Space between words
            } else {
                logger.warning(
                    "Word '\(cleanWord)' not in dictionary - dictionary has \(phonemeDictionary.count) words")
                print("DEBUG: Word '\(cleanWord)' not found in dictionary")
                // For unknown words, could fall back to letter-based approximation
                // For now, skip unknown words
            }
        }

        // Remove trailing space
        if allPhonemes.last == " " {
            allPhonemes.removeLast()
        }

        logger.info("Generated \(allPhonemes.count) phonemes: \(allPhonemes.prefix(20).joined(separator: " "))")
        print("DEBUG: Total phonemes generated: \(allPhonemes.count)")
        if allPhonemes.isEmpty {
            print("WARNING: No phonemes generated! Check dictionary loading.")
        }
        return allPhonemes
    }

    /// Convert phonemes to input IDs using correct vocabulary
    public static func phonemesToInputIds(_ phonemes: [String]) -> [Int32] {
        let vocabulary = KokoroVocabulary.getVocabulary()

        // Start with padding token [0]
        var ids: [Int32] = [0]

        for phoneme in phonemes {
            if let id = vocabulary[phoneme] {
                ids.append(id)
            } else {
                logger.debug("Unknown phoneme: '\(phoneme)' - skipping")
            }
        }

        // End with padding token [0]
        ids.append(0)

        print("DEBUG: Input IDs: \(ids.prefix(20))")
        return ids
    }

    /// Load voice embedding from indexed JSON (matching main.py's approach)
    public static func loadVoiceEmbedding(
        voice: String = KokoroVoiceCatalog.defaultVoiceId,
        phonemeCount: Int
    ) throws -> [Float] {
        let canonicalVoice = KokoroVoiceCatalog.canonicalVoiceId(for: voice) ?? voice
        if canonicalVoice != voice {
            logger.debug("Voice alias '\(voice)' normalized to '\(canonicalVoice)'")
        }

        // Try to load from kokoro_voices_indexed.json
        let currentDir = FileManager.default.currentDirectoryPath
        let voicesURL = URL(fileURLWithPath: currentDir).appendingPathComponent("kokoro_voices_indexed.json")

        if voiceEmbeddings.isEmpty && FileManager.default.fileExists(atPath: voicesURL.path) {
            let data = try Data(contentsOf: voicesURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: [String: [Double]]] {
                // The JSON structure is { "voice_name": { "length": [embedding_values] } }
                for (voiceName, lengthDict) in json {
                    var voiceEmbeddingsByLength: [Int: [Float]] = [:]
                    for (lengthStr, embedding) in lengthDict {
                        if let length = Int(lengthStr) {
                            voiceEmbeddingsByLength[length] = embedding.map { Float($0) }
                        }
                    }
                    let key = KokoroVoiceCatalog.canonicalVoiceId(for: voiceName) ?? voiceName
                    voiceEmbeddings[key] = voiceEmbeddingsByLength
                }
                logger.info("Loaded voice embeddings for \(voiceEmbeddings.count) voices")
            }
        }

        // Get embedding for this phoneme count
        let embeddingIndex = phonemeCount - 1  // Same as Python: len(ps) - 1

        if let voiceData = voiceEmbeddings[canonicalVoice],
            let embedding = voiceData[phonemeCount]
        {
            logger.info("Using embedding at index \(embeddingIndex) for \(phonemeCount) phonemes")
            print("DEBUG: Voice embedding first 5 values: \(embedding.prefix(5))")
            return embedding
        } else {
            logger.warning(
                "No embedding found for \(canonicalVoice) with \(phonemeCount) phonemes, using zeros")
            print(
                "WARNING: Using zero embeddings! Voice: \(canonicalVoice), phoneme count: \(phonemeCount)")
            print("DEBUG: Available voices: \(voiceEmbeddings.keys)")
            if let voiceData = voiceEmbeddings[canonicalVoice] {
                print("DEBUG: Available lengths for \(canonicalVoice): \(voiceData.keys.sorted())")
            }
            return Array(repeating: 0.0, count: 256)
        }
    }

    /// Main synthesis function (following main.py logic)
    public static func synthesize(
        text: String,
        voice: String = KokoroVoiceCatalog.defaultVoiceId
    ) throws -> Data {
        let canonicalVoice = KokoroVoiceCatalog.canonicalVoiceId(for: voice) ?? voice
        logger.info("Synthesizing: '\(text)' with voice: \(canonicalVoice)")
        if canonicalVoice != voice {
            logger.debug("Voice alias '\(voice)' resolved to '\(canonicalVoice)'")
        }

        let voiceMetadata = KokoroVoiceCatalog.voice(for: canonicalVoice)
        activeLanguageCode = voiceMetadata?.languageCode.lowercased()
        try loadPhonemeDictionary(for: activeLanguageCode)

        // Step 1: Text to phonemes
        let phonemes = try textToPhonemes(text)

        // Step 2: Phonemes to input IDs
        var inputIds = phonemesToInputIds(phonemes)

        // Step 3: Pad to 249 (model requirement)
        while inputIds.count < 249 {
            inputIds.append(0)
        }
        if inputIds.count > 249 {
            inputIds = Array(inputIds.prefix(249))
        }

        // Step 4: Get voice embedding
        let voiceEmbedding = try loadVoiceEmbedding(voice: canonicalVoice, phonemeCount: phonemes.count)

        // Step 5: Create model inputs
        let inputShape = [1, 249] as [NSNumber]
        let inputArray = try MLMultiArray(shape: inputShape, dataType: .int32)
        for (i, id) in inputIds.enumerated() {
            inputArray[i] = NSNumber(value: id)
        }

        let refShape = [1, 256] as [NSNumber]
        let refArray = try MLMultiArray(shape: refShape, dataType: .float32)
        let refPointer = refArray.dataPointer.bindMemory(to: Float.self, capacity: voiceEmbedding.count)
        voiceEmbedding.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            refPointer.initialize(from: base, count: src.count)
        }

        // Fixed phases (all zeros, matching main.py)
        let phasesShape = [1, 9] as [NSNumber]
        let phasesArray = try MLMultiArray(shape: phasesShape, dataType: .float32)
        let phasesPointer = phasesArray.dataPointer.bindMemory(to: Float.self, capacity: 9)
        phasesPointer.initialize(repeating: 0, count: 9)

        logger.info(
            "Input shapes - input_ids: \(inputArray.shape), ref_s: \(refArray.shape), phases: \(phasesArray.shape)")
        logger.debug("First 10 input_ids: \(inputIds.prefix(10))")
        logger.debug("First 5 ref_s values: \(voiceEmbedding.prefix(5))")

        // Step 6: Load and run model - use unified compiled model only (v21 preferred)
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let modelURL = cwd.appendingPathComponent("kokoro_completev21.mlmodelc")

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw TTSError.modelNotFound("Model not found at \(modelURL.path)")
        }

        let model = try MLModel(contentsOf: modelURL)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputArray),
            "ref_s": MLFeatureValue(multiArray: refArray),
            "random_phases": MLFeatureValue(multiArray: phasesArray),
        ])

        // Step 7: Run inference
        let output = try model.prediction(from: input)

        // Step 8: Robustly select audio output
        // Prefer named 'audio', else first float MLMultiArray
        let audioOutput: MLMultiArray = {
            if let arr = output.featureValue(for: "audio")?.multiArrayValue { return arr }
            // Fallback: search first float array
            for name in output.featureNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue,
                    arr.dataType == .float32 || arr.dataType == .float16
                {
                    return arr
                }
            }
            return MLMultiArray()
        }()
        if audioOutput.count == 0 {
            throw TTSError.processingFailed(
                "No float audio output found in model outputs: \(Array(output.featureNames))")
        }

        logger.info("Audio output shape: \(audioOutput.shape), count: \(audioOutput.count)")

        // Optional trim using audio_length_samples if provided
        var effectiveCount = audioOutput.count
        if let lenFV = output.featureValue(for: "audio_length_samples") {
            if let la = lenFV.multiArrayValue, la.count > 0 {
                effectiveCount = max(0, la[0].intValue)
            } else if lenFV.type == .int64 {
                effectiveCount = max(0, Int(lenFV.int64Value))
            } else if lenFV.type == .double {
                effectiveCount = max(0, Int(lenFV.doubleValue))
            }
            effectiveCount = min(effectiveCount, audioOutput.count)
        }

        // Step 9: Convert to audio data (normalize like main.py)
        var audioSamples: [Float] = []
        var maxVal: Float = 0.0

        // First pass: find max for normalization
        for i in 0..<effectiveCount {
            let val = audioOutput[i].floatValue
            maxVal = max(maxVal, abs(val))
        }

        // Second pass: normalize and collect samples
        for i in 0..<effectiveCount {
            let normalizedSample = audioOutput[i].floatValue / (maxVal + 1e-8)
            audioSamples.append(normalizedSample)
        }

        // Convert to WAV data via shared utility
        let audioData = try AudioWAV.data(from: audioSamples, sampleRate: 24000)

        logger.info("Generated \(audioData.count) bytes of audio")
        return audioData
    }

    // Removed duplicate WAV utility; see AudioWAV in AudioConverter.swift
}

// TTSError is already defined in TtsModels.swift
