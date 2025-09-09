import CoreML
import Foundation
import OSLog

/// Direct TTS implementation using pre-computed phoneme dictionary and MLPackage model
/// This follows the main.py approach - no pipeline, just dictionary lookup + model inference
@available(macOS 13.0, iOS 16.0, *)
public struct Kokoro {
    private static let logger = Logger(subsystem: "com.fluidaudio.tts", category: "KokoroDirect")

    // Phoneme dictionary loaded from kokoro_word_phonemes_full.json
    private static var phonemeDictionary: [String: [String]] = [:]
    private static var isDictionaryLoaded = false

    // Voice embeddings loaded from JSON
    private static var voiceEmbeddings: [String: [Int: [Float]]] = [:]

    /// Load the comprehensive phoneme dictionary
    public static func loadPhonemeDictionary() throws {
        guard !isDictionaryLoaded else { return }

        let currentDir = FileManager.default.currentDirectoryPath
        let dictURL = URL(fileURLWithPath: currentDir).appendingPathComponent("word_phonemes.json")

        guard FileManager.default.fileExists(atPath: dictURL.path) else {
            throw TTSError.modelNotFound("Phoneme dictionary not found at \(dictURL.path)")
        }

        let data = try Data(contentsOf: dictURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        if let wordToPhonemes = json["word_to_phonemes"] as? [String: [String]] {
            phonemeDictionary = wordToPhonemes
            isDictionaryLoaded = true
            logger.info("Loaded \(phonemeDictionary.count) words from phoneme dictionary")
        } else {
            throw TTSError.processingFailed("Invalid phoneme dictionary format")
        }
    }

    /// Convert text to phonemes using dictionary lookup (like main.py)
    public static func textToPhonemes(_ text: String) throws -> [String] {
        try loadPhonemeDictionary()

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
    public static func loadVoiceEmbedding(voice: String = "af_heart", phonemeCount: Int) throws -> [Float] {
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
                    voiceEmbeddings[voiceName] = voiceEmbeddingsByLength
                }
                logger.info("Loaded voice embeddings for \(voiceEmbeddings.count) voices")
            }
        }

        // Get embedding for this phoneme count
        let embeddingIndex = phonemeCount - 1  // Same as Python: len(ps) - 1

        if let voiceData = voiceEmbeddings[voice],
            let embedding = voiceData[phonemeCount]
        {
            logger.info("Using embedding at index \(embeddingIndex) for \(phonemeCount) phonemes")
            print("DEBUG: Voice embedding first 5 values: \(embedding.prefix(5))")
            return embedding
        } else {
            logger.warning("No embedding found for \(voice) with \(phonemeCount) phonemes, using zeros")
            print("WARNING: Using zero embeddings! Voice: \(voice), phoneme count: \(phonemeCount)")
            print("DEBUG: Available voices: \(voiceEmbeddings.keys)")
            if let voiceData = voiceEmbeddings[voice] {
                print("DEBUG: Available lengths for \(voice): \(voiceData.keys.sorted())")
            }
            return Array(repeating: 0.0, count: 256)
        }
    }

    /// Main synthesis function (following main.py logic)
    public static func synthesize(text: String, voice: String = "af_heart") throws -> Data {
        logger.info("Synthesizing: '\(text)' with voice: \(voice)")

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
        let voiceEmbedding = try loadVoiceEmbedding(voice: voice, phonemeCount: phonemes.count)

        // Step 5: Create model inputs
        let inputShape = [1, 249] as [NSNumber]
        let inputArray = try MLMultiArray(shape: inputShape, dataType: .int32)
        for (i, id) in inputIds.enumerated() {
            inputArray[i] = NSNumber(value: id)
        }

        let refShape = [1, 256] as [NSNumber]
        let refArray = try MLMultiArray(shape: refShape, dataType: .float32)
        for (i, val) in voiceEmbedding.enumerated() {
            refArray[i] = NSNumber(value: val)
        }

        // Fixed phases (all zeros, matching main.py)
        let phasesShape = [1, 9] as [NSNumber]
        let phasesArray = try MLMultiArray(shape: phasesShape, dataType: .float32)
        for i in 0..<9 {
            phasesArray[i] = NSNumber(value: Float(0.0))
        }

        logger.info(
            "Input shapes - input_ids: \(inputArray.shape), ref_s: \(refArray.shape), phases: \(phasesArray.shape)")
        logger.debug("First 10 input_ids: \(inputIds.prefix(10))")
        logger.debug("First 5 ref_s values: \(voiceEmbedding.prefix(5))")

        // Step 6: Load and run model - try compiled version first
        let compiledModelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(
            "kokoro.mlmodelc")
        let packageModelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(
            "kokoro.mlpackage")

        let modelURL =
            FileManager.default.fileExists(atPath: compiledModelURL.path) ? compiledModelURL : packageModelURL

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

        // Step 8: Process output (var_7495)
        guard let audioOutput = output.featureValue(for: "var_7495")?.multiArrayValue else {
            throw TTSError.processingFailed("No audio output from model")
        }

        logger.info("Audio output shape: \(audioOutput.shape), count: \(audioOutput.count)")

        // Step 9: Convert to audio data (normalize like main.py)
        var audioSamples: [Float] = []
        var maxVal: Float = 0.0

        // First pass: find max for normalization
        for i in 0..<audioOutput.count {
            let val = audioOutput[i].floatValue
            maxVal = max(maxVal, abs(val))
        }

        // Second pass: normalize and collect samples
        for i in 0..<audioOutput.count {
            let normalizedSample = audioOutput[i].floatValue / (maxVal + 1e-8)
            audioSamples.append(normalizedSample)
        }

        // Convert to WAV data
        let audioData = try createWAVData(samples: audioSamples, sampleRate: 24000)

        logger.info("Generated \(audioData.count) bytes of audio")
        return audioData
    }

    /// Create WAV file data from samples
    private static func createWAVData(samples: [Float], sampleRate: Int) throws -> Data {
        var data = Data()

        // WAV header
        let numChannels: Int16 = 1
        let bitsPerSample: Int16 = 16
        let byteRate = Int32(sampleRate) * Int32(numChannels) * Int32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = Int32(samples.count * 2)  // 16-bit samples
        let fileSize = dataSize + 36

        // RIFF header
        data.append("RIFF".data(using: .ascii)!)
        data.append(withUnsafeBytes(of: fileSize.littleEndian) { Data($0) })
        data.append("WAVE".data(using: .ascii)!)

        // fmt chunk
        data.append("fmt ".data(using: .ascii)!)
        data.append(withUnsafeBytes(of: Int32(16).littleEndian) { Data($0) })  // Chunk size
        data.append(withUnsafeBytes(of: Int16(1).littleEndian) { Data($0) })  // PCM format
        data.append(withUnsafeBytes(of: numChannels.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: Int32(sampleRate).littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: byteRate.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: blockAlign.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: bitsPerSample.littleEndian) { Data($0) })

        // data chunk
        data.append("data".data(using: .ascii)!)
        data.append(withUnsafeBytes(of: dataSize.littleEndian) { Data($0) })

        // Convert float samples to 16-bit PCM
        for sample in samples {
            let clipped = max(-1.0, min(1.0, sample))
            let intSample = Int16(clipped * 32767)
            data.append(withUnsafeBytes(of: intSample.littleEndian) { Data($0) })
        }

        return data
    }
}

// TTSError is already defined in TtsModels.swift
