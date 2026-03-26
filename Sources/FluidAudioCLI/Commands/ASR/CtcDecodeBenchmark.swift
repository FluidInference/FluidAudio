#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

/// Simple CTC decode benchmark comparing greedy vs beam search with optional ARPA LM.
/// Demonstrates practical WER improvements from language model rescoring.
public enum CtcDecodeBenchmark {

    public static func runCLI(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var audioFile: String?
        var ctcModelPath: String?
        var arpaModelPath: String?
        var referenceText: String?
        var ctcVariant: CtcModelVariant = .ctc06b  // 0.6B pure CTC is better for greedy
        var lmWeight: Float = 0.3
        var beamWidth: Int = 100
        var wordBonus: Float = 0.0
        var tokenCandidates: Int = 40

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--audio":
                if i + 1 < arguments.count {
                    audioFile = arguments[i + 1]
                    i += 1
                }
            case "--ctc-model":
                if i + 1 < arguments.count {
                    ctcModelPath = arguments[i + 1]
                    i += 1
                }
            case "--arpa":
                if i + 1 < arguments.count {
                    arpaModelPath = arguments[i + 1]
                    i += 1
                }
            case "--reference":
                if i + 1 < arguments.count {
                    referenceText = arguments[i + 1]
                    i += 1
                }
            case "--ctc-variant":
                if i + 1 < arguments.count {
                    let variant = arguments[i + 1].lowercased()
                    if variant == "06b" || variant == "0.6b" {
                        ctcVariant = .ctc06b
                    } else if variant == "110m" {
                        ctcVariant = .ctc110m
                    }
                    i += 1
                }
            case "--lm-weight":
                if i + 1 < arguments.count {
                    lmWeight = Float(arguments[i + 1]) ?? 0.3
                    i += 1
                }
            case "--beam-width":
                if i + 1 < arguments.count {
                    beamWidth = Int(arguments[i + 1]) ?? 100
                    i += 1
                }
            case "--word-bonus":
                if i + 1 < arguments.count {
                    wordBonus = Float(arguments[i + 1]) ?? 0.0
                    i += 1
                }
            case "--token-candidates":
                if i + 1 < arguments.count {
                    tokenCandidates = Int(arguments[i + 1]) ?? 40
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        // Use defaults if not specified
        if ctcModelPath == nil {
            ctcModelPath = defaultCtcModelPath(for: ctcVariant)
        }

        print(String(repeating: "=", count: 70))
        print("CTC Decode Benchmark: Greedy vs Beam Search")
        print(String(repeating: "=", count: 70))
        print("Audio:        \(audioFile ?? "not specified")")
        print("CTC model:    \(ctcModelPath ?? "not found")")
        print("ARPA LM:      \(arpaModelPath ?? "none")")
        print("CTC variant:  \(ctcVariant.displayName)")
        print("LM weight:    \(lmWeight)")
        print("Beam width:   \(beamWidth)")
        print("Word bonus:   \(wordBonus)")
        print("=" * 70)

        guard let audioPath = audioFile else {
            print("ERROR: --audio <file> required")
            printUsage()
            return
        }

        guard let modelPath = ctcModelPath else {
            print("ERROR: CTC model not found")
            print("💡 Download \(ctcVariant.repo.folderName) model to:")
            print("   ~/Library/Application Support/FluidAudio/Models/\(ctcVariant.repo.folderName)/")
            print("   Or specify: --ctc-model <path>")
            return
        }

        guard FileManager.default.fileExists(atPath: audioPath) else {
            print("ERROR: Audio file not found: \(audioPath)")
            return
        }

        do {
            // Load CTC models
            print("\n📦 Loading CTC models from: \(modelPath)")
            let modelDir = URL(fileURLWithPath: modelPath)
            let ctcModels = try await CtcModels.loadDirect(from: modelDir, variant: ctcVariant)
            print("✅ Loaded CTC vocabulary with \(ctcModels.vocabulary.count) tokens")

            let vocabSize = ctcModels.vocabulary.count
            let blankId = vocabSize

            // Load ARPA model if specified
            let arpaLM: ARPALanguageModel?
            if let arpaPath = arpaModelPath {
                let arpaURL = URL(fileURLWithPath: arpaPath)
                guard FileManager.default.fileExists(atPath: arpaURL.path) else {
                    print("ERROR: ARPA file not found: \(arpaPath)")
                    return
                }
                print("\n📚 Loading ARPA language model: \(arpaPath)")
                let loadedLM = try ARPALanguageModel.load(from: arpaURL)
                print("✅ Loaded LM with \(loadedLM.unigrams.count) unigrams, \(loadedLM.bigrams.count) bigram contexts")
                arpaLM = loadedLM
            } else {
                arpaLM = nil
                print("\n⚠️  No ARPA model specified - beam search will use acoustic scores only")
            }

            // Load audio
            print("\n🎵 Loading audio: \(audioPath)")
            let audioURL = URL(fileURLWithPath: audioPath)
            let audioFile = try AVAudioFile(forReading: audioURL)
            let audioLength = Double(audioFile.length) / audioFile.processingFormat.sampleRate
            print("   Duration: \(String(format: "%.2f", audioLength))s")

            let format = audioFile.processingFormat
            let frameCount = AVAudioFrameCount(audioFile.length)

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                print("ERROR: Failed to create audio buffer")
                return
            }
            try audioFile.read(into: buffer)

            // Resample to 16kHz
            let converter = AudioConverter()
            let samples = try converter.resampleBuffer(buffer)
            print("   Resampled to 16kHz: \(samples.count) samples")

            // Run CTC inference
            print("\n🔬 Running CTC inference...")
            let startInference = Date()
            let encoder = ctcModels.encoder

            // Prepare input features
            let batchSize = 1
            let sequenceLength = samples.count
            let featuresShape = [NSNumber(value: batchSize), NSNumber(value: sequenceLength)]
            let featuresArray = try MLMultiArray(shape: featuresShape, dataType: .float32)
            let featuresPtr = featuresArray.dataPointer.assumingMemoryBound(to: Float32.self)
            for (idx, sample) in samples.enumerated() {
                featuresPtr[idx] = sample
            }

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "audio_signal": MLFeatureValue(multiArray: featuresArray)
            ])

            let output = try await encoder.prediction(from: input)
            let inferenceTime = Date().timeIntervalSince(startInference)

            guard let logProbs = output.featureValue(for: "logits")?.multiArrayValue else {
                print("ERROR: No logits output from CTC model")
                return
            }

            print("✅ CTC inference complete (\(String(format: "%.3f", inferenceTime))s)")
            print("   Output shape: \(logProbs.shape) (batch, time, vocab)")

            // Extract [[Float]] for greedy/beam
            let timeSteps = logProbs.shape[1].intValue
            let vocabDim = logProbs.shape[2].intValue
            let logProbsPtr = logProbs.dataPointer.assumingMemoryBound(to: Float32.self)

            var frames: [[Float]] = []
            frames.reserveCapacity(timeSteps)
            for t in 0..<timeSteps {
                let base = t * vocabDim
                var frame = [Float](repeating: 0, count: vocabDim)
                for v in 0..<vocabDim {
                    frame[v] = logProbsPtr[base + v]
                }
                frames.append(frame)
            }

            print("   Extracted \(frames.count) frames with \(vocabDim) vocab")

            // Decode with different methods
            print("\n" + String(repeating: "=", count: 70))
            print("DECODING RESULTS")
            print(String(repeating: "=", count: 70))

            // 1. Greedy decode
            print("\n1️⃣  Greedy Decode (baseline)")
            let greedyStart = Date()
            let greedyText = ctcGreedyDecode(
                logProbs: frames,
                vocabulary: ctcModels.vocabulary,
                blankId: blankId
            )
            let greedyTime = Date().timeIntervalSince(greedyStart)
            print("   Result: \"\(greedyText)\"")
            print("   Time:   \(String(format: "%.3f", greedyTime))s")

            // 2. Beam search without LM
            print("\n2️⃣  Beam Search (no LM)")
            let beamNoLMStart = Date()
            let beamNoLMText = ctcBeamSearch(
                logProbs: frames,
                vocabulary: ctcModels.vocabulary,
                lm: nil,
                beamWidth: beamWidth,
                blankId: blankId,
                tokenCandidates: tokenCandidates
            )
            let beamNoLMTime = Date().timeIntervalSince(beamNoLMStart)
            print("   Result: \"\(beamNoLMText)\"")
            print("   Time:   \(String(format: "%.3f", beamNoLMTime))s")

            // 3. Beam search with LM
            guard let lm = arpaLM else {
                print("\n⚠️  Skipping LM decode (no ARPA model provided)")
                print("\n" + String(repeating: "=", count: 70))
                print("SUMMARY")
                print(String(repeating: "=", count: 70))
                print("Audio length:       \(String(format: "%.2f", audioLength))s")
                print("Inference time:     \(String(format: "%.3f", inferenceTime))s")
                print("Greedy decode:      \(String(format: "%.3f", greedyTime))s")
                print("Beam decode (no LM):\(String(format: "%.3f", beamNoLMTime))s")
                print(String(repeating: "=", count: 70))
                return
            }

            print("\n3️⃣  Beam Search + ARPA LM")
            let beamLMStart = Date()
            let beamLMText = ctcBeamSearch(
                logProbs: frames,
                vocabulary: ctcModels.vocabulary,
                lm: lm,
                beamWidth: beamWidth,
                lmWeight: lmWeight,
                wordBonus: wordBonus,
                blankId: blankId,
                tokenCandidates: tokenCandidates
            )
            let beamLMTime = Date().timeIntervalSince(beamLMStart)
            print("   Result: \"\(beamLMText)\"")
            print("   Time:   \(String(format: "%.3f", beamLMTime))s")

            // Compare to reference if provided
            guard let reference = referenceText else {
                print("\n" + String(repeating: "=", count: 70))
                print("SUMMARY")
                print(String(repeating: "=", count: 70))
                print("Audio length:       \(String(format: "%.2f", audioLength))s")
                print("Inference time:     \(String(format: "%.3f", inferenceTime))s")
                print("Greedy decode:      \(String(format: "%.3f", greedyTime))s")
                print("Beam decode (no LM):\(String(format: "%.3f", beamNoLMTime))s")
                print("Total RTFx:         \(String(format: "%.2f", audioLength / (inferenceTime + beamNoLMTime)))x")
                print(String(repeating: "=", count: 70))
                return
            }

            print("\n" + String(repeating: "=", count: 70))
            print("WER COMPARISON (reference provided)")
            print(String(repeating: "=", count: 70))
            print("Reference:      \"\(reference)\"")

            let refNorm = TextNormalizer.normalize(reference)
            let greedyNorm = TextNormalizer.normalize(greedyText)
            let beamNoLMNorm = TextNormalizer.normalize(beamNoLMText)
            let beamLMNorm = TextNormalizer.normalize(beamLMText)

            let refWords = refNorm.split(separator: " ").map(String.init)
            let greedyWords = greedyNorm.split(separator: " ").map(String.init)
            let beamNoLMWords = beamNoLMNorm.split(separator: " ").map(String.init)
            let beamLMWords = beamLMNorm.split(separator: " ").map(String.init)

            let greedyWER = calculateWER(reference: refWords, hypothesis: greedyWords)
            let beamNoLMWER = calculateWER(reference: refWords, hypothesis: beamNoLMWords)
            let beamLMWER = calculateWER(reference: refWords, hypothesis: beamLMWords)

            print("\nGreedy:         \(String(format: "%.1f", greedyWER * 100))% WER")
            print("Beam (no LM):   \(String(format: "%.1f", beamNoLMWER * 100))% WER")
            print("Beam + LM:      \(String(format: "%.1f", beamLMWER * 100))% WER ✅")

            let improvement = ((greedyWER - beamLMWER) / greedyWER) * 100
            if improvement > 0 {
                print("\n🎯 LM Improvement: \(String(format: "%.1f", improvement))% reduction in WER")
            }

            print("\n" + String(repeating: "=", count: 70))
            print("SUMMARY")
            print(String(repeating: "=", count: 70))
            print("Audio length:       \(String(format: "%.2f", audioLength))s")
            print("Inference time:     \(String(format: "%.3f", inferenceTime))s")
            print("Greedy decode:      \(String(format: "%.3f", greedyTime))s")
            print("Beam decode (no LM):\(String(format: "%.3f", beamNoLMTime))s")
            if arpaLM != nil {
                print("Total RTFx:         \(String(format: "%.2f", audioLength / (inferenceTime + beamNoLMTime)))x")
            }
            print(String(repeating: "=", count: 70))

        } catch {
            print("ERROR: Benchmark failed: \(error)")
        }
    }

    private static func defaultCtcModelPath(for variant: CtcModelVariant) -> String? {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let modelPath = appSupport.appendingPathComponent("FluidAudio/Models/\(variant.repo.folderName)")
        if FileManager.default.fileExists(atPath: modelPath.path) {
            return modelPath.path
        }
        return nil
    }

    private static func calculateWER(reference: [String], hypothesis: [String]) -> Double {
        if reference.isEmpty {
            return hypothesis.isEmpty ? 0.0 : 1.0
        }

        let m = reference.count
        let n = hypothesis.count
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if reference[i - 1] == hypothesis[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1
                }
            }
        }

        return Double(dp[m][n]) / Double(m)
    }

    private static func printUsage() {
        print(
            """
            CTC Decode Benchmark - Compare greedy vs beam search with ARPA LM

            Usage: fluidaudiocli ctc-decode-benchmark [options]

            Required:
                --audio <file>           Path to audio file (WAV, 16kHz recommended)

            Optional:
                --ctc-model <path>       Path to CTC model directory (auto-detected if in standard location)
                --ctc-variant <var>      CTC model variant: '06b' (default) or '110m'
                --arpa <file>            Path to ARPA language model file
                --reference <text>       Reference text for WER calculation
                --lm-weight <float>      LM scaling factor (alpha, default: 0.3)
                --beam-width <int>       Beam width (default: 100)
                --word-bonus <float>     Per-word insertion bonus (beta, default: 0.0)
                --token-candidates <int> Top-K tokens per frame (default: 40)

            Examples:
                # Compare greedy vs beam (no LM)
                fluidaudiocli ctc-decode-benchmark --audio speech.wav

                # With ARPA language model
                fluidaudiocli ctc-decode-benchmark \\
                    --audio speech.wav \\
                    --arpa medical.arpa \\
                    --reference "patient has diabetes"

                # Tune LM parameters
                fluidaudiocli ctc-decode-benchmark \\
                    --audio speech.wav \\
                    --arpa medical.arpa \\
                    --lm-weight 0.5 \\
                    --beam-width 200

            Default model location:
                ~/Library/Application Support/FluidAudio/Models/parakeet-ctc-0.6b-coreml/

            Creating an ARPA model:
                # Using KenLM
                lmplz -o 2 < corpus.txt > model.arpa

                # Using SRILM
                ngram-count -text corpus.txt -order 2 -arpa model.arpa
            """)
    }
}
#endif
