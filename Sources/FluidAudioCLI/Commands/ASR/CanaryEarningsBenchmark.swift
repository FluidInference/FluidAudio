#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// `canary-earnings-benchmark [--max-files N] [--no-vocab] [--min-similarity F]`
///
/// Canary-1B-v2 base transcription + CTC-spotter custom-vocabulary boosting on the
/// Earnings22-kws dataset. Mirrors the metrics of `ctc-earnings-benchmark`
/// (macro WER + dictionary recall) so the two engines can be compared directly.
enum CanaryEarningsBenchmark {
    private static let logger = AppLogger(category: "CanaryEarningsBenchmark")

    static func run(arguments: [String]) async {
        var maxFiles = Int.max
        var useVocab = true
        var minSimilarity: Float = 0.60
        var insertOnMiss = true
        var insertScore: Float = -6.0
        var dataDir: String?

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--max-files":
                i += 1
                if i < arguments.count { maxFiles = Int(arguments[i]) ?? .max }
            case "--no-vocab": useVocab = false
            case "--min-similarity":
                i += 1
                if i < arguments.count { minSimilarity = Float(arguments[i]) ?? minSimilarity }
            case "--no-insert": insertOnMiss = false
            case "--insert-score":
                i += 1
                if i < arguments.count { insertScore = Float(arguments[i]) ?? insertScore }
            case "--data-dir":
                i += 1
                if i < arguments.count { dataDir = arguments[i] }
            case "--help", "-h":
                print(
                    """
                    Usage: fluidaudio canary-earnings-benchmark [options]
                      --max-files N        limit files
                      --no-vocab           canary only (no keyword boosting) baseline
                      --min-similarity F   fuzzy-replace threshold (default 0.60)
                      --data-dir PATH      earnings22-kws test-dataset dir
                    """)
                return
            default: break
            }
            i += 1
        }

        let root =
            dataDir.map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Application Support/FluidAudio/earnings22-kws/test-dataset")
        guard FileManager.default.fileExists(atPath: root.path) else {
            logger.error("earnings22-kws not found at \(root.path); run: fluidaudio download --dataset earnings22-kws")
            return
        }

        do {
            logger.info("Loading canary + CTC keyword booster...")
            let canary = try await CanaryManager.load(precision: .int4)
            let booster =
                useVocab
                ? try await CanaryKeywordBooster.load(
                    minSimilarity: minSimilarity, insertOnMiss: insertOnMiss, insertScoreFloor: insertScore)
                : nil
            let converter = AudioConverter(sampleRate: 16000)

            let wavs =
                (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil))?
                .filter { $0.pathExtension.lowercased() == "wav" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []
            let files = Array(wavs.prefix(maxFiles))
            guard !files.isEmpty else {
                logger.error("no wavs under \(root.path)")
                return
            }
            logger.info("Canary earnings benchmark on \(files.count) files (vocab: \(useVocab))...")

            var werBaseSum = 0.0
            var werBoostSum = 0.0
            var scored = 0
            var dictFound = 0
            var dictTotal = 0
            // OpenBench-style keyword metric (same whole-word definition as ctc-earnings-benchmark)
            var tp = 0
            var fp = 0
            var fn = 0
            var termsApplied = 0
            var audioTotal = 0.0
            var computeTotal = 0.0

            for (idx, wav) in files.enumerated() {
                let stem = wav.deletingPathExtension()
                let refURL = stem.appendingPathExtension("text").appendingPathExtension("txt")
                let dictURL = stem.appendingPathExtension("dictionary").appendingPathExtension("txt")
                guard
                    let ref = (try? String(contentsOf: refURL, encoding: .utf8))?.trimmingCharacters(
                        in: .whitespacesAndNewlines), !ref.isEmpty
                else { continue }

                do {
                    let samples = try converter.resampleAudioFile(wav)
                    let start = Date()
                    let baseText = try await canary.transcribe(audio: samples)

                    var finalText = baseText
                    var dictTerms: [String] = []
                    if useVocab, let booster, FileManager.default.fileExists(atPath: dictURL.path) {
                        let vocab = try CustomVocabularyContext.loadFromSimpleFormat(from: dictURL)
                        dictTerms = vocab.terms.map { $0.text }
                        let r = try await booster.boost(transcript: baseText, audioSamples: samples, vocabulary: vocab)
                        finalText = r.text
                        termsApplied += r.applied.count
                    }
                    computeTotal += Date().timeIntervalSince(start)
                    audioTotal += Double(samples.count) / 16000.0

                    let wBase = WERCalculator.calculateWERMetrics(hypothesis: baseText, reference: ref).wer
                    let wBoost = WERCalculator.calculateWERMetrics(hypothesis: finalText, reference: ref).wer
                    werBaseSum += wBase
                    werBoostSum += wBoost
                    scored += 1

                    if !dictTerms.isEmpty {
                        // Strict, engine-agnostic keyword scoring (whole-word match in
                        // normalized text) — identical definition to ctc-earnings-benchmark
                        // and OpenBench: TP = in ref & hyp, FP = in hyp not ref, FN = in ref not hyp.
                        let refLower = TextNormalizer.normalize(ref).lowercased()
                        let hypLower = TextNormalizer.normalize(finalText).lowercased()
                        let hayLower = finalText.lowercased()
                        for term in dictTerms {
                            dictTotal += 1
                            if hayLower.contains(term.lowercased()) { dictFound += 1 }  // loose recall (substring)
                            let tl = term.lowercased()
                            let inRef = containsWholeWord(tl, in: refLower)
                            let inHyp = containsWholeWord(tl, in: hypLower)
                            if inRef && inHyp {
                                tp += 1
                            } else if inHyp && !inRef {
                                fp += 1
                            } else if inRef && !inHyp {
                                fn += 1
                            }
                        }
                    }

                    if (idx + 1) % 25 == 0 || idx + 1 == files.count {
                        logger.info("  \(idx + 1)/\(files.count) done")
                    }
                } catch {
                    logger.error("  failed \(wav.lastPathComponent): \(error)")
                }
            }

            let rtfx = computeTotal > 0 ? audioTotal / computeTotal : 0
            let werBase = scored > 0 ? werBaseSum / Double(scored) * 100 : -1
            let werBoost = scored > 0 ? werBoostSum / Double(scored) * 100 : -1
            let recall = dictTotal > 0 ? Double(dictFound) / Double(dictTotal) * 100 : -1
            print("")
            print("===== Canary earnings benchmark =====")
            print("Files scored     : \(scored)")
            print("RTFx             : \(String(format: "%.2f", rtfx))x")
            print("WER canary-only  : \(String(format: "%.2f", werBase))%")
            if useVocab {
                let precision = (tp + fp) > 0 ? Double(tp) / Double(tp + fp) : 0
                let kwRecall = (tp + fn) > 0 ? Double(tp) / Double(tp + fn) : 0
                let f1 = (precision + kwRecall) > 0 ? 2 * precision * kwRecall / (precision + kwRecall) : 0
                print("WER canary+vocab : \(String(format: "%.2f", werBoost))%")
                print("Terms applied    : \(termsApplied)")
                print("Dict recall(loose): \(dictFound)/\(dictTotal) (\(String(format: "%.1f", recall))%)")
                print(
                    "Keyword Recall   : \(String(format: "%.3f", kwRecall)) (TP=\(tp), FN=\(fn))")
                print(
                    "Keyword Precision: \(String(format: "%.3f", precision)) (TP=\(tp), FP=\(fp))")
                print("Keyword F1       : \(String(format: "%.3f", f1))")
            }
            print("=====================================")
        } catch {
            logger.error("Benchmark failed: \(error)")
        }
    }

    /// Whole-word (\\b…\\b) case-insensitive match, matching the CTC benchmark's
    /// keyword scoring so canary and ctc numbers are directly comparable.
    private static func containsWholeWord(_ word: String, in text: String) -> Bool {
        let pattern = "\\b\(NSRegularExpression.escapedPattern(for: word))\\b"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text.contains(word) }
        return regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }
}
#endif
