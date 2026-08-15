#if os(macOS)
import FluidAudio
import Foundation

/// Dump LuxTTS G2P output for a sentence corpus as JSONL
/// (`{"text": ..., "phonemes": ..., "ids": [...]}` per line).
///
/// This feeds the espeak-oracle gate in
/// `mobius/models/tts/zipvoice/coreml/g2p/validate.py`:
///
///     swift run fluidaudiocli luxtts-g2p-dump \
///         --corpus corpus_en_1000.txt --tokens tokens.txt --out swift_dump.jsonl
///     python -m coreml.g2p.validate score \
///         --oracle oracle_tokens.jsonl --swift swift_dump.jsonl
enum LuxTtsG2pDumpCommand {

    private static let logger = AppLogger(category: "LuxTtsG2pDump")

    static func run(arguments: [String]) async {
        var corpusPath: String? = nil
        var outPath: String? = nil
        var tokensPath: String? = nil
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--corpus":
                if i + 1 < arguments.count {
                    corpusPath = arguments[i + 1]
                    i += 1
                }
            case "--out":
                if i + 1 < arguments.count {
                    outPath = arguments[i + 1]
                    i += 1
                }
            case "--tokens":
                if i + 1 < arguments.count {
                    tokensPath = arguments[i + 1]
                    i += 1
                }
            case "--help", "-h":
                print(
                    "Usage: fluidaudiocli luxtts-g2p-dump --corpus <sentences.txt> "
                        + "--tokens <tokens.txt> --out <dump.jsonl>")
                return
            default:
                break
            }
            i += 1
        }
        guard let corpusPath, let outPath, let tokensPath else {
            logger.error("luxtts-g2p-dump requires --corpus, --tokens and --out")
            exit(1)
        }
        do {
            let g2p = try LuxTtsG2p()
            let tokenizer = try LuxTtsTokenizer(
                tokensFileURL: URL(fileURLWithPath: tokensPath))

            let corpus = try String(contentsOfFile: corpusPath, encoding: .utf8)
            var lines: [String] = []
            for sentence in corpus.split(separator: "\n", omittingEmptySubsequences: true) {
                let text = sentence.trimmingCharacters(in: .whitespaces)
                if text.isEmpty { continue }
                let phonemes = g2p.phonemize(text: text)
                let ids = tokenizer.tokenIds(phonemes: phonemes)
                let record: [String: Any] = [
                    "text": text, "phonemes": phonemes, "ids": ids,
                ]
                let data = try JSONSerialization.data(withJSONObject: record)
                lines.append(String(data: data, encoding: .utf8)!)
            }
            try (lines.joined(separator: "\n") + "\n")
                .write(toFile: outPath, atomically: true, encoding: .utf8)
            logger.info("Wrote \(lines.count) entries to \(outPath)")
        } catch {
            logger.error("luxtts-g2p-dump failed: \(error)")
            exit(1)
        }
    }
}
#endif
