import FluidAudio
import Foundation
import OSLog

extension KokoroSynthesizer {
    actor LexiconCache {
        private var wordToPhonemes: [String: [String]] = [:]
        private var caseSensitiveWordToPhonemes: [String: [String]] = [:]
        private var isLoaded = false

        struct Metrics: Sendable {
            let entryCount: Int
            let tokenCount: Int
            let characterCount: Int

            var estimatedBytes: Int {
                characterCount * 2
            }
        }

        private struct CachePayload: Codable {
            let lower: [String: [String]]
            let caseSensitive: [String: [String]]
        }

        func ensureLoaded(kokoroDirectory: URL, allowedTokens: Set<String>) async throws {
            if isLoaded && !caseSensitiveWordToPhonemes.isEmpty { return }

            let cacheURL = kokoroDirectory.appendingPathComponent("us_lexicon_cache.json")
            if await loadFromCache(cacheURL, allowedTokens: allowedTokens) {
                return
            }
            throw TTSError.processingFailed("Missing lexicon cache (expected us_lexicon_cache.json)")
        }

        func lexicons() -> (word: [String: [String]], caseSensitive: [String: [String]]) {
            (wordToPhonemes, caseSensitiveWordToPhonemes)
        }

        func metrics() -> Metrics {
            var entryCount = 0
            var tokenCount = 0
            var characterCount = 0

            for (key, value) in wordToPhonemes {
                entryCount += 1
                characterCount += key.utf16.count
                for token in value {
                    tokenCount += 1
                    characterCount += token.utf16.count
                }
            }

            for (key, value) in caseSensitiveWordToPhonemes {
                entryCount += 1
                characterCount += key.utf16.count
                for token in value {
                    tokenCount += 1
                    characterCount += token.utf16.count
                }
            }

            return Metrics(entryCount: entryCount, tokenCount: tokenCount, characterCount: characterCount)
        }

        private func loadFromCache(_ url: URL, allowedTokens: Set<String>) async -> Bool {
            guard FileManager.default.fileExists(atPath: url.path) else { return false }
            do {
                let data = try Data(contentsOf: url)
                let payload = try JSONDecoder().decode(CachePayload.self, from: data)
                let filteredLower = payload.lower.mapValues { $0.filter { allowedTokens.contains($0) } }
                let filteredCase = payload.caseSensitive.mapValues { $0.filter { allowedTokens.contains($0) } }

                guard !filteredLower.isEmpty else { return false }

                wordToPhonemes = filteredLower
                caseSensitiveWordToPhonemes = filteredCase
                isLoaded = true
                KokoroSynthesizer.logger.info("Loaded lexicon cache: \(filteredLower.count) entries")
                return true
            } catch {
                KokoroSynthesizer.logger.warning("Failed to load lexicon cache: \(error.localizedDescription)")
                wordToPhonemes = [:]
                caseSensitiveWordToPhonemes = [:]
                isLoaded = false
                return false
            }
        }

        private func writeCache(payload: CachePayload, to url: URL) async {
            do {
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.sortedKeys]
                let data = try encoder.encode(payload)
                try data.write(to: url, options: [.atomic])
                KokoroSynthesizer.logger.info("Wrote lexicon cache to \(url.path)")
            } catch {
                KokoroSynthesizer.logger.warning("Failed to persist lexicon cache: \(error.localizedDescription)")
            }
        }
    }
}
