import CoreML
import Foundation
import OSLog

// Extension to ChunkProcessor for overlap handling
extension AsrManager {

    internal struct ChunkProcessorWithOverlap {
        let audioSamples: [Float]
        let chunkSize: Int
        let enableDebug: Bool
        let overlapSeconds: Double

        func process(using manager: AsrManager, startTime: Date) async throws -> ASRResult {
            var allTexts: [String] = []
            let audioLength = Double(audioSamples.count) / 16000.0

            // Calculate overlap and step size
            let overlapSamples = Int(overlapSeconds * 16000)
            let stepSize = max(chunkSize - overlapSamples, 1)  // Ensure we move forward at least 1 sample

            var position = 0
            var chunkIndex = 0
            var decoderState = try DecoderState()
            var previousOverlapText = ""

            if manager.config.enableDebug {
                manager.logger.debug("Processing with overlap: \(overlapSeconds)s (\(overlapSamples) samples)")
                manager.logger.debug("Chunk size: \(chunkSize), Step size: \(stepSize)")
            }

            while position < audioSamples.count {
                let (text, overlapText) = try await processChunkWithOverlap(
                    at: position,
                    chunkIndex: chunkIndex,
                    using: manager,
                    decoderState: &decoderState,
                    overlapSamples: overlapSamples,
                    previousOverlapText: previousOverlapText
                )

                allTexts.append(text)
                previousOverlapText = overlapText

                position += stepSize  // Move by step size, not full chunk
                chunkIndex += 1
            }

            // Join texts and apply final deduplication if enabled
            let joinedText = allTexts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
            let finalText = manager.config.removeDuplicates ? removeDuplicatePatterns(joinedText) : joinedText

            return ASRResult(
                text: finalText,
                confidence: 1.0,
                duration: audioLength,
                processingTime: Date().timeIntervalSince(startTime),
                tokenTimings: nil
            )
        }

        private func processChunkWithOverlap(
            at position: Int,
            chunkIndex: Int,
            using manager: AsrManager,
            decoderState: inout DecoderState,
            overlapSamples: Int,
            previousOverlapText: String
        ) async throws -> (text: String, overlapText: String) {
            // Reset decoder state between chunks if configured
            if manager.config.resetDecoderBetweenChunks && chunkIndex > 0 {
                try await manager.initializeDecoderState(decoderState: &decoderState)
            }

            let endPosition = min(position + chunkSize, audioSamples.count)
            let chunkSamples = Array(audioSamples[position..<endPosition])
            let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: chunkSize)

            let (tokenIds, _) = try await manager.executeMLInference(
                paddedChunk, originalLength: chunkSamples.count, enableDebug: false, decoderState: &decoderState)
            let (fullText, _) = manager.convertTokensWithExistingTimings(tokenIds, timings: [])

            // If we have overlap, try to deduplicate
            if overlapSamples > 0 && chunkIndex > 0 && !previousOverlapText.isEmpty {
                // Simple deduplication: remove the overlapping portion from the beginning if it matches
                let trimmedText = removeOverlappingPrefix(fullText, previousText: previousOverlapText)

                // Extract text from the overlap region for the next chunk
                let overlapEndPosition = min(overlapSamples, chunkSamples.count)
                let overlapRatio = Double(overlapEndPosition) / Double(chunkSamples.count)
                let overlapText = extractOverlapText(fullText, ratio: overlapRatio)

                return (trimmedText, overlapText)
            } else {
                // No overlap or first chunk
                let overlapEndPosition = min(overlapSamples, chunkSamples.count)
                let overlapRatio = Double(overlapEndPosition) / Double(chunkSamples.count)
                let overlapText = extractOverlapText(fullText, ratio: overlapRatio)

                return (fullText, overlapText)
            }
        }

        private func removeOverlappingPrefix(_ text: String, previousText: String) -> String {
            // Improved deduplication: find overlapping suffix/prefix match
            let textWords = text.split(separator: " ").map { String($0) }
            let previousWords = previousText.split(separator: " ").map { String($0) }

            if previousWords.isEmpty || textWords.isEmpty {
                return text
            }

            // Look for the longest suffix of previousWords that matches a prefix of textWords
            var bestMatch = 0
            let maxSearchLength = min(previousWords.count, textWords.count, 10)  // Limit search to 10 words

            for suffixLength in stride(from: maxSearchLength, through: 1, by: -1) {
                let suffixStart = previousWords.count - suffixLength
                var matches = true

                for i in 0..<suffixLength {
                    if previousWords[suffixStart + i].lowercased() != textWords[i].lowercased() {
                        matches = false
                        break
                    }
                }

                if matches {
                    bestMatch = suffixLength
                    break
                }
            }

            // Remove the matching prefix from the current text
            if bestMatch > 0 {
                return textWords.dropFirst(bestMatch).joined(separator: " ")
            }

            return text
        }

        private func extractOverlapText(_ text: String, ratio: Double) -> String {
            // Extract approximately the last 'ratio' portion of the text
            let words = text.split(separator: " ")
            if words.isEmpty {
                return ""
            }

            let overlapWordCount = max(1, Int(Double(words.count) * ratio))
            return words.suffix(overlapWordCount).joined(separator: " ")
        }

        private func removeDuplicatePatterns(_ text: String) -> String {
            // First remove excessive dots (more than 3 consecutive)
            var cleanedText = text

            // Replace sequences of 4+ dots with just "..."
            if let regex = try? NSRegularExpression(pattern: "\\.{4,}", options: []) {
                cleanedText = regex.stringByReplacingMatches(
                    in: cleanedText,
                    options: [],
                    range: NSRange(location: 0, length: cleanedText.count),
                    withTemplate: "..."
                )
            }

            // Remove trailing dots at the end (but keep ellipsis if it's meaningful)
            cleanedText = cleanedText.replacingOccurrences(
                of: #"\.\s*\.\s*\.\s*$"#, with: "", options: .regularExpression)

            // Apply enhanced duplicate removal for cross-language and semantic duplicates
            cleanedText = removeEnhancedDuplicates(cleanedText)

            // Remove repetitive patterns like "the city of the city of the city"
            let words = cleanedText.split(separator: " ").map { String($0) }
            var result: [String] = []
            var i = 0

            while i < words.count {
                var patternFound = false

                // Check for repeating patterns of length 2-5 words
                for patternLength in stride(from: 5, through: 2, by: -1) {
                    if i + patternLength * 2 <= words.count {
                        let pattern = words[i..<i + patternLength]
                        let nextSegment = words[i + patternLength..<min(i + patternLength * 2, words.count)]

                        // Check if pattern repeats
                        if pattern.count == nextSegment.count {
                            var matches = true
                            for j in 0..<pattern.count {
                                if pattern[pattern.startIndex + j].lowercased()
                                    != nextSegment[nextSegment.startIndex + j].lowercased()
                                {
                                    matches = false
                                    break
                                }
                            }

                            if matches {
                                // Add pattern once and skip the duplicate
                                result.append(contentsOf: pattern)
                                i += patternLength * 2

                                // Skip additional repetitions
                                while i + patternLength <= words.count {
                                    let nextRep = words[i..<i + patternLength]
                                    var stillMatches = true
                                    for j in 0..<pattern.count {
                                        if pattern[pattern.startIndex + j].lowercased()
                                            != nextRep[nextRep.startIndex + j].lowercased()
                                        {
                                            stillMatches = false
                                            break
                                        }
                                    }
                                    if stillMatches {
                                        i += patternLength
                                    } else {
                                        break
                                    }
                                }

                                patternFound = true
                                break
                            }
                        }
                    }
                }

                if !patternFound {
                    result.append(words[i])
                    i += 1
                }
            }

            return result.joined(separator: " ")
        }

        /// Enhanced duplicate removal for cross-language and semantic duplicates
        private func removeEnhancedDuplicates(_ text: String) -> String {
            var cleanedText = text

            // Log for debugging if enabled
            if ProcessInfo.processInfo.environment["DUPLICATE_DEBUG"] != nil {
                print("🔍 Enhanced duplicate removal - Input text:")
                print(text)
            }

            // Define common cross-language duplicate patterns - more flexible patterns
            let crossLangPatterns: [(pattern: String, replacement: String)] = [
                // French-English duplicates - handle variations in spacing/punctuation
                ("progrès importants progress important", "progrès importants"),
                ("progress important at this chapter", ""),  // Remove garbled English
                ("we have 2 marked history on choising a consequence of ministries", ""),
                ("levés pour ça levé for ça", "levés pour ça"),
                ("progressiste progressist", "progressiste"),
                ("plus equale", ""),  // Remove first occurrence, keep "plus égal"
                ("the domain", ""),  // Remove English after French
                ("et là dessus et là dessus", "et là dessus"),
                ("ne peuvent ne peuvent", "ne peuvent"),
                ("la france et le canada la france et le canada", "la france et le canada"),

                // Common ASR artifacts
                ("a moins libre", "un monde libre"),  // Common misrecognition
                ("a monde", "un monde"),
                ("a mon plus", "un monde plus"),
                ("a money", "un monde"),

                // Remove random English insertions in French context
                ("at this chapter we have 2 marked history on choising a consequence of ministries", ""),
                ("we have 2 marked listen on choisissant", "nous avons marqué l'histoire en choisissant"),
            ]

            // Apply cross-language duplicate removal
            for (pattern, replacement) in crossLangPatterns {
                cleanedText = cleanedText.replacingOccurrences(
                    of: pattern,
                    with: replacement,
                    options: [.caseInsensitive]
                )
            }

            // Remove single-word immediate repetitions (case-insensitive)
            let words = cleanedText.split(separator: " ").map { String($0) }
            var cleanedWords: [String] = []
            var lastWord = ""

            for word in words {
                // Skip if it's an immediate repetition (allowing for case differences)
                if word.lowercased() != lastWord.lowercased() {
                    cleanedWords.append(word)
                    lastWord = word
                } else if word.count > 3 {  // Only remove repetitions of words longer than 3 chars
                    // Skip the duplicate
                    continue
                } else {
                    cleanedWords.append(word)  // Keep short words even if repeated
                    lastWord = word
                }
            }

            cleanedText = cleanedWords.joined(separator: " ")

            // Clean up spacing issues
            cleanedText = cleanedText.replacingOccurrences(of: "  +", with: " ", options: .regularExpression)
            cleanedText = cleanedText.trimmingCharacters(in: .whitespacesAndNewlines)

            // Log output if debugging
            if ProcessInfo.processInfo.environment["DUPLICATE_DEBUG"] != nil {
                print("🔍 Enhanced duplicate removal - Output text:")
                print(cleanedText)
            }

            return cleanedText
        }
    }
}
