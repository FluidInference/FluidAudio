import FluidAudio
import Foundation

extension KokoroSynthesizer {
    public struct TokenCapacities {
        public let short: Int
        public let long: Int

        public func capacity(for variant: ModelNames.TTS.Variant) -> Int {
            switch variant {
            case .fiveSecond:
                return short
            case .fifteenSecond:
                return long
            }
        }
    }

    public struct SynthesisResult: Sendable {
        public let audio: Data
        public let chunks: [ChunkInfo]
        public let diagnostics: Diagnostics?

        public init(audio: Data, chunks: [ChunkInfo], diagnostics: Diagnostics? = nil) {
            self.audio = audio
            self.chunks = chunks
            self.diagnostics = diagnostics
        }
    }

    public struct Diagnostics: Sendable {
        public let variantFootprints: [ModelNames.TTS.Variant: Int]
        public let lexiconEntryCount: Int
        public let lexiconEstimatedBytes: Int
        public let audioSampleBytes: Int
        public let outputWavBytes: Int

        public func updating(audioSampleBytes: Int, outputWavBytes: Int) -> Diagnostics {
            Diagnostics(
                variantFootprints: variantFootprints,
                lexiconEntryCount: lexiconEntryCount,
                lexiconEstimatedBytes: lexiconEstimatedBytes,
                audioSampleBytes: audioSampleBytes,
                outputWavBytes: outputWavBytes
            )
        }
    }

    public struct ChunkInfo: Sendable {
        public let index: Int
        public let text: String
        public let wordCount: Int
        public let words: [String]
        public let atoms: [String]
        public let pauseAfterMs: Int
        public let tokenCount: Int
        public let samples: [Float]
        public let variant: ModelNames.TTS.Variant

        public init(
            index: Int,
            text: String,
            wordCount: Int,
            words: [String],
            atoms: [String],
            pauseAfterMs: Int,
            tokenCount: Int,
            samples: [Float],
            variant: ModelNames.TTS.Variant
        ) {
            self.index = index
            self.text = text
            self.wordCount = wordCount
            self.words = words
            self.atoms = atoms
            self.pauseAfterMs = pauseAfterMs
            self.tokenCount = tokenCount
            self.samples = samples
            self.variant = variant
        }
    }

    struct ChunkInfoTemplate: Sendable {
        let index: Int
        let text: String
        let wordCount: Int
        let words: [String]
        let atoms: [String]
        let pauseAfterMs: Int
        let tokenCount: Int
        let variant: ModelNames.TTS.Variant
        let targetTokens: Int
    }

    struct ChunkEntry: Sendable {
        let chunk: TextChunk
        let inputIds: [Int32]
        let template: ChunkInfoTemplate
    }
}
