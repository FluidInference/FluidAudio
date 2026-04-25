import Foundation

extension PocketTtsSynthesizer {

    /// Result of a PocketTTS synthesis operation.
    public struct SynthesisResult: Sendable {
        /// WAV audio data (24kHz, 16-bit mono).
        public let audio: Data
        /// Raw Float32 audio samples.
        public let samples: [Float]
        /// Number of 80ms frames generated.
        public let frameCount: Int
        /// Generation step at which EOS was detected (nil if max length reached).
        public let eosStep: Int?
    }

    /// CoreML output key names for the conditioning and generation step models
    /// are discovered at model-load time via `PocketTtsLayerKeys.discover(...)`.
    /// Mimi decoder I/O is discovered at model-load via `PocketTtsMimiKeys.discover(...)`.
    /// Both use discovery because CoreML auto-generates `var_NNN` names that
    /// differ between English/v2 packs and 6L/24L variants.
}
