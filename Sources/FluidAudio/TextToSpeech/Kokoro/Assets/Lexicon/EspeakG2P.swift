#if canImport(ESpeakNG)

import ESpeakNG
import Foundation

/// Thread-safe wrapper around eSpeak NG C API to get IPA phonemes for a word.
/// Uses espeak_TextToPhonemes with IPA mode.
@available(macOS 13.0, iOS 16.0, *)
final class EspeakG2P {
    static let shared = EspeakG2P()
    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "EspeakG2P")
    private let queue = DispatchQueue(label: "com.fluidaudio.tts.espeak.g2p")
    private var initialized = false

    private init() {}

    /// Return IPA tokens for a word, or nil on failure.
    func phonemize(word: String) -> [String]? {
        return queue.sync {
            guard initializeIfNeeded() else { return nil }
            return word.withCString { cstr -> [String]? in
                var raw: UnsafeRawPointer? = UnsafeRawPointer(cstr)
                let modeIPA = Int32(espeakPHONEMES_IPA)
                let textmode = Int32(espeakCHARS_AUTO)
                guard let outPtr = espeak_TextToPhonemes(&raw, textmode, modeIPA) else {
                    logger.warning("espeak_TextToPhonemes returned nil for word: \(word)")
                    return nil
                }
                let s = String(cString: outPtr)
                if s.isEmpty { return nil }
                if s.contains(where: { $0.isWhitespace }) {
                    return s.split { $0.isWhitespace }.map { String($0) }
                } else {
                    return s.unicodeScalars.map { String($0) }
                }
            }
        }
    }

    private func initializeIfNeeded() -> Bool {
        if initialized { return true }

        guard let base = try? TtsModels.cacheDirectoryURL() else {
            logger.warning("Unable to resolve TTS cache directory; disabling eSpeak G2P")
            return false
        }

        let dataDir =
            base
            .appendingPathComponent("Models/kokoro/Resources/espeak-ng/espeak-ng-data.bundle/espeak-ng-data")

        let voicesPath = dataDir.appendingPathComponent("voices")
        guard FileManager.default.fileExists(atPath: voicesPath.path) else {
            logger.warning("eSpeak NG voices directory not found at \(dataDir.path); disabling G2P")
            return false
        }

        logger.info("Using eSpeak NG data from: \(dataDir.path)")
        let rc: Int32 = dataDir.path.withCString { espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, $0, 0) }

        guard rc >= 0 else {
            logger.error("eSpeak NG initialization failed (rc=\(rc))")
            return false
        }
        _ = "en-us".withCString { espeak_SetVoiceByName($0) }
        initialized = true
        return true
    }

    static func isDataAvailable() -> Bool {
        guard let base = try? TtsModels.cacheDirectoryURL() else { return false }
        let voices =
            base
            .appendingPathComponent("Models/kokoro/Resources/espeak-ng/espeak-ng-data.bundle/espeak-ng-data/voices")
        return FileManager.default.fileExists(atPath: voices.path)
    }
}
#endif
