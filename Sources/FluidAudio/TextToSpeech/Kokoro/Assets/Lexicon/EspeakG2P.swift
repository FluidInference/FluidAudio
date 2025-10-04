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
    private var currentVoice: String = ""

    private init() {}

    deinit {
        queue.sync {
            if initialized {
                espeak_Terminate()
            }
        }
    }

    static var isAvailable: Bool {
        true
    }

    func phonemize(word: String, espeakVoice: String = "en-us") -> [String]? {
        return queue.sync {
            guard initializeIfNeeded(espeakVoice: espeakVoice) else { return nil }
            return word.withCString { cstr -> [String]? in
                var raw: UnsafeRawPointer? = UnsafeRawPointer(cstr)
                let modeIPA = Int32(espeakPHONEMES_IPA)
                let textmode = Int32(espeakCHARS_AUTO)
                guard let outPtr = espeak_TextToPhonemes(&raw, textmode, modeIPA) else {
                    logger.warning("espeak_TextToPhonemes returned nil for word: \(word)")
                    return nil
                }
                // eSpeak returns a static buffer that doesn't need to be freed
                let s = String(cString: outPtr)
                if s.isEmpty { return nil }
                // Split by whitespace for word-level phonemes, otherwise Unicode scalars for character-level
                if s.contains(where: { $0.isWhitespace }) {
                    return s.split { $0.isWhitespace }.map { String($0) }
                } else {
                    return s.unicodeScalars.map { String($0) }
                }
            }
        }
    }

    private func initializeIfNeeded(espeakVoice: String = "en-us") -> Bool {
        if initialized {
            if espeakVoice != currentVoice {
                let result = espeakVoice.withCString { espeak_SetVoiceByName($0) }
                if result != EE_OK {
                    logger.error("Failed to set voice to \(espeakVoice), error code: \(result)")
                    return false
                }
                currentVoice = espeakVoice
            }
            return true
        }

        guard let dataDir = Self.frameworkBundledDataPath() else {
            logger.warning("eSpeak NG data bundle not found in ESpeakNG.xcframework; disabling G2P")
            return false
        }

        logger.info("Using eSpeak NG data from framework: \(dataDir.path)")
        let rc: Int32 = dataDir.path.withCString { espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, $0, 0) }

        guard rc >= 0 else {
            logger.error("eSpeak NG initialization failed (rc=\(rc))")
            return false
        }
        let voiceResult = espeakVoice.withCString { espeak_SetVoiceByName($0) }
        if voiceResult != EE_OK {
            logger.error("Failed to set initial voice to \(espeakVoice), error code: \(voiceResult)")
            espeak_Terminate()
            return false
        }
        currentVoice = espeakVoice
        initialized = true
        return true
    }

    private static let staticLogger = AppLogger(subsystem: "com.fluidaudio.tts", category: "EspeakG2P")

    private static func frameworkBundledDataPath() -> URL? {
        let logger = staticLogger

        // The espeak-ng-data.bundle should be in the ESpeakNG.framework's Resources
        guard let bundle = Bundle(identifier: "com.kokoro.espeakng") else {
            logger.warning("Could not find ESpeakNG framework bundle (com.kokoro.espeakng)")
            return nil
        }

        guard let bundleURL = bundle.url(forResource: "espeak-ng-data", withExtension: "bundle") else {
            logger.warning("Could not find espeak-ng-data.bundle in ESpeakNG framework")
            return nil
        }

        let dataDir = bundleURL.appendingPathComponent("espeak-ng-data")
        let voicesPath = dataDir.appendingPathComponent("voices")

        if FileManager.default.fileExists(atPath: voicesPath.path) {
            logger.info("Found eSpeak data at: \(dataDir.path)")
            return dataDir
        }

        logger.error("espeak-ng-data.bundle found but voices directory missing")
        return nil
    }

    static func isDataAvailable() -> Bool {
        return frameworkBundledDataPath() != nil
    }
}
