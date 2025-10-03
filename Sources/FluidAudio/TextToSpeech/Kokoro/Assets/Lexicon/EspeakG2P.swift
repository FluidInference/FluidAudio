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

    private var currentVoice: String = "en-us"

    private func initializeIfNeeded(espeakVoice: String = "en-us") -> Bool {
        if initialized {
            if espeakVoice != currentVoice {
                _ = espeakVoice.withCString { espeak_SetVoiceByName($0) }
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
        _ = espeakVoice.withCString { espeak_SetVoiceByName($0) }
        currentVoice = espeakVoice
        initialized = true
        return true
    }

    private static func frameworkBundledDataPath() -> URL? {
        // ESpeakNG.xcframework has espeak-ng-data.bundle embedded
        guard let frameworkBundle = Bundle(identifier: "com.fluidaudio.ESpeakNG") else {
            return nil
        }

        guard
            let bundleURL = frameworkBundle.url(
                forResource: "espeak-ng-data",
                withExtension: "bundle"
            )
        else {
            return nil
        }

        let dataDir = bundleURL.appendingPathComponent("espeak-ng-data")
        let voicesPath = dataDir.appendingPathComponent("voices")

        guard FileManager.default.fileExists(atPath: voicesPath.path) else {
            return nil
        }

        return dataDir
    }

    static func isDataAvailable() -> Bool {
        return frameworkBundledDataPath() != nil
    }
}
