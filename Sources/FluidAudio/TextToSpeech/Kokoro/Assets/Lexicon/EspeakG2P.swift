import ESpeakNG
import Foundation

/// Thread-safe wrapper around eSpeak NG C API to get IPA phonemes for a word.
/// Uses espeak_TextToPhonemes with IPA mode.
final class EspeakG2P {
    enum EspeakG2PError: Error, LocalizedError {
        case frameworkBundleMissing
        case dataBundleMissing
        case voicesDirectoryMissing
        case initializationFailed(code: Int32)
        case voiceSelectionFailed(voice: String, error: espeak_ERROR)

        var errorDescription: String? {
            switch self {
            case .frameworkBundleMissing:
                return "ESpeakNG.framework is not bundled with this build."
            case .dataBundleMissing:
                return "espeak-ng-data.bundle is missing from the ESpeakNG framework resources."
            case .voicesDirectoryMissing:
                return "eSpeak NG voices directory is missing inside espeak-ng-data.bundle."
            case .initializationFailed(let code):
                return "eSpeak NG initialization failed with status code \(code)."
            case .voiceSelectionFailed(let voice, let error):
                return "Failed to select eSpeak NG voice \(voice) (status code \(error.rawValue))."
            }
        }
    }

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

    func phonemize(word: String, espeakVoice: String = "en-us") throws -> [String]? {
        return try queue.sync {
            try initializeIfNeeded(espeakVoice: espeakVoice)
            return word.withCString { cstr -> [String]? in
                var raw: UnsafeRawPointer? = UnsafeRawPointer(cstr)
                let modeIPA = Int32(espeakPHONEMES_IPA)
                let textmode = Int32(espeakCHARS_AUTO)
                guard let outPtr = espeak_TextToPhonemes(&raw, textmode, modeIPA) else {
                    logger.warning("espeak_TextToPhonemes returned nil for word: \(word)")
                    return nil
                }
                let phonemeString = String(cString: outPtr)
                if phonemeString.isEmpty { return nil }
                if phonemeString.contains(where: { $0.isWhitespace }) {
                    return phonemeString.split { $0.isWhitespace }.map { String($0) }
                } else {
                    return phonemeString.unicodeScalars.map { String($0) }
                }
            }
        }
    }

    private func initializeIfNeeded(espeakVoice: String = "en-us") throws {
        if initialized {
            if espeakVoice != currentVoice {
                let result = espeakVoice.withCString { espeak_SetVoiceByName($0) }
                guard result == EE_OK else {
                    logger.error("Failed to set voice to \(espeakVoice), error code: \(result)")
                    throw EspeakG2PError.voiceSelectionFailed(voice: espeakVoice, error: result)
                }
                currentVoice = espeakVoice
            }
            return
        }

        let dataDir = try Self.ensureResourcesAvailable()
        logger.info("Using eSpeak NG data from framework: \(dataDir.path)")
        let rc: Int32 = dataDir.path.withCString { espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, $0, 0) }

        guard rc >= 0 else {
            logger.error("eSpeak NG initialization failed (rc=\(rc))")
            throw EspeakG2PError.initializationFailed(code: rc)
        }
        let voiceResult = espeakVoice.withCString { espeak_SetVoiceByName($0) }
        guard voiceResult == EE_OK else {
            logger.error("Failed to set initial voice to \(espeakVoice), error code: \(voiceResult)")
            espeak_Terminate()
            throw EspeakG2PError.voiceSelectionFailed(voice: espeakVoice, error: voiceResult)
        }
        currentVoice = espeakVoice
        initialized = true
    }

    private static let staticLogger = AppLogger(subsystem: "com.fluidaudio.tts", category: "EspeakG2P")

    @discardableResult
    static func ensureResourcesAvailable() throws -> URL {
        let url = try frameworkBundledDataPath()
        staticLogger.info("eSpeak NG data directory: \(url.path)")
        return url
    }

    private static func frameworkBundledDataPath() throws -> URL {
        guard let espeakBundle = Bundle(identifier: "com.kokoro.espeakng") else {
            staticLogger.error("ESpeakNG.framework not found; ensure it is embedded with the application.")
            throw EspeakG2PError.frameworkBundleMissing
        }

        guard let bundleURL = espeakBundle.url(forResource: "espeak-ng-data", withExtension: "bundle") else {
            staticLogger.error(
                "espeak-ng-data.bundle missing from ESpeakNG.framework resources at \(espeakBundle.bundlePath)")
            throw EspeakG2PError.dataBundleMissing
        }

        let dataDir = bundleURL.appendingPathComponent("espeak-ng-data")
        let voicesPath = dataDir.appendingPathComponent("voices")

        guard FileManager.default.fileExists(atPath: voicesPath.path) else {
            staticLogger.error("espeak-ng-data.bundle found but voices directory missing at \(voicesPath.path)")
            throw EspeakG2PError.voicesDirectoryMissing
        }

        return dataDir
    }
}
