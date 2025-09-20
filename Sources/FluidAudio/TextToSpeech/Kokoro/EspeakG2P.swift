#if canImport(ESpeakNG) || canImport(CEspeakNG)

#if canImport(ESpeakNG)
import ESpeakNG
#else
import CEspeakNG
#endif
import Foundation

/// Thread-safe wrapper around eSpeak NG C API to get IPA phonemes for a word.
/// Uses espeak_TextToPhonemes with IPA mode.
@available(macOS 13.0, iOS 16.0, *)
final class EspeakG2P {
    static let shared = EspeakG2P()
    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "EspeakG2P")
    private let queue = DispatchQueue(label: "com.fluidaudio.tts.espeak.g2p")
    private var initialized = false

    private init() {
        _ = initializeIfNeeded()
    }

    private func initializeIfNeeded() -> Bool {
        if initialized { return true }

        // Try to find bundled data path
        let dataPath = EspeakG2P.findEspeakDataPath()

        // Only initialize if a valid data path is present. Some builds of eSpeak NG
        // crash when initialized without an explicit data directory. We prefer to
        // disable G2P gracefully rather than risking a segfault.
        guard let dataPath = dataPath else {
            logger.warning("eSpeak NG data not found; disabling G2P fallback")
            return false
        }

        logger.info("Using eSpeak NG data from: \(dataPath)")
        let rc: Int32 = dataPath.withCString { espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, $0, 0) }

        guard rc >= 0 else {
            logger.error("eSpeak NG initialization failed (rc=\(rc))")
            return false
        }
        _ = "en-us".withCString { espeak_SetVoiceByName($0) }
        initialized = true
        return true
    }

    static func findEspeakDataPath() -> String? {
        let fm = FileManager.default

        func valid(_ path: String) -> String? {
            let voices = URL(fileURLWithPath: path).appendingPathComponent("voices").path
            return fm.fileExists(atPath: voices) ? path : nil
        }

        // 1. Check the embedded ESpeakNG.xcframework bundle first
        #if canImport(ESpeakNG)
        // Look for the framework in standard locations
        let frameworkPaths = [
            // Local project framework
            "./Frameworks/ESpeakNG.xcframework/macos-arm64/ESpeakNG.framework/Resources/espeak-ng-data.bundle/espeak-ng-data",
            "./Frameworks/ESpeakNG.xcframework/macos-arm64/ESpeakNG.framework/Versions/A/Resources/espeak-ng-data.bundle/espeak-ng-data",
            "./Frameworks/ESpeakNG.xcframework/macos-arm64/ESpeakNG.framework/Versions/Current/Resources/espeak-ng-data.bundle/espeak-ng-data",
        ]

        // Convert relative paths to absolute
        let cwd = fm.currentDirectoryPath
        for relativePath in frameworkPaths {
            let absolutePath = URL(fileURLWithPath: cwd).appendingPathComponent(relativePath).path
            if let ok = valid(absolutePath) { return ok }
        }

        // Try Bundle lookup
        if let bundle = Bundle(identifier: "org.espeak-ng.ESpeakNG") {
            let bundleDataPath = bundle.bundlePath
                .appending("/Resources/espeak-ng-data.bundle/espeak-ng-data")
            if let ok = valid(bundleDataPath) { return ok }

            // Also check Versions/A path structure
            let versionedPath = bundle.bundlePath
                .appending("/Versions/A/Resources/espeak-ng-data.bundle/espeak-ng-data")
            if let ok = valid(versionedPath) { return ok }
        }
        #endif

        // 2. Fallback to local Resources directory if it exists
        let localResources = URL(fileURLWithPath: fm.currentDirectoryPath)
            .appendingPathComponent("Resources/espeak-ng-data")
        if let ok = valid(localResources.path) { return ok }

        // 3. Finally check the models cache (download from HF)
        if let base = try? TtsModels.cacheDirectoryURL() {
            let models = base.appendingPathComponent("Models/kokoro/Resources/espeak-ng/espeak-ng-data.bundle/espeak-ng-data")
            if let ok = valid(models.path) { return ok }
        }

        // Nothing found
        return nil
    }

    /// Whether eSpeak NG data is available in the embedded framework bundle.
    static func isDataAvailable() -> Bool {
        return findEspeakDataPath() != nil
    }

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
}
#endif
