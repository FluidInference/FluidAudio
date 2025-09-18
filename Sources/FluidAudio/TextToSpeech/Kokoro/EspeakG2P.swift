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
    private var currentVoice: String?
    private var availableVoicesByIdentifier: [String: String] = [:]
    private var voicesByLanguage: [String: [String]] = [:]

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
        populateVoiceCache()
        return true
    }

    private func populateVoiceCache() {
        availableVoicesByIdentifier.removeAll()
        voicesByLanguage.removeAll()

        guard let voiceList = espeak_ListVoices(nil) else { return }

        var index = 0
        while let voicePointer = voiceList.advanced(by: index).pointee {
            let voice = voicePointer.pointee

            if let identifierPtr = voice.identifier {
                let identifier = String(cString: identifierPtr)
                let canonical = identifier.trimmingCharacters(in: .whitespacesAndNewlines)
                if !canonical.isEmpty {
                    availableVoicesByIdentifier[canonical.lowercased()] = canonical
                }

                if let languagesPtr = voice.languages {
                    let rawLanguages = String(cString: languagesPtr)
                    let cleaned =
                        rawLanguages
                        .replacingOccurrences(of: "\u{05}", with: "")
                        .replacingOccurrences(of: "\u{02}", with: "")
                        .replacingOccurrences(of: "\n", with: "")
                        .replacingOccurrences(of: "\r", with: "")
                    let components = cleaned.split(separator: "+")
                    for component in components {
                        let lang = component.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                        guard !lang.isEmpty else { continue }
                        voicesByLanguage[lang, default: []].append(canonical)
                        if let hyphen = lang.firstIndex(of: "-") {
                            let prefix = String(lang[..<hyphen])
                            voicesByLanguage[prefix, default: []].append(canonical)
                        }
                    }
                }
            }

            index += 1
        }
    }

    private func selectVoiceIfNeeded(voiceIdentifier: String?, languageCode: String?) -> Bool {
        guard let desiredVoice = resolveVoice(voiceIdentifier: voiceIdentifier, languageCode: languageCode) else {
            return false
        }

        if currentVoice == desiredVoice { return true }

        let rc = desiredVoice.withCString { espeak_SetVoiceByName($0) }
        if rc == EE_OK {
            currentVoice = desiredVoice
            return true
        }
        logger.debug("espeak_SetVoiceByName failed for \(desiredVoice) (rc=\(rc))")
        return false
    }

    private func resolveVoice(voiceIdentifier: String?, languageCode: String?) -> String? {
        if let voiceIdentifier {
            let key = voiceIdentifier.lowercased()
            if let exact = availableVoicesByIdentifier[key] {
                return exact
            }
        }

        if let languageCode, !languageCode.isEmpty {
            let lower = languageCode.lowercased()
            if let matches = voicesByLanguage[lower], let first = matches.first {
                return first
            }
            if let hyphen = lower.firstIndex(of: "-") {
                let prefix = String(lower[..<hyphen])
                if let matches = voicesByLanguage[prefix], let first = matches.first {
                    return first
                }
            }
        }

        // Fallback to English voices if available
        if let enVoice = voicesByLanguage["en"]?.first ?? voicesByLanguage["en-us"]?.first {
            return enVoice
        }

        return availableVoicesByIdentifier.values.first
    }

    static func findEspeakDataPath() -> String? {
        let fm = FileManager.default

        func valid(_ url: URL) -> String? {
            let voices = url.appendingPathComponent("voices")
            return fm.fileExists(atPath: voices.path) ? url.path : nil
        }

        var candidates: [URL] = []

        // Highest priority: explicit environment override
        let envKeys = ["ESPEAKNG_DATA_PATH", "ESPEAKNG_DATA", "ESPEAK_DATA_PATH"]
        for key in envKeys {
            if let value = ProcessInfo.processInfo.environment[key], !value.isEmpty {
                candidates.append(URL(fileURLWithPath: value))
            }
        }

        // Preferred location: cache directory maintained by FluidAudio downloads
        if let cacheBase = try? TtsModels.cacheDirectoryURL() {
            let cacheCandidates = [
                "Models/espeak-ng/espeak-ng-data",
                "Models/espeak-ng/espeak-ng-data.bundle/espeak-ng-data",
                "Models/kokoro/Resources/espeak-ng/espeak-ng-data.bundle/espeak-ng-data",
                "Models/kokoro/espeak-ng-data",
            ]
            candidates.append(contentsOf: cacheCandidates.map { cacheBase.appendingPathComponent($0) })
        }

        // Check for system-wide installs (e.g. Homebrew or package managers)
        let systemPaths = [
            "/usr/local/share/espeak-ng-data",
            "/opt/homebrew/share/espeak-ng-data",
            "/usr/share/espeak-ng-data",
        ]
        candidates.append(contentsOf: systemPaths.map { URL(fileURLWithPath: $0) })

        // Embedded framework/bundle (when the xcframework ships the data)
        #if canImport(ESpeakNG)
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let frameworkRelative = [
            "Frameworks/ESpeakNG.xcframework/macos-arm64/ESpeakNG.framework/Resources/espeak-ng-data.bundle/espeak-ng-data",
            "Frameworks/ESpeakNG.xcframework/macos-arm64/ESpeakNG.framework/Versions/A/Resources/espeak-ng-data.bundle/espeak-ng-data",
            "Frameworks/ESpeakNG.xcframework/macos-arm64/ESpeakNG.framework/Versions/Current/Resources/espeak-ng-data.bundle/espeak-ng-data",
        ]
        candidates.append(contentsOf: frameworkRelative.map { cwd.appendingPathComponent($0) })

        if let bundle = Bundle(identifier: "org.espeak-ng.ESpeakNG") {
            let base = URL(fileURLWithPath: bundle.bundlePath)
            let bundleCandidates = [
                base.appendingPathComponent("Resources/espeak-ng-data.bundle/espeak-ng-data"),
                base.appendingPathComponent("Versions/A/Resources/espeak-ng-data.bundle/espeak-ng-data"),
            ]
            candidates.append(contentsOf: bundleCandidates)
        }
        #endif

        for candidate in candidates {
            if let ok = valid(candidate) {
                return ok
            }
        }

        return nil
    }

    /// Whether eSpeak NG data is available in the embedded framework bundle.
    static func isDataAvailable() -> Bool {
        return findEspeakDataPath() != nil
    }

    /// Return IPA tokens for a word, or nil on failure.
    func phonemize(word: String, voiceIdentifier: String? = nil, languageCode: String? = nil) -> [String]? {
        return queue.sync { () -> [String]? in
            guard initializeIfNeeded() else { return nil }

            let didSelectVoice = selectVoiceIfNeeded(voiceIdentifier: voiceIdentifier, languageCode: languageCode)
            if !didSelectVoice {
                logger.debug(
                    "Using default eSpeak voice; unable to resolve \(voiceIdentifier ?? languageCode ?? "default")")
            }

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
