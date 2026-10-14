import Foundation

/// Bundled LuxTTS English G2P lexicon + aux tables. A separate SwiftPM target so
/// the resource bundle is only linked when the `TTS` trait is enabled (#990).
/// CocoaPods compiles this file into the FluidAudio pod and ships the same files
/// as the `LuxTtsG2pResources` resource bundle.
public enum LuxTtsG2pResources {
    #if SWIFT_PACKAGE
    public static var bundle: Bundle { .module }
    #else
    private final class BundleToken {}

    public static var bundle: Bundle {
        let host = Bundle(for: BundleToken.self)
        for candidate in [host, Bundle.main] {
            if let url = candidate.url(forResource: "LuxTtsG2pResources", withExtension: "bundle"),
                let bundle = Bundle(url: url)
            {
                return bundle
            }
        }
        return host
    }
    #endif
}
