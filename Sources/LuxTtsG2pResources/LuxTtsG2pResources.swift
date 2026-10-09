import Foundation

/// Bundled LuxTTS English G2P lexicon + aux tables. A separate target so the
/// resource bundle is only linked when the `TTS` trait is enabled (#990).
public enum LuxTtsG2pResources {
    public static var bundle: Bundle { .module }
}
