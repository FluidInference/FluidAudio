import Foundation
import OSLog

/// Inverse Text Normalization (ITN) for post-processing ASR output.
///
/// Converts spoken-form text to written form:
/// - "two hundred thirty two" → "232"
/// - "five dollars and fifty cents" → "$5.50"
/// - "january fifth twenty twenty five" → "January 5, 2025"
///
/// For full ITN support, use text-processing-rs:
/// https://github.com/FluidInference/text-processing-rs
///
/// ## Usage
///
/// ```swift
/// let normalizer = TextNormalizer()
/// let result = normalizer.normalize("two hundred dollars")
/// // result is "$200" (with native library) or "two hundred dollars" (without)
/// ```
public final class TextNormalizer: @unchecked Sendable {

    private let logger = Logger(subsystem: "FluidAudio", category: "ITN")

    /// Whether the native NeMo library is available
    public let isNativeAvailable: Bool

    /// Shared instance for convenience
    public static let shared = TextNormalizer()

    public init() {
        // Check if native library is linked by trying to resolve the symbol
        self.isNativeAvailable = Self.checkNativeAvailability()
    }

    /// Normalize spoken-form text to written form.
    ///
    /// - Parameter input: Spoken-form text from ASR (e.g., "two hundred")
    /// - Returns: Written-form text (e.g., "200"), or original if no normalization applies
    public func normalize(_ input: String) -> String {
        guard isNativeAvailable else {
            return input
        }

        guard let normalize = Self.nemoNormalize,
            let freeString = Self.nemoFreeString
        else {
            return input
        }

        guard let resultPtr = input.withCString({ normalize($0) }) else {
            return input
        }

        defer { freeString(resultPtr) }
        return String(cString: resultPtr)
    }

    /// Normalize an ASR result, returning a new result with normalized text.
    ///
    /// - Parameter result: The original ASR result
    /// - Returns: A new ASR result with normalized text
    public func normalize(result: ASRResult) -> ASRResult {
        let normalizedText = normalize(result.text)

        // If text didn't change, return original
        guard normalizedText != result.text else {
            return result
        }

        return ASRResult(
            text: normalizedText,
            confidence: result.confidence,
            duration: result.duration,
            processingTime: result.processingTime,
            tokenTimings: result.tokenTimings,
            ctcDetectedTerms: result.ctcDetectedTerms,
            ctcAppliedTerms: result.ctcAppliedTerms
        )
    }

    /// Get the native library version, or nil if not available.
    public var version: String? {
        guard isNativeAvailable,
            let getVersion = Self.nemoVersion,
            let versionPtr = getVersion()
        else {
            return nil
        }
        return String(cString: versionPtr)
    }

    // MARK: - Dynamic Library Loading

    nonisolated(unsafe) private static var nemoNormalize:
        (
            @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?
        )?
    nonisolated(unsafe) private static var nemoFreeString:
        (
            @convention(c) (UnsafeMutablePointer<CChar>?) -> Void
        )?
    nonisolated(unsafe) private static var nemoVersion: (@convention(c) () -> UnsafePointer<CChar>?)?

    private static func checkNativeAvailability() -> Bool {
        // Try to load function pointers from the linked library
        guard let handle = dlopen(nil, RTLD_NOW) else {
            return false
        }

        guard let normalizePtr = dlsym(handle, "nemo_normalize"),
            let freePtr = dlsym(handle, "nemo_free_string"),
            let versionPtr = dlsym(handle, "nemo_version")
        else {
            return false
        }

        nemoNormalize = unsafeBitCast(
            normalizePtr,
            to: (@convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?).self
        )
        nemoFreeString = unsafeBitCast(
            freePtr, to: (@convention(c) (UnsafeMutablePointer<CChar>?) -> Void).self
        )
        nemoVersion = unsafeBitCast(
            versionPtr, to: (@convention(c) () -> UnsafePointer<CChar>?).self
        )

        return true
    }
}
