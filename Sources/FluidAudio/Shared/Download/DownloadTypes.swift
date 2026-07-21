import Foundation

/// Errors thrown by the model-download stack: one merged, top-level error
/// domain. Absorbs the two pre-0.16 nested enums — `DownloadUtils
/// .HuggingFaceDownloadError` (the six download cases) and `DownloadUtils
/// .OfflineError` (`networkDisabled`/`modelMissing`) — with all case names
/// preserved, so catch-pattern migration is a prefix swap.
public enum DownloadError: LocalizedError {
    case invalidResponse
    case rateLimited(statusCode: Int, message: String)
    case downloadFailed(path: String, underlying: Error)
    case modelNotFound(path: String)
    case htmlErrorResponse(path: String, snippet: String)
    case invalidArtifact(path: String, reason: String)

    /// The byte stream stalled: fewer than `DownloadConfig.minStallBytes`
    /// arrived within `window` seconds, so the transfer was cancelled early
    /// for retry — byte-range resume continues from the bytes already on disk
    /// — instead of hanging on the multi-minute idle `timeout`. Retryable.
    case stalled(path: String, window: TimeInterval)

    /// A code path that would have hit the network was blocked by
    /// `ModelHub.offlineMode`. `operation` is the short tag of the blocked
    /// entry point (e.g. `"download(parakeet-tdt-0.6b-v3-coreml)"`).
    case networkDisabled(operation: String)

    /// `loadModels` was invoked in offline mode but one or more required
    /// files are missing from the local cache; the missing list lets the
    /// caller decide whether to ship a fix or fail loudly.
    case modelMissing(repo: String, missing: [String])

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Received an invalid response from Hugging Face."
        case .rateLimited(_, let message):
            return "Hugging Face rate limit encountered: \(message)"
        case .downloadFailed(let path, let underlying):
            return "Failed to download \(path): \(underlying.localizedDescription)"
        case .htmlErrorResponse(let path, let snippet):
            return "HuggingFace returned HTML instead of JSON for \(path) (rate limit or server issue): \(snippet)"
        case .modelNotFound(let path):
            return "Model file not found: \(path)"
        case .invalidArtifact(let path, let reason):
            return "Downloaded artifact for \(path) is invalid (\(reason)); refusing to cache it."
        case .stalled(let path, let window):
            return
                "Download of \(path) stalled (no meaningful progress for \(Int(window))s); cancelled for retry."
        case .networkDisabled(let operation):
            return "FluidAudio offline mode: \(operation) blocked"
        case .modelMissing(let repo, let missing):
            return
                "FluidAudio offline mode: required models missing for \(repo): \(missing.joined(separator: ", "))"
        }
    }
}

/// Download configuration shared by the download stack.
public struct DownloadConfig: Sendable {
    /// Per-request idle timeout (`URLRequest.timeoutInterval`). This timer
    /// resets whenever *any* byte arrives, so on its own it only fires after a
    /// full `timeout` seconds of complete silence — it will not catch a
    /// connection that trickles a few bytes. The stall watchdog below is what
    /// detects a frozen transfer quickly; `timeout` remains the outer bound.
    public let timeout: TimeInterval

    /// Stall watchdog threshold: the minimum number of bytes that must arrive
    /// within each `stallWindow` interval for the transfer to be considered
    /// alive. If fewer arrive, the transfer is cancelled and retried (byte-range
    /// resume continues from the bytes already on disk), surfacing a frozen CDN
    /// connection in seconds instead of waiting out `timeout`. Set to `0` to
    /// disable the watchdog. Defaults to 1 MiB.
    public let minStallBytes: Int64

    /// Length of each stall-watchdog observation window, in seconds. The
    /// watchdog wakes every `stallWindow` seconds and fails the transfer when
    /// fewer than `minStallBytes` arrived since the previous wake. Defaults to
    /// 120 s.
    public let stallWindow: TimeInterval

    public init(
        timeout: TimeInterval = 1800,  // 30 minutes for large models
        minStallBytes: Int64 = 1 << 20,  // 1 MiB
        stallWindow: TimeInterval = 120
    ) {
        self.timeout = timeout
        self.minStallBytes = minStallBytes
        self.stallWindow = stallWindow
    }

    public static let `default` = DownloadConfig()

    /// `true` when the stall watchdog is active (positive threshold and window).
    var stallWatchdogEnabled: Bool { minStallBytes > 0 && stallWindow > 0 }
}

/// Phase of a model download operation.
public enum DownloadPhase: Sendable {
    /// Listing files from the remote repository.
    case listing
    /// Downloading model files. `completedFiles` / `totalFiles` track per-file progress.
    case downloading(completedFiles: Int, totalFiles: Int)
    /// Compiling CoreML models after download.
    case compiling(modelName: String)
}

/// Progress snapshot passed to ``ProgressHandler`` closures.
public struct DownloadProgress: Sendable {
    /// Fraction complete in [0, 1].
    public let fractionCompleted: Double
    /// Current phase of the operation.
    public let phase: DownloadPhase

    public init(fractionCompleted: Double, phase: DownloadPhase) {
        self.fractionCompleted = fractionCompleted
        self.phase = phase
    }
}

/// Callback type for download progress reporting.
///
/// Called on an unspecified queue. If you need to update UI, dispatch to
/// the main actor inside your handler.
public typealias ProgressHandler = @Sendable (DownloadProgress) -> Void
