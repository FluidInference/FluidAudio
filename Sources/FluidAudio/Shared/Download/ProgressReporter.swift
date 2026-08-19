import Foundation
import os

/// Models download-operation progress as weighted phases (#765 Wave 4),
/// replacing the fraction math previously copy-pasted at call sites.
///
/// The documented invariant: `fractionCompleted` is a monotonic fraction of
/// THIS operation in [0, 1]; inspect `phase` for what is happening. Repo
/// loads declare `downloadPhaseWeight: 0.5` (compile occupies the rest);
/// subdirectory downloads declare `1.0`.
struct ProgressReporter: Sendable {
    let handler: ProgressHandler?
    /// Fraction of the overall operation the download phase occupies.
    let downloadPhaseWeight: Double

    private func emit(_ fraction: Double, _ phase: DownloadPhase) {
        handler?(DownloadProgress(fractionCompleted: fraction, phase: phase))
    }

    /// Byte-weighted fraction of the download phase: bytes when total bytes
    /// are known, file counts otherwise; complete when there are no files.
    static func downloadFraction(
        completedBytes: Int64,
        totalBytes: Int64,
        completedFiles: Int,
        totalFiles: Int
    ) -> Double {
        guard totalFiles > 0 else { return 1.0 }
        guard totalBytes > 0 else {
            return min(Double(completedFiles) / Double(totalFiles), 1.0)
        }
        return min(Double(completedBytes) / Double(totalBytes), 1.0)
    }

    /// The listing phase opens every operation at fraction 0.
    func listing() {
        emit(0.0, .listing)
    }

    /// Live byte progress within a file. No emission when total bytes are
    /// unknown — mid-file bytes of unweighted files would inflate then snap
    /// back, breaking monotonicity.
    func liveBytes(
        completedBytes: Int64, totalBytes: Int64, fileIndex: Int, totalFiles: Int
    ) {
        guard totalBytes > 0 else { return }
        let fraction =
            downloadPhaseWeight
            * Self.downloadFraction(
                completedBytes: completedBytes, totalBytes: totalBytes,
                completedFiles: fileIndex, totalFiles: totalFiles)
        emit(fraction, .downloading(completedFiles: fileIndex, totalFiles: totalFiles))
    }

    /// Factory for the per-file live-bytes callback both download loops hand
    /// to FileDownloader; nil when there is no handler (skip the closure
    /// allocation and delegate churn entirely).
    func liveBytesCallback(
        baseBytes: Int64, totalBytes: Int64, fileIndex: Int, totalFiles: Int
    ) -> (@Sendable (Int64, Int64) -> Void)? {
        guard handler != nil else { return nil }
        return { bytesWritten, _ in
            self.liveBytes(
                completedBytes: baseBytes + bytesWritten,
                totalBytes: totalBytes,
                fileIndex: fileIndex,
                totalFiles: totalFiles)
        }
    }

    /// A file boundary (downloaded, skipped, or created empty).
    func fileBoundary(
        completedBytes: Int64, totalBytes: Int64, completedFiles: Int, totalFiles: Int
    ) {
        let fraction =
            downloadPhaseWeight
            * Self.downloadFraction(
                completedBytes: completedBytes, totalBytes: totalBytes,
                completedFiles: completedFiles, totalFiles: totalFiles)
        emit(fraction, .downloading(completedFiles: completedFiles, totalFiles: totalFiles))
    }

    /// The cached fast path: download phase complete without network.
    func cachedModelsAvailable() {
        emit(downloadPhaseWeight, .downloading(completedFiles: 0, totalFiles: 0))
    }

    /// Compiling model `index` of `count`. No emission for an empty set —
    /// dividing by zero would hand the handler a NaN fraction.
    func compiling(name: String, index: Int, count: Int) {
        guard count > 0 else { return }
        let compileWeight = 1.0 - downloadPhaseWeight
        emit(
            downloadPhaseWeight + compileWeight * Double(index) / Double(count),
            .compiling(modelName: name))
    }

    /// The operation is complete.
    func finished() {
        emit(1.0, .compiling(modelName: ""))
    }
}

/// Progress bookkeeping for a download loop whose files complete out of
/// order (#853). The sequential loops thread `completedBytes` through as a
/// plain running total; with a bounded task group that bookkeeping races, so
/// this class owns the counters instead. Byte updates and file boundaries
/// arrive from concurrent tasks; both the counter update and the handler
/// call happen under one lock, so emissions stay ordered and
/// `fractionCompleted` keeps the documented monotonic invariant.
final class ConcurrentProgress: Sendable {
    private struct Counters {
        /// Cumulative bytes reported so far by each still-downloading file.
        var inFlightBytes: [Int: Int64] = [:]
        var completedBytes: Int64 = 0
        var completedFiles = 0
        /// High-water mark; emissions below it are lifted to it so a file
        /// boundary landing behind another file's live bytes never regresses.
        var maxFraction = 0.0
    }

    private let state = OSAllocatedUnfairLock<Counters>(initialState: Counters())
    private let reporter: ProgressReporter
    private let totalBytes: Int64
    private let totalFiles: Int

    init(reporter: ProgressReporter, totalBytes: Int64, totalFiles: Int) {
        self.reporter = reporter
        self.totalBytes = totalBytes
        self.totalFiles = totalFiles
    }

    /// Live byte callback for file `index`; nil when there is no handler
    /// (skip the closure allocation and delegate churn entirely).
    func liveBytesCallback(fileIndex: Int) -> (@Sendable (Int64, Int64) -> Void)? {
        guard reporter.handler != nil else { return nil }
        return { bytesWritten, _ in
            self.updateBytes(fileIndex: fileIndex, bytesWritten: bytesWritten)
        }
    }

    /// Record that file `index` finished (downloaded, cached, or created
    /// empty) and emit its boundary. Returns the new completed-file count
    /// for the caller's log cadence.
    func fileCompleted(fileIndex: Int, size: Int) -> Int {
        state.withLock { counters in
            counters.inFlightBytes[fileIndex] = nil
            counters.completedBytes += Int64(max(0, size))
            counters.completedFiles += 1
            emitLocked(&counters, phaseFiles: counters.completedFiles)
            return counters.completedFiles
        }
    }

    private func updateBytes(fileIndex: Int, bytesWritten: Int64) {
        // Unknown-size files carry zero weight in totalBytes, so their real
        // bytes would inflate the fraction mid-file and snap back at the
        // boundary. Mirrors the sequential loop's guard.
        guard totalBytes > 0 else { return }
        state.withLock { counters in
            counters.inFlightBytes[fileIndex] = bytesWritten
            emitLocked(&counters, phaseFiles: counters.completedFiles)
        }
    }

    private func emitLocked(_ counters: inout Counters, phaseFiles: Int) {
        let bytes = counters.completedBytes + counters.inFlightBytes.values.reduce(0, +)
        let fraction =
            reporter.downloadPhaseWeight
            * ProgressReporter.downloadFraction(
                completedBytes: bytes, totalBytes: totalBytes,
                completedFiles: phaseFiles, totalFiles: totalFiles)
        counters.maxFraction = max(counters.maxFraction, fraction)
        reporter.handler?(
            DownloadProgress(
                fractionCompleted: counters.maxFraction,
                phase: .downloading(completedFiles: phaseFiles, totalFiles: totalFiles)))
    }
}
