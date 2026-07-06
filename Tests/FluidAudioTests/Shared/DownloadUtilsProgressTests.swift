import XCTest

@testable import FluidAudio

final class DownloadUtilsProgressTests: XCTestCase {

    func testSubdirectoryProgressUsesByteWeightingDuringLargeFile() {
        let fraction = DownloadUtils.subdirectoryProgressFraction(
            completedBytes: 399,
            totalBytes: 665,
            completedFiles: 9,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, 399.0 / 665.0, accuracy: 0.0001)
        XCTAssertGreaterThan(fraction, Double(9) / Double(22))
    }

    func testSubdirectoryProgressIsMonotonicForRealisticSequence() {
        let progress = [
            DownloadUtils.subdirectoryProgressFraction(
                completedBytes: 0,
                totalBytes: 665,
                completedFiles: 0,
                totalFiles: 22
            ),
            DownloadUtils.subdirectoryProgressFraction(
                completedBytes: 40,
                totalBytes: 665,
                completedFiles: 4,
                totalFiles: 22
            ),
            DownloadUtils.subdirectoryProgressFraction(
                completedBytes: 120,
                totalBytes: 665,
                completedFiles: 9,
                totalFiles: 22
            ),
            DownloadUtils.subdirectoryProgressFraction(
                completedBytes: 399,
                totalBytes: 665,
                completedFiles: 9,
                totalFiles: 22
            ),
            DownloadUtils.subdirectoryProgressFraction(
                completedBytes: 565,
                totalBytes: 665,
                completedFiles: 10,
                totalFiles: 22
            ),
            DownloadUtils.subdirectoryProgressFraction(
                completedBytes: 665,
                totalBytes: 665,
                completedFiles: 22,
                totalFiles: 22
            ),
        ]

        XCTAssertEqual(progress, progress.sorted())
    }

    func testSubdirectoryProgressClampsToOne() {
        let fraction = DownloadUtils.subdirectoryProgressFraction(
            completedBytes: 700,
            totalBytes: 665,
            completedFiles: 22,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, 1.0)
    }

    func testSubdirectoryProgressFallsBackToFileCountWhenTotalBytesUnknown() {
        let fraction = DownloadUtils.subdirectoryProgressFraction(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 9,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, Double(9) / Double(22), accuracy: 0.0001)
    }

    func testSubdirectoryProgressIsCompleteWhenThereAreNoFiles() {
        let fraction = DownloadUtils.subdirectoryProgressFraction(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 0,
            totalFiles: 0
        )

        XCTAssertEqual(fraction, 1.0)
    }
}
