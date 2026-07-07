import XCTest

@testable import FluidAudio

final class ProgressReporterTests: XCTestCase {

    func testSubdirectoryProgressUsesByteWeightingDuringLargeFile() {
        let fraction = ProgressReporter.downloadFraction(
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
            ProgressReporter.downloadFraction(
                completedBytes: 0,
                totalBytes: 665,
                completedFiles: 0,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 40,
                totalBytes: 665,
                completedFiles: 4,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 120,
                totalBytes: 665,
                completedFiles: 9,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 399,
                totalBytes: 665,
                completedFiles: 9,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 565,
                totalBytes: 665,
                completedFiles: 10,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 665,
                totalBytes: 665,
                completedFiles: 22,
                totalFiles: 22
            ),
        ]

        XCTAssertEqual(progress, progress.sorted())
    }

    func testSubdirectoryProgressClampsToOne() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 700,
            totalBytes: 665,
            completedFiles: 22,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, 1.0)
    }

    func testSubdirectoryProgressFallsBackToFileCountWhenTotalBytesUnknown() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 9,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, Double(9) / Double(22), accuracy: 0.0001)
    }

    func testSubdirectoryProgressIsCompleteWhenThereAreNoFiles() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 0,
            totalFiles: 0
        )

        XCTAssertEqual(fraction, 1.0)
    }
}
