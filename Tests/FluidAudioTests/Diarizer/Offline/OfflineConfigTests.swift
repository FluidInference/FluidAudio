import XCTest

@testable import FluidAudio

final class OfflineConfigTests: XCTestCase {

    func testSegmentationBatchSizeDefaultsTo32() throws {
        XCTAssertEqual(OfflineDiarizerConfig.Segmentation.community.batchSize, 32)
        XCTAssertEqual(OfflineDiarizerConfig.default.segmentation.batchSize, 32)
        try OfflineDiarizerConfig.default.validate()
    }

    func testSegmentationBatchSizeValidation() throws {
        var config = OfflineDiarizerConfig.default
        for valid in [1, 8, 32] {
            config.segmentation.batchSize = valid
            XCTAssertNoThrow(try config.validate(), "batchSize \(valid)")
        }
        for invalid in [0, -1, 33] {
            config.segmentation.batchSize = invalid
            XCTAssertThrowsError(try config.validate(), "batchSize \(invalid)")
        }
    }

    func testClusteringDefaultsHaveNilSpeakerConstraints() {
        let clustering = OfflineDiarizerConfig.Clustering.community
        XCTAssertNil(clustering.minSpeakers)
        XCTAssertNil(clustering.maxSpeakers)
        XCTAssertNil(clustering.numSpeakers)
        XCTAssertTrue(clustering.constrainedAssignment)
    }

    func testClusteringAcceptsSpeakerConstraints() {
        let clustering = OfflineDiarizerConfig.Clustering(
            threshold: 0.6,
            warmStartFa: 0.07,
            warmStartFb: 0.8,
            minSpeakers: 2,
            maxSpeakers: 5,
            numSpeakers: nil
        )
        XCTAssertEqual(clustering.minSpeakers, 2)
        XCTAssertEqual(clustering.maxSpeakers, 5)
        XCTAssertNil(clustering.numSpeakers)
    }

    func testClusteringNumSpeakersOverridesMinMax() {
        let clustering = OfflineDiarizerConfig.Clustering(
            threshold: 0.6,
            warmStartFa: 0.07,
            warmStartFb: 0.8,
            minSpeakers: 1,
            maxSpeakers: 10,
            numSpeakers: 3
        )
        XCTAssertEqual(clustering.numSpeakers, 3)
    }

    func testFullConfigIncludesSpeakerConstraints() {
        var config = OfflineDiarizerConfig.default
        config.clustering.minSpeakers = 2
        config.clustering.maxSpeakers = 5

        XCTAssertEqual(config.clustering.minSpeakers, 2)
        XCTAssertEqual(config.clustering.maxSpeakers, 5)
        XCTAssertNil(config.clustering.numSpeakers)
    }

    func testConfigWithSpeakersConvenienceMethod() {
        let config = OfflineDiarizerConfig.default.withSpeakers(
            min: 2,
            max: 5
        )
        XCTAssertEqual(config.clustering.minSpeakers, 2)
        XCTAssertEqual(config.clustering.maxSpeakers, 5)
        XCTAssertNil(config.clustering.numSpeakers)
    }

    func testConfigWithExactSpeakersConvenienceMethod() {
        let config = OfflineDiarizerConfig.default.withSpeakers(exactly: 3)
        XCTAssertEqual(config.clustering.numSpeakers, 3)
    }
}
