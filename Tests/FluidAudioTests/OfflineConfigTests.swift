import XCTest

@testable import FluidAudio

final class OfflineConfigTests: XCTestCase {

    func testClusteringDefaultsHaveNilSpeakerConstraints() {
        let clustering = OfflineDiarizerConfig.Clustering.community
        XCTAssertNil(clustering.minSpeakers)
        XCTAssertNil(clustering.maxSpeakers)
        XCTAssertNil(clustering.numSpeakers)
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
