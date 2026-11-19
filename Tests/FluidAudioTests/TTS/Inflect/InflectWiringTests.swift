import XCTest

@testable import FluidAudio

/// Static wiring: repo paths, required-model set, and bucket bookkeeping.
final class InflectWiringTests: XCTestCase {

    func testRepoPathsForBothVariants() {
        XCTAssertEqual(Repo.inflectMicro.remotePath, "FluidInference/inflect-v2-coreml")
        XCTAssertEqual(Repo.inflectNano.remotePath, "FluidInference/inflect-v2-coreml")
        XCTAssertEqual(Repo.inflectMicro.subPath, "micro")
        XCTAssertEqual(Repo.inflectNano.subPath, "nano")
        XCTAssertEqual(Repo.inflectMicro.folderName, "inflect-v2-coreml/micro")
        XCTAssertEqual(Repo.inflectNano.folderName, "inflect-v2-coreml/nano")
    }

    func testVariantMapsToRepo() {
        XCTAssertEqual(InflectVariant.micro.repo, .inflectMicro)
        XCTAssertEqual(InflectVariant.nano.repo, .inflectNano)
    }

    func testRequiredModelsIsEncoderPlusEightBuckets() {
        let required = ModelNames.Inflect.requiredModels
        XCTAssertEqual(required.count, 9)
        XCTAssertTrue(required.contains(ModelNames.Inflect.encoderFile))
        for frames in InflectConstants.frameBuckets {
            XCTAssertTrue(required.contains(ModelNames.Inflect.synthesizerFile(frames: frames)))
        }
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .inflectMicro, variant: nil), required)
    }

    func testFrameBucketsAreSortedAscending() {
        XCTAssertEqual(InflectConstants.frameBuckets, InflectConstants.frameBuckets.sorted())
        XCTAssertEqual(InflectConstants.maxFrames, 2048)
    }

    /// `ModelNames.Inflect.frameBuckets` is the download-side source of truth
    /// (it lives outside the `TTS` guard, #990); the runtime constants must
    /// keep reading the same list so every downloaded bucket is loadable.
    func testRuntimeFrameBucketsMatchDownloadedBuckets() {
        XCTAssertEqual(InflectConstants.frameBuckets, ModelNames.Inflect.frameBuckets)
        XCTAssertEqual(InflectConstants.frameBuckets, [256, 384, 512, 640, 768, 896, 1024, 2048])
        XCTAssertEqual(ModelNames.Inflect.requiredModels.count, ModelNames.Inflect.frameBuckets.count + 1)
    }

    func testInterChannelsPerVariant() {
        XCTAssertEqual(InflectConstants.interChannels(for: .micro), 192)
        XCTAssertEqual(InflectConstants.interChannels(for: .nano), 128)
    }
}
