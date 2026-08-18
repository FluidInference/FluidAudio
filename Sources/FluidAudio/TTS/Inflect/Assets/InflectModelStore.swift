@preconcurrency import CoreML
import Foundation

/// Actor store for one Inflect variant's CoreML bundles: the fixed-shape
/// `encoder` (loaded eagerly) and the eight `synthesizer_f<N>` frame buckets
/// (loaded lazily and cached the first time a bucket is needed).
///
/// All buckets are *downloaded* up front by `InflectResourceDownloader`
/// (they total ~8–19 MB), so bucket loads never hit the network.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public actor InflectModelStore {

    private let logger = AppLogger(category: "InflectModelStore")

    private let variant: InflectVariant
    private let directory: URL?
    private let computeUnits: MLComputeUnits

    private var encoderModel: MLModel?
    private var bucketModels: [Int: MLModel] = [:]
    private var repoDirectory: URL?

    public init(
        variant: InflectVariant,
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) {
        self.variant = variant
        self.directory = directory
        self.computeUnits = computeUnits
    }

    /// Download (if missing) and load the encoder. Buckets stay lazy.
    public func loadIfNeeded() async throws {
        if encoderModel != nil { return }

        let repoDir = try await InflectResourceDownloader.ensureModels(
            variant: variant, directory: directory)
        self.repoDirectory = repoDir

        encoderModel = try loadModel(
            repoDir: repoDir, fileName: ModelNames.Inflect.encoderFile)
        logger.info("Inflect \(variant.rawValue) encoder loaded from \(repoDir.path)")
    }

    public func encoder() throws -> MLModel {
        guard let encoderModel else { throw InflectError.notInitialized }
        return encoderModel
    }

    /// Return the synthesizer for the smallest bucket ≥ `frames`, loading and
    /// caching it on first use.
    public func synthesizer(forFrames frames: Int) async throws -> (model: MLModel, bucket: Int) {
        guard let bucket = InflectConstants.frameBuckets.first(where: { frames <= $0 }) else {
            throw InflectError.durationOverflow(
                frames: frames, maxFrames: InflectConstants.maxFrames)
        }
        if let cached = bucketModels[bucket] {
            return (cached, bucket)
        }
        guard let repoDir = repoDirectory else { throw InflectError.notInitialized }
        let model = try loadModel(
            repoDir: repoDir, fileName: ModelNames.Inflect.synthesizerFile(frames: bucket))
        bucketModels[bucket] = model
        logger.info("Inflect \(variant.rawValue) synthesizer bucket f\(bucket) loaded")
        return (model, bucket)
    }

    public func unload() {
        encoderModel = nil
        bucketModels.removeAll(keepingCapacity: false)
    }

    // MARK: - Helpers

    private func loadModel(repoDir: URL, fileName: String) throws -> MLModel {
        let url = repoDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw InflectError.modelFileNotFound(fileName)
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        do {
            return try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw InflectError.corruptedModel(fileName, underlying: "\(error)")
        }
    }
}
