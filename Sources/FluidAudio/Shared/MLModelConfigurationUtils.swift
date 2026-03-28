@preconcurrency import CoreML
import Foundation

/// Shared utilities for creating `MLModelConfiguration` instances and resolving model directories.
public enum MLModelConfigurationUtils {

    /// Create a default `MLModelConfiguration` with low-precision GPU accumulation enabled.
    ///
    /// - Parameter computeUnits: Compute units to use (default: `.cpuAndNeuralEngine`).
    /// - Returns: Configured `MLModelConfiguration`.
    public static func defaultConfiguration(
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        config.computeUnits = computeUnits
        return config
    }

    /// Default models directory under Application Support.
    ///
    /// - Parameter repo: Optional repository whose `folderName` is appended. When `nil`,
    ///   returns `~/Library/Application Support/FluidAudio/Models/`.
    /// - Returns: URL for the models directory.
    public static func defaultModelsDirectory(for repo: Repo? = nil) -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        var url =
            base
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
        if let repo {
            url = url.appendingPathComponent(repo.folderName, isDirectory: true)
        }
        return url
    }
}
