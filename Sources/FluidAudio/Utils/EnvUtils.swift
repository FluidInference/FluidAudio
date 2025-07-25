import AVFoundation
import CoreML

public class EnvUtils {

    public static func isSumulator() -> Bool {
        #if targetEnvironment(simulator)
            return true
        #else
            return false
        #endif
    }

    public static func preferredComputeUnits() -> MLComputeUnits {
        if isSumulator() {
            return .cpuOnly
        } else {
            return .cpuAndNeuralEngine
        }
    }

    /// Get models directory
    public static func modelsDirectory(for modelName: String) -> URL {
        let directory: URL

        #if os(iOS)
            // Use Documents directory on iOS for better compatibility with sandboxing
            let documents = FileManager.default.urls(
                for: .documentDirectory, in: .userDomainMask
            ).first!
            directory = documents.appendingPathComponent(
                "FluidAudio/models/\(modelName)", isDirectory: true)
        #else
            // Use Application Support on macOS
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first!
            directory = appSupport.appendingPathComponent(
                "FluidAudio/\(modelName)", isDirectory: true)
        #endif

        return directory
    }

}
