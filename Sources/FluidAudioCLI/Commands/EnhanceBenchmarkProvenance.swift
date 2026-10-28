#if os(macOS)
import FluidAudio
import Foundation

/// Fingerprints the exact inputs consumed by an enhancement benchmark run.
enum EnhanceBenchmarkProvenance {
    static func modelFiles(variants: [LocalVqeVariant], chunk: LocalVqeChunk) throws -> [String: String] {
        let asrDirectory = AsrModels.defaultCacheDirectory(for: .v3)
        let enhancementDirectory = MLModelConfigurationUtils.defaultModelsDirectory(for: .localVqe)
        var hashes: [String: String] = [:]
        for name in ModelNames.ASR.requiredModelsV3().union([ModelNames.ASR.vocabularyFile]).sorted() {
            try fingerprint(asrDirectory.appendingPathComponent(name), prefix: "parakeet-v3/\(name)", into: &hashes)
        }
        for variant in variants {
            let name = ModelNames.LocalVQE.modelFile(variant: variant, chunk: chunk)
            try fingerprint(
                enhancementDirectory.appendingPathComponent(name), prefix: "localvqe/\(name)", into: &hashes)
        }
        return hashes
    }

    private static func fingerprint(_ url: URL, prefix: String, into hashes: inout [String: String]) throws {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
        if values.isDirectory == true {
            let children = try FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey], options: [.skipsHiddenFiles])
            guard !children.isEmpty else {
                throw LocalVqeError.modelProcessingFailed("Empty model directory: \(url.path)")
            }
            for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                try fingerprint(child, prefix: "\(prefix)/\(child.lastPathComponent)", into: &hashes)
            }
            return
        }
        guard values.isRegularFile == true else {
            throw LocalVqeError.modelProcessingFailed("Unsupported model file: \(url.path)")
        }
        hashes[prefix] = try EnhanceBenchmarkDataset.sha256(of: url)
    }

    static func audioFiles(_ examples: [EnhanceBenchmarkDataset.Example]) throws -> [String: String] {
        var hashes: [String: String] = [:]
        for example in examples {
            for url in [example.mic, example.lpb, example.clean] {
                hashes[url.lastPathComponent] = try EnhanceBenchmarkDataset.sha256(of: url)
            }
        }
        return hashes
    }
}
#endif
