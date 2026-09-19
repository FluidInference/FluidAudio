import CryptoKit
import Foundation

/// Fetches the portable model from the reviewed HF commit, once, into a dedicated cache.
public enum DemoAssets {
    /// Exact artifact revision from the model PR; no dependency on whether it has merged.
    public static let revision = "c87b915d302bdbff644709408d8fbd8c8effe894"
    private static let packageName = "cua_s1_forms_fp16_options32.mlpackage"
    private static let files = [
        "Manifest.json": "2bc0f5f62337b27fb6b0ecde248f1e3dc269e1ba4b65516aaeede2a60e293dcc",
        "Data/com.apple.CoreML/model.mlmodel": "70485fc18cbb21785df833cbddddc0b5b59acb00d22394b76e55307e2c135dd0",
        "Data/com.apple.CoreML/weights/weight.bin": "4da9259f798e44f5a1b50769ee1916fd3747c4d723dd9997b516c7fe238c7895",
    ]

    /// Return a verified cached package, downloading only missing or damaged files.
    public static func modelURL() async throws -> URL {
        let support = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let root = support.appendingPathComponent("FluidAudio/Demos/CuaS1Forms/\(revision)/\(packageName)")
        for (name, expected) in files.sorted(by: { $0.key < $1.key }) {
            try Task.checkCancellation()
            let destination = root.appendingPathComponent(name)
            if let existing = try? Data(contentsOf: destination), checksum(existing) == expected { continue }
            let address =
                "https://huggingface.co/FluidInference/cua-s1-forms-coreml/resolve/\(revision)/\(packageName)/\(name)"
            guard let url = URL(string: address) else { throw DemoError("Invalid model asset URL.") }
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
                throw DemoError(
                    "Could not download the model. Check your connection, or use --model with a local package.")
            }
            guard checksum(data) == expected else { throw DemoError("Checksum mismatch for \(name).") }
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: destination, options: .atomic)
        }
        return root
    }

    private static func checksum(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
