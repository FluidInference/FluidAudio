#if TTS
import CryptoKit
import Foundation

/// Pinned source-package downloads, compiled locally by the v3 runtime.
/// Kept separate from legacy caches so different checkpoints cannot be mixed.
enum KokoroAneV3Assets {
    static let revision = "6c0750dd02ef9d981fb82f147d784a3f171755ee"
    static let baseRevision = "006395f65025af251858b1ab0a7178a6a1e73f9f"
    static let repository = "FluidInference/kokoro-82m-coreml"

    struct File: Codable, Sendable {
        let path: String
        let bytes: Int
        let sha256: String
    }
    struct Manifest: Decodable, Sendable {
        struct Base: Decodable, Sendable {
            let repoId: String
            let revision: String
            let directory: String
            let models: [String]
            enum CodingKeys: String, CodingKey {
                case repoId = "repo_id"
                case revision, directory, models
            }
        }
        let schemaVersion: Int
        let language: String
        let base: Base
        let files: [File]
        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case language, base, files
        }
    }
    struct TreeFile: Codable, Sendable {
        struct LFS: Codable, Sendable { let oid: String }
        let type: String
        let path: String
        let size: Int?
        let oid: String?
        let lfs: LFS?
    }

    static func validateVariant(_ variant: KokoroAneVariant) throws {
        guard [.english, .japanese, .mandarin].contains(variant) else {
            throw KokoroAneError.inputProcessingFailed("ANE-v3 currently supports English, Japanese and Mandarin")
        }
    }

    static let originalBaseModels: Set<String> = [
        "KokoroAlbert", "KokoroPostAlbert", "KokoroAlignment", "KokoroProsody",
        "KokoroNoise_v2", "KokoroVocoder", "KokoroTail",
    ]

    static func language(_ variant: KokoroAneVariant) -> String {
        switch variant {
        case .english: return "en"
        case .japanese: return "ja"
        case .mandarin: return "zh"
        case .spanish: return "es"
        case .french: return "fr"
        }
    }

    static func prefix(_ variant: KokoroAneVariant) -> String {
        variant == .english ? "ANE-v3" : "ANE-v3/\(language(variant))"
    }

    static func cacheDirectory(variant: KokoroAneVariant, directory: URL?) throws -> URL {
        let root = try directory ?? TtsCacheDirectory.ensure().appendingPathComponent("Models")
        return root.appendingPathComponent("kokoro-82m-coreml/ANE-v3/\(revision)/\(language(variant))")
    }

    static func validatePath(_ path: String) throws {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"),
            !path.contains("?"), !path.contains("#"), !path.contains("%"),
            path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({
                !$0.isEmpty && $0 != "." && $0 != ".."
            })
        else { throw KokoroAneError.downloadFailed("Unsafe asset path: \(path)") }
    }

    static func remoteURL(_ path: String, revision: String) throws -> URL {
        try validatePath(path)
        guard let url = URL(string: "\(ModelRegistry.baseURL)/\(repository)/resolve/\(revision)/\(path)") else {
            throw KokoroAneError.downloadFailed("Invalid asset URL")
        }
        return url
    }

    static func fetch(_ path: String, revision: String) async throws -> Data {
        try await ModelHub.fetchFile(
            from: remoteURL(path, revision: revision), description: "Kokoro ANE-v3 \(path)")
    }

    static func matches(_ data: Data, file: File) -> Bool {
        data.count == file.bytes && SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == file.sha256
    }

    static func selectedFiles(_ manifest: Manifest) throws -> [File] {
        guard manifest.schemaVersion == 1 else { throw KokoroAneError.downloadFailed("Unsupported v3 manifest") }
        for file in manifest.files { try validatePath(file.path) }
        let selected = manifest.files.filter {
            ($0.path.hasPrefix("fast/") || $0.path.hasPrefix("decoder/")) && $0.path.contains(".mlpackage/")
                || $0.path == "fast/native-source.json" || $0.path == "vocab.json"
                || (!$0.path.contains("/") && $0.path.hasSuffix(".bin"))
        }
        let packages = [
            "fast/KokoroAlbert_32", "fast/KokoroAlbert_64", "fast/KokoroSourceGenerator",
            "decoder/KokoroDecoderPre_120", "decoder/KokoroDecoderPre_200",
        ]
        guard Set(selected.map(\.path)).count == selected.count,
            selected.contains(where: { $0.path == "fast/native-source.json" }),
            packages.allSatisfy({ name in selected.contains(where: { $0.path == name + ".mlpackage/Manifest.json" }) })
        else { throw KokoroAneError.downloadFailed("Incomplete v3 manifest") }
        return selected
    }

    static func ensure(variant: KokoroAneVariant, directory: URL?) async throws -> URL {
        try validateVariant(variant)
        let root = try cacheDirectory(variant: variant, directory: directory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let manifestURL = root.appendingPathComponent("manifest.json")
        let expectedHash: String
        switch variant {
        case .english: expectedHash = "445815d58ff6b3df588e154bc2ff72dd3cd1126d48d71d8050f683f1720259b6"
        case .japanese: expectedHash = "588ef27b57d604b66f4b45a13f1ec2d03f42dbf2df64bae09574c586bf3a58db"
        case .mandarin: expectedHash = "b27d67089f99d4de1d965d349a619c04d08acf1a16e917223864ae479c0f9702"
        case .spanish, .french: throw KokoroAneError.inputProcessingFailed("Unsupported ANE-v3 language")
        }
        var data = (try? Data(contentsOf: manifestURL)) ?? Data()
        if SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() != expectedHash {
            data = try await fetch("\(prefix(variant))/manifest.json", revision: revision)
        }
        guard SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == expectedHash else {
            throw KokoroAneError.downloadFailed("Invalid pinned v3 manifest checksum")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        guard manifest.language == language(variant), manifest.base.repoId == repository,
            manifest.base.revision == baseRevision,
            manifest.base.directory == (variant == .mandarin ? "ANE-zh" : "ANE"),
            Set(manifest.base.models) == originalBaseModels
        else { throw KokoroAneError.downloadFailed("Unexpected v3 checkpoint dependencies") }
        let files = try selectedFiles(manifest)
        for file in files {
            try Task.checkCancellation()
            let target = root.appendingPathComponent(file.path)
            if let cached = try? Data(contentsOf: target, options: .mappedIfSafe), matches(cached, file: file) {
                continue
            }
            let bytes = try await fetch("\(prefix(variant))/\(file.path)", revision: revision)
            guard matches(bytes, file: file) else {
                throw KokoroAneError.downloadFailed("Checksum mismatch: \(file.path)")
            }
            try FileManager.default.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: target, options: .atomic)
        }
        let baseFiles = try await baseIndex(manifest: manifest, root: root)
        for file in baseFiles {
            try Task.checkCancellation()
            let relative = String(file.path.dropFirst(manifest.base.directory.count + 1))
            let target = root.appendingPathComponent("base/\(relative)")
            if let cached = try? Data(contentsOf: target, options: .mappedIfSafe), matches(cached, treeFile: file) {
                continue
            }
            let bytes = try await fetch(file.path, revision: baseRevision)
            guard matches(bytes, treeFile: file) else {
                throw KokoroAneError.downloadFailed("Checksum mismatch: \(file.path)")
            }
            try FileManager.default.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: target, options: .atomic)
        }
        if variant == .english {
            let file = File(
                path: "vocab.json", bytes: 1416,
                sha256: "8d65b0188b77eafc60751dac42bbac7ab5f5685074af44db91d1877b42dc1d7c")
            let vocab = root.appendingPathComponent(file.path)
            let cached = (try? Data(contentsOf: vocab)) ?? Data()
            if !matches(cached, file: file) {
                let bytes = try await fetch("ANE/vocab.json", revision: baseRevision)
                guard matches(bytes, file: file) else {
                    throw KokoroAneError.downloadFailed("Invalid English vocab checksum")
                }
                try bytes.write(to: vocab, options: .atomic)
            }
        }

        _ = try await ensureVoice(variant.defaultVoice, variant: variant, root: root)
        try data.write(to: manifestURL, options: .atomic)
        return root
    }

    static func matches(_ data: Data, treeFile file: TreeFile) -> Bool {
        guard data.count == file.size else { return false }
        if let lfs = file.lfs {
            return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == lfs.oid
        }
        var blob = Data("blob \(data.count)\0".utf8)
        blob.append(data)
        return Insecure.SHA1.hash(data: blob).map { String(format: "%02x", $0) }.joined() == file.oid
    }

    private static func baseIndex(manifest: Manifest, root: URL) async throws -> [TreeFile] {
        let cache = root.appendingPathComponent("base-index-v3.json")
        let all: [TreeFile]
        if let data = try? Data(contentsOf: cache), let cached = try? JSONDecoder().decode([TreeFile].self, from: data)
        {
            all = cached
        } else {
            var items: [TreeFile] = []
            var next: URL? = try ModelRegistry.apiModels(
                repository, "tree/\(baseRevision)/\(manifest.base.directory)?recursive=true&limit=1000")
            var visited = Set<URL>()
            while let url = next {
                guard visited.insert(url).inserted else {
                    throw KokoroAneError.downloadFailed("Repeated asset listing page")
                }
                let (data, response) = try await ModelHub.fetchWithAuth(from: url)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    throw KokoroAneError.downloadFailed("Unable to list pinned base packages")
                }
                items += try JSONDecoder().decode([TreeFile].self, from: data)
                next = HFClient.nextPageURL(from: http)
                if let next, !next.absoluteString.hasPrefix(ModelRegistry.baseURL + "/api/models/" + repository + "/") {
                    throw KokoroAneError.downloadFailed("Invalid pagination URL")
                }
            }
            all = items
        }
        // Adopt the SDK's fp32 prosody and COLA-corrected tail from the same
        // immutable revision. Their source packages are not published. Mandarin
        // Noise_v2 also has an incomplete source package, so use its compiled bundle.
        let prefixes = manifest.base.models.map { original in
            let name = original == "KokoroProsody" || original == "KokoroTail" ? original + "_v2" : original
            let compiled =
                name == "KokoroProsody_v2" || name == "KokoroTail_v2"
                || (manifest.base.directory == "ANE-zh" && name == "KokoroNoise_v2")
            return "\(manifest.base.directory)/\(name).\(compiled ? "mlmodelc" : "mlpackage")/"
        }
        let selected = all.filter { item in item.type == "file" && prefixes.contains(where: { item.path.hasPrefix($0) })
        }
        for item in selected { try validatePath(item.path) }
        guard
            prefixes.allSatisfy({ prefix in
                let required =
                    prefix.hasSuffix(".mlmodelc/")
                    ? ["coremldata.bin", "model.mil", "weights/weight.bin"]
                    : [
                        "Manifest.json", "Data/com.apple.CoreML/model.mlmodel",
                        "Data/com.apple.CoreML/weights/weight.bin",
                    ]
                return required.allSatisfy { file in selected.contains(where: { $0.path == prefix + file }) }
            })
        else {
            throw KokoroAneError.downloadFailed("Pinned base packages are incomplete")
        }
        try JSONEncoder().encode(selected).write(to: cache, options: .atomic)
        return selected
    }

    static func validateVoice(_ voice: String, variant: KokoroAneVariant) throws {
        let prefixes: [String]
        switch variant {
        case .english: prefixes = ["af_", "am_", "bf_", "bm_"]
        case .japanese: prefixes = ["jf_", "jm_"]
        case .mandarin: prefixes = ["zf_", "zm_"]
        case .spanish, .french: throw KokoroAneError.invalidVoicePack("Unsupported ANE-v3 language")
        }
        guard prefixes.contains(where: voice.hasPrefix),
            voice.utf8.allSatisfy({ (48...57).contains($0) || (97...122).contains($0) || $0 == 95 })
        else {
            throw KokoroAneError.invalidVoicePack("Voice '\(voice)' does not belong to \(variant.rawValue)")
        }
    }

    static func decodeEnglishVoice(_ json: Data) throws -> Data {
        let rows = try JSONDecoder().decode([String: [Float]].self, from: json)
        var output = Data(capacity: 510 * 256 * 4)
        for index in 1...KokoroAneConstants.voicePackRows {
            guard let row = rows[String(index)], row.count == KokoroAneConstants.voicePackCols,
                row.allSatisfy(\.isFinite)
            else {
                throw KokoroAneError.invalidVoicePack("Invalid English voice row \(index)")
            }
            for value in row {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { output.append(contentsOf: $0) }
            }
        }
        return output
    }

    static func validVoiceData(_ data: Data) -> Bool {
        guard data.count == KokoroAneConstants.voicePackRows * KokoroAneConstants.voicePackCols * 4 else {
            return false
        }
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).allSatisfy {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
                    .isFinite
            }
        }
    }

    static func ensureVoice(_ voice: String, variant: KokoroAneVariant, root: URL) async throws -> URL {
        try validateVoice(voice, variant: variant)
        let bundled = root.appendingPathComponent("\(voice).bin")
        if let bytes = try? Data(contentsOf: bundled), validVoiceData(bytes) { return bundled }
        let local = root.appendingPathComponent("voices/\(voice).bin")
        if let bytes = try? Data(contentsOf: local), validVoiceData(bytes) { return local }
        let folder: String
        switch variant {
        case .english: folder = "ANE"
        case .japanese: folder = "ANE-ja/voices"
        case .mandarin: folder = "ANE-zh/voices"
        case .spanish, .french: throw KokoroAneError.invalidVoicePack("Unsupported ANE-v3 language")
        }
        let data: Data
        if variant == .english && voice != variant.defaultVoice {
            // The original English repository publishes additional voices as
            // numbered JSON rows 1...510 (row 1 corresponds to phoneme length 1).
            let json = try await fetch("voices/\(voice).json", revision: baseRevision)
            data = try decodeEnglishVoice(json)
        } else {
            data = try await fetch("\(folder)/\(voice).bin", revision: baseRevision)
        }
        guard validVoiceData(data) else {
            throw KokoroAneError.invalidVoicePack("Invalid downloaded voice size")
        }
        try FileManager.default.createDirectory(
            at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: local, options: .atomic)
        return local
    }
}
#endif
