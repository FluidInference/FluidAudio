import Accelerate
import CoreML
import FluidAudio
import Foundation

extension KokoroSynthesizer {
    struct VoiceEmbeddingPayload {
        let sourceURL: URL
        let json: Any
    }

    struct VoiceEmbeddingCacheKey: Hashable {
        let voiceID: String
        let phonemeCount: Int
    }

    struct VoiceEmbeddingData: Sendable {
        let voiceID: String
        let vector: [Float]
        let l2Norm: Float
    }

    static func candidateVoiceEmbeddingURLs(
        for voiceID: String,
        cwd: URL,
        voicesDir: URL
    ) -> [URL] {
        [
            cwd.appendingPathComponent("voices/\(voiceID).json"),
            cwd.appendingPathComponent("\(voiceID).json"),
            voicesDir.appendingPathComponent("\(voiceID).json"),
        ]
    }

    static func cachedVoiceEmbeddingPayload(
        for voiceID: String,
        candidates: [URL]
    ) throws -> VoiceEmbeddingPayload {
        voiceEmbeddingLock.lock()
        if let payload = voiceEmbeddingPayloads[voiceID] {
            voiceEmbeddingLock.unlock()
            return payload
        }
        voiceEmbeddingLock.unlock()

        guard let source = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            let checkedPaths = candidates.map { $0.path }.joined(separator: ", ")
            throw TTSError.modelNotFound(
                "Voice embedding unavailable for \(voiceID); checked paths: \(checkedPaths)")
        }

        let data = try Data(contentsOf: source)
        let json = try JSONSerialization.jsonObject(with: data)
        let payload = VoiceEmbeddingPayload(sourceURL: source, json: json)

        voiceEmbeddingLock.lock()
        voiceEmbeddingPayloads[voiceID] = payload
        voiceEmbeddingLock.unlock()

        return payload
    }

    static func cachedVoiceEmbeddingVector(for key: VoiceEmbeddingCacheKey) -> [Float]? {
        voiceEmbeddingLock.lock()
        let vector = voiceEmbeddingVectors[key]
        voiceEmbeddingLock.unlock()
        return vector
    }

    static func storeVoiceEmbeddingVector(_ vector: [Float], for key: VoiceEmbeddingCacheKey) {
        voiceEmbeddingLock.lock()
        voiceEmbeddingVectors[key] = vector
        voiceEmbeddingLock.unlock()
    }

    static func fetchVoiceEmbeddingData(
        voice: String,
        phonemeCount: Int,
        expectedDimension: Int
    ) throws -> VoiceEmbeddingData {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        func resolveVector(for voiceID: String) throws -> ([Float]?, URL, Bool) {
            let candidates = candidateVoiceEmbeddingURLs(for: voiceID, cwd: cwd, voicesDir: voicesDir)
            do {
                let payload = try cachedVoiceEmbeddingPayload(for: voiceID, candidates: candidates)
                let key = VoiceEmbeddingCacheKey(voiceID: voiceID, phonemeCount: phonemeCount)
                if let cached = Self.cachedVoiceEmbeddingVector(for: key) {
                    return (cached, payload.sourceURL, true)
                }
                let vector = parseVoiceEmbeddingVector(payload.json, voiceID: voiceID, phonemeCount: phonemeCount)
                if let vector {
                    Self.storeVoiceEmbeddingVector(vector, for: key)
                } else {
                    Self.logger.warning(
                        "Voice embedding payload lacked usable vector for \(voiceID) at \(payload.sourceURL.path)"
                    )
                }
                return (vector, payload.sourceURL, false)
            } catch {
                Self.logger.warning("Failed to load voice embedding for \(voiceID): \(error.localizedDescription)")
                return (nil, candidates.first ?? voicesDir, false)
            }
        }

        var voiceUsed = voice
        var attemptedVoices = [voice]
        var (vector, sourceURL, wasCached) = try resolveVector(for: voiceUsed)

        if vector == nil && voice != TtsConstants.recommendedVoice {
            Self.logger.warning(
                "Voice embedding for \(voice) not found; falling back to \(TtsConstants.recommendedVoice)")
            voiceUsed = TtsConstants.recommendedVoice
            attemptedVoices.append(TtsConstants.recommendedVoice)
            (vector, sourceURL, wasCached) = try resolveVector(for: voiceUsed)
        }

        guard let resolvedVector = vector else {
            throw TTSError.modelNotFound(
                "Voice embedding unavailable for \(attemptedVoices.joined(separator: ", ")); checked \(sourceURL.path)"
            )
        }

        guard resolvedVector.count == expectedDimension else {
            throw TTSError.modelNotFound(
                "Voice embedding for \(voiceUsed) has unexpected length (expected \(expectedDimension), got \(resolvedVector.count))"
            )
        }

        var sumSquares: Float = 0
        resolvedVector.withUnsafeBufferPointer { sourcePointer in
            guard let baseAddress = sourcePointer.baseAddress, !sourcePointer.isEmpty else { return }
            vDSP_svesq(baseAddress, 1, &sumSquares, vDSP_Length(sourcePointer.count))
        }
        let norm = sqrt(Double(sumSquares))

        let formattedNorm = String(format: "%.3f", norm)
        let payloadCached = isVoiceEmbeddingPayloadCached(for: voiceUsed)
        if wasCached || payloadCached {
            Self.logger.debug(
                "Reusing cached voice embedding: \(voiceUsed), dim=\(expectedDimension), l2norm=\(formattedNorm)")
        } else {
            Self.logger.info("Loaded voice embedding: \(voiceUsed), dim=\(expectedDimension), l2norm=\(formattedNorm)")
        }

        return VoiceEmbeddingData(voiceID: voiceUsed, vector: resolvedVector, l2Norm: Float(norm))
    }

    static func isVoiceEmbeddingPayloadCached(for voiceID: String) -> Bool {
        voiceEmbeddingLock.lock()
        let cached = voiceEmbeddingPayloads[voiceID] != nil
        voiceEmbeddingLock.unlock()
        return cached
    }

    static func parseVoiceEmbeddingVector(
        _ json: Any,
        voiceID: String,
        phonemeCount: Int
    ) -> [Float]? {
        func parseArray(_ any: Any) -> [Float]? {
            if let doubles = any as? [Double] { return doubles.map(Float.init) }
            if let floats = any as? [Float] { return floats }
            if let numbers = any as? [NSNumber] { return numbers.map { $0.floatValue } }
            if let anyArray = any as? [Any] {
                var out: [Float] = []
                out.reserveCapacity(anyArray.count)
                for value in anyArray {
                    if let num = value as? NSNumber {
                        out.append(num.floatValue)
                    } else if let dbl = value as? Double {
                        out.append(Float(dbl))
                    } else if let flt = value as? Float {
                        out.append(flt)
                    } else {
                        return nil
                    }
                }
                return out
            }
            return nil
        }

        if let direct = parseArray(json) {
            return direct
        }

        guard let dict = json as? [String: Any] else { return nil }

        if let embed = dict["embedding"], let parsed = parseArray(embed) {
            return parsed
        }

        if let voiceSpecific = dict[voiceID], let parsed = parseArray(voiceSpecific) {
            return parsed
        }

        var numericCandidates: [(Int, [Float])] = []
        numericCandidates.reserveCapacity(dict.count)
        for (key, value) in dict {
            guard let intKey = Int(key), let parsed = parseArray(value) else { continue }
            numericCandidates.append((intKey, parsed))
        }

        numericCandidates.sort { $0.0 < $1.0 }

        if let exact = numericCandidates.first(where: { $0.0 == phonemeCount }) {
            return exact.1
        }

        if let fallback = numericCandidates.last(where: { $0.0 <= phonemeCount }) {
            return fallback.1
        }

        for value in dict.values {
            if let parsed = parseArray(value) {
                return parsed
            }
        }

        return nil
    }

    static func prepareVoiceEmbeddingCache(
        voice: String,
        entries: [ChunkEntry],
        embeddingDimension: Int
    ) throws -> [Int: VoiceEmbeddingData] {
        let uniqueCounts = Set(entries.map { $0.inputIds.count })
        var cache: [Int: VoiceEmbeddingData] = [:]
        cache.reserveCapacity(uniqueCounts.count)

        for count in uniqueCounts {
            cache[count] = try fetchVoiceEmbeddingData(
                voice: voice,
                phonemeCount: count,
                expectedDimension: embeddingDimension
            )
        }

        return cache
    }

    public static func loadVoiceEmbedding(
        voice: String = TtsConstants.recommendedVoice, phonemeCount: Int
    ) async throws -> MLMultiArray {
        let cache = try currentModelCache()
        let expectedDimension = try await cache.referenceEmbeddingDimension()
        let data = try fetchVoiceEmbeddingData(
            voice: voice,
            phonemeCount: phonemeCount,
            expectedDimension: expectedDimension
        )

        let embedding = try MLMultiArray(
            shape: [1, NSNumber(value: data.vector.count)] as [NSNumber],
            dataType: .float32
        )

        data.vector.withUnsafeBufferPointer { sourcePointer in
            guard let baseAddress = sourcePointer.baseAddress, !sourcePointer.isEmpty else { return }
            let destinationPointer = embedding.dataPointer.assumingMemoryBound(to: Float.self)
            destinationPointer.update(from: baseAddress, count: sourcePointer.count)
        }

        return embedding
    }

    internal static func refDim(from model: MLModel) -> Int {
        if let desc = model.modelDescription.inputDescriptionsByName["ref_s"],
            let shape = desc.multiArrayConstraint?.shape,
            shape.count >= 2
        {
            let n = shape.last!.intValue
            if n > 0 { return n }
        }
        return 256
    }
}
