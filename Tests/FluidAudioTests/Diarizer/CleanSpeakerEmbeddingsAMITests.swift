import Foundation
import XCTest

@testable import FluidAudio

/// `SpeakerUtilities.cleanEmbeddingsBySpeaker` on real meetings with the real `DiarizerManager` and
/// `OfflineDiarizerManager` models (the offline windows overlap 80%, so clean speech must count as audio).
///
/// Opt-in: set `FLUIDAUDIO_AMI_FIXTURES` to a directory of meetings, each `<meeting>/audio.wav` plus
/// `<meeting>/turns.json` (reference turns: `[{"speaker": "MEE071", "start": 0.0, "end": 1.37}, ...]`,
/// e.g. exported from the AMI corpus). Each chunk is labeled with the reference speaker who talks most inside
/// its span, and turns come from the reference, so the tests measure this function, not clustering.
final class CleanSpeakerEmbeddingsAMITests: XCTestCase {

    private struct Turn: Decodable, Sendable {
        let speaker: String
        let start: Double
        let end: Double
    }

    private struct Meeting: Sendable {
        let id: String
        let result: DiarizationResult
        let offline: DiarizationResult
        let turns: [Turn]

        /// `DiarizerManager`'s chunk embeddings, labeled with reference speakers and reference turns.
        var labeled: DiarizationResult { labeled(minTurn: 0) }

        /// `OfflineDiarizerManager`'s overlapping windows, labeled the same way.
        var offlineLabeled: DiarizationResult { Self.label(offline, turns: turns, minTurn: 0) }

        /// `labeled`, keeping only turns of `minTurn` seconds or more: `DiarizerManager` drops turns under
        /// `minSpeechDuration` (1 s), so other speakers' short interjections aren't subtracted.
        func labeled(minTurn: Double) -> DiarizationResult { Self.label(result, turns: turns, minTurn: minTurn) }

        static func label(_ result: DiarizationResult, turns: [Turn], minTurn: Double) -> DiarizationResult {
            DiarizationResult(
                segments: turns.filter { $0.end - $0.start >= minTurn }.map {
                    TimedSpeakerSegment(
                        speakerId: $0.speaker, embedding: [], startTimeSeconds: Float($0.start),
                        endTimeSeconds: Float($0.end), qualityScore: 1)
                },
                chunkEmbeddings: (result.chunkEmbeddings ?? []).map { chunk in
                    ChunkEmbedding(
                        speakerId: CleanSpeakerEmbeddingsAMITests.reference(
                            in: (chunk.startTimeSeconds, chunk.endTimeSeconds), turns: turns),
                        chunkIndex: chunk.chunkIndex, speakerIndex: chunk.speakerIndex,
                        startTimeSeconds: chunk.startTimeSeconds, endTimeSeconds: chunk.endTimeSeconds,
                        embedding256: chunk.embedding256)
                })
        }
    }

    /// Loaded once for every test; a task, so concurrent tests share one load.
    private static let loaded = Task<[Meeting], Error> { try await load() }

    private func meetings() async throws -> [Meeting] {
        guard ProcessInfo.processInfo.environment["FLUIDAUDIO_AMI_FIXTURES"] != nil else {
            throw XCTSkip("set FLUIDAUDIO_AMI_FIXTURES to run on real meetings")
        }
        let meetings = try await Self.loaded.value
        guard !meetings.isEmpty else { throw XCTSkip("no meetings with audio.wav and turns.json") }
        return meetings
    }

    private static func load() async throws -> [Meeting] {
        let root = ProcessInfo.processInfo.environment["FLUIDAUDIO_AMI_FIXTURES"] ?? ""
        let models = try await DiarizerModels.downloadIfNeeded()
        var config = DiarizerConfig.default
        config.exposeChunkEmbeddings = true
        let diarizer = DiarizerManager(config: config)
        diarizer.initialize(models: models)
        var offlineConfig = OfflineDiarizerConfig.default
        offlineConfig.exposeChunkEmbeddings = true
        let offlineDiarizer = OfflineDiarizerManager(config: offlineConfig)
        try await offlineDiarizer.prepareModels()

        var out: [Meeting] = []
        let dirs = try FileManager.default.contentsOfDirectory(
            at: URL(fileURLWithPath: (root as NSString).expandingTildeInPath), includingPropertiesForKeys: nil
        ).sorted { $0.lastPathComponent < $1.lastPathComponent }
        for dir in dirs {
            let audio = dir.appendingPathComponent("audio.wav")
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("turns.json")),
                FileManager.default.fileExists(atPath: audio.path)
            else { continue }
            let turns = try JSONDecoder().decode([Turn].self, from: data)
            guard Set(turns.map(\.speaker)).count >= 2 else { continue }
            let samples = try AudioConverter().resampleAudioFile(audio)
            let result = try diarizer.performCompleteDiarization(samples)
            let offline = try await offlineDiarizer.process(audio: samples)
            out.append(Meeting(id: dir.lastPathComponent, result: result, offline: offline, turns: turns))
        }
        return out
    }

    private static func reference(in span: (Double, Double), turns: [Turn]) -> String {
        var talk: [String: Double] = [:]
        for t in turns { talk[t.speaker, default: 0] += max(0, min(t.end, span.1) - max(t.start, span.0)) }
        return talk.max { $0.value < $1.value }?.key ?? ""
    }

    /// Relabels two diarizer speakers as one, as a clustering merge would.
    private func merging(_ a: String, _ b: String, in result: DiarizationResult) -> DiarizationResult {
        let rename = { (id: String) in id == a || id == b ? "merged" : id }
        return DiarizationResult(
            segments: result.segments.map {
                TimedSpeakerSegment(
                    speakerId: rename($0.speakerId), embedding: $0.embedding, startTimeSeconds: $0.startTimeSeconds,
                    endTimeSeconds: $0.endTimeSeconds, qualityScore: $0.qualityScore)
            },
            chunkEmbeddings: result.chunkEmbeddings?.map {
                ChunkEmbedding(
                    speakerId: rename($0.speakerId), chunkIndex: $0.chunkIndex, speakerIndex: $0.speakerIndex,
                    startTimeSeconds: $0.startTimeSeconds, endTimeSeconds: $0.endTimeSeconds,
                    embedding256: $0.embedding256)
            })
    }

    private func similarity(_ a: [Float], _ b: [Float]) -> Float { 1 - SpeakerUtilities.cosineDistance(a, b) }

    func testEmbeddingsIdentifyHeldOutChunks() async throws {
        var right = 0
        var total = 0
        for m in try await meetings() {
            let labeled = m.labeled
            let chunks = labeled.chunkEmbeddings ?? []
            guard let end = chunks.map(\.endTimeSeconds).max() else { continue }
            let early = DiarizationResult(
                segments: labeled.segments, chunkEmbeddings: chunks.filter { $0.endTimeSeconds <= end / 2 })
            let embeddings = SpeakerUtilities.cleanEmbeddingsBySpeaker(early)
            // Only late chunks that hold clean speech, as enrollment would see them.
            let late = DiarizationResult(
                segments: labeled.segments, chunkEmbeddings: chunks.filter { $0.startTimeSeconds > end / 2 })
            for chunk in late.chunkEmbeddings ?? [] where embeddings[chunk.speakerId] != nil {
                let alone = DiarizationResult(segments: labeled.segments, chunkEmbeddings: [chunk])
                guard !SpeakerUtilities.cleanEmbeddingsBySpeaker(alone, minTotalCleanSpeech: 0, minChunks: 1).isEmpty
                else {
                    continue
                }
                let best = embeddings.max {
                    similarity(chunk.embedding256, $0.value) < similarity(chunk.embedding256, $1.value)
                }!.key
                right += best == chunk.speakerId ? 1 : 0
                total += 1
            }
        }
        print("[clean-embeddings] held-out clean chunks identified \(right)/\(total)")
        XCTAssertGreaterThan(total, 50)
        // Measured 0.83 (2026-10-12). Some misses are label noise: a chunk's reference speaker is whoever
        // talks most inside its span.
        XCTAssertGreaterThan(Double(right) / Double(total), 0.75)
    }

    func testMergedSpeakersGetNoBlend() async throws {
        var withheld = 0
        var majority = 0
        for m in try await meetings() {
            let pure = SpeakerUtilities.cleanEmbeddingsBySpeaker(m.labeled)
            let ids = pure.keys.sorted()
            for (i, a) in ids.enumerated() {
                for b in ids[(i + 1)...] {
                    guard let merged = SpeakerUtilities.cleanEmbeddingsBySpeaker(merging(a, b, in: m.labeled))["merged"]
                    else {
                        withheld += 1
                        continue
                    }
                    majority += 1
                    let toA = similarity(merged, pure[a]!)
                    let toB = similarity(merged, pure[b]!)
                    let voices = similarity(pure[a]!, pure[b]!)
                    print("[clean-embeddings] \(m.id) \(a)+\(b): to A \(toA), to B \(toB), A to B \(voices)")
                    XCTAssertGreaterThan(max(toA, toB), 0.9, "\(m.id) \(a)+\(b): one of the two voices")
                    // Measured at most 0.05 at the default (2026-10-12).
                    XCTAssertLessThan(min(toA, toB), voices + 0.1, "\(m.id) \(a)+\(b): no pull toward the other")
                }
            }
        }
        print("[clean-embeddings] merged speakers: \(withheld) withheld, \(majority) took the majority voice")
        XCTAssertGreaterThan(withheld + majority, 5)
        XCTAssertGreaterThan(withheld, 0, "some merges must be caught")
    }

    /// Prints the trade-off `maxDistanceToMedoid` sets, for recalibration.
    func testDistanceSweep() async throws {
        let ms = try await meetings()
        for maxDistance: Float in [0.5, 0.55, 0.6, 0.65, 0.7] {
            var produced = 0
            var withheld = 0
            var worstLeak: Float = -1
            for m in ms {
                let pure = SpeakerUtilities.cleanEmbeddingsBySpeaker(m.labeled, maxDistanceToMedoid: maxDistance)
                produced += pure.count
                let reference = SpeakerUtilities.cleanEmbeddingsBySpeaker(m.labeled)
                let ids = reference.keys.sorted()
                for (i, a) in ids.enumerated() {
                    for b in ids[(i + 1)...] {
                        guard
                            let merged = SpeakerUtilities.cleanEmbeddingsBySpeaker(
                                merging(a, b, in: m.labeled), maxDistanceToMedoid: maxDistance)["merged"]
                        else {
                            withheld += 1
                            continue
                        }
                        let leak =
                            min(similarity(merged, reference[a]!), similarity(merged, reference[b]!))
                            - similarity(reference[a]!, reference[b]!)
                        worstLeak = max(worstLeak, leak)
                    }
                }
            }
            print(
                "[clean-embeddings] maxDistanceToMedoid \(maxDistance): \(produced) embeddings, "
                    + "merged withheld \(withheld), worst leak \(worstLeak)")
        }
    }

    /// With turns as `DiarizerManager` reports them (none under 1 s), short interjections by other speakers
    /// count as clean speech. The embeddings should barely move.
    func testShortInterjectionsBarelyMoveEmbeddings() async throws {
        var shifts: [Float] = []
        for m in try await meetings() {
            let full = SpeakerUtilities.cleanEmbeddingsBySpeaker(m.labeled)
            let filtered = SpeakerUtilities.cleanEmbeddingsBySpeaker(m.labeled(minTurn: 1))
            for (speaker, embedding) in full {
                if let other = filtered[speaker] { shifts.append(similarity(embedding, other)) }
            }
        }
        print("[clean-embeddings] similarity with vs without turns under 1 s: \(shifts.sorted())")
        XCTAssertFalse(shifts.isEmpty)
        XCTAssertGreaterThan(shifts.min() ?? 0, 0.95)
    }

    /// Pass rates for a pair of chunks at each distance, same speaker vs different speakers (printed).
    func testChunkPairDistances() async throws {
        var same: [Float] = []
        var other: [Float] = []
        for m in try await meetings() {
            let chunks = (m.labeled.chunkEmbeddings ?? []).filter { !$0.speakerId.isEmpty }
            for i in chunks.indices {
                for j in chunks.indices where j > i {
                    let d = SpeakerUtilities.cosineDistance(chunks[i].embedding256, chunks[j].embedding256)
                    if chunks[i].speakerId == chunks[j].speakerId { same.append(d) } else { other.append(d) }
                }
            }
        }
        for t: Float in [0.5, 0.6, 0.7] {
            let keep = Double(same.filter { $0 <= t }.count) / Double(max(same.count, 1))
            let leak = Double(other.filter { $0 <= t }.count) / Double(max(other.count, 1))
            print(
                String(
                    format: "[clean-embeddings] pair distance %.1f: same %.1f%%, different %.1f%%", t, 100 * keep,
                    100 * leak))
        }
    }

    /// The offline pipeline's windows overlap 80%: counted as audio, its embeddings should be the same
    /// voices as `DiarizerManager`'s and identify held-out windows as well.
    func testOfflineOverlappingWindows() async throws {
        var right = 0
        var total = 0
        var agreement: [Float] = []
        var withheld = 0
        var majority = 0
        var worstLeak: Float = -1
        for m in try await meetings() {
            let windows = m.offlineLabeled.chunkEmbeddings ?? []
            let spans = windows.map { $0.endTimeSeconds - $0.startTimeSeconds }
            print(
                "[clean-embeddings] \(m.id) offline: \(windows.count) windows, \(String(format: "%.1f", spans.max() ?? 0)) s long"
            )
            let offline = SpeakerUtilities.cleanEmbeddingsBySpeaker(m.offlineLabeled)
            let online = SpeakerUtilities.cleanEmbeddingsBySpeaker(m.labeled)
            for (speaker, e) in offline { if let o = online[speaker] { agreement.append(similarity(e, o)) } }

            guard let end = windows.map(\.endTimeSeconds).max() else { continue }
            let early = DiarizationResult(
                segments: m.offlineLabeled.segments, chunkEmbeddings: windows.filter { $0.endTimeSeconds <= end / 2 })
            let embeddings = SpeakerUtilities.cleanEmbeddingsBySpeaker(early)
            for w in windows where w.startTimeSeconds > end / 2 && embeddings[w.speakerId] != nil {
                let best = embeddings.max {
                    similarity(w.embedding256, $0.value) < similarity(w.embedding256, $1.value)
                }!.key
                right += best == w.speakerId ? 1 : 0
                total += 1
            }

            let ids = offline.keys.sorted()
            for (i, a) in ids.enumerated() {
                for b in ids[(i + 1)...] {
                    guard
                        let merged = SpeakerUtilities.cleanEmbeddingsBySpeaker(merging(a, b, in: m.offlineLabeled))[
                            "merged"]
                    else {
                        withheld += 1
                        continue
                    }
                    majority += 1
                    let leak =
                        min(similarity(merged, offline[a]!), similarity(merged, offline[b]!))
                        - similarity(offline[a]!, offline[b]!)
                    worstLeak = max(worstLeak, leak)
                }
            }
        }
        print(
            "[clean-embeddings] offline vs DiarizerManager embeddings, same speaker: \(agreement.sorted().map { String(format: "%.2f", $0) })"
        )
        print(
            "[clean-embeddings] offline held-out windows identified \(right)/\(total); merged: \(withheld) withheld, \(majority) majority, worst leak \(worstLeak)"
        )
        XCTAssertGreaterThan(total, 50)
        XCTAssertGreaterThan(Double(right) / Double(total), 0.75)
        XCTAssertLessThan(worstLeak, 0.1, "no merged pair blends toward the other voice")
    }
}
