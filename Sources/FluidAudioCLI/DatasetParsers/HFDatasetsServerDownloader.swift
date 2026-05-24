#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Downloads audio + transcripts from gated HuggingFace datasets (MCV-17,
/// MLS) via the public `datasets-server.huggingface.co` REST API. Paginates
/// `/rows`, then concurrently downloads each clip's audio URL with skip-
/// if-exists semantics so partial runs are resumable.
///
/// Writes the LibriSpeech-style layout that
/// `NemotronMultilingualFleursBenchmark.loadSamples(...)` expects:
///
/// ```
/// <cacheRoot>/<dataset-subdir>/<lang>/
///     <lang>_0000.{mp3|flac}
///     <lang>_0001.{mp3|flac}
///     ...
///     <lang>.trans.txt   # "fileId<TAB>transcript" per line
/// ```
///
/// Both datasets are gated on HuggingFace. Set `HF_TOKEN` (or
/// `HUGGING_FACE_HUB_TOKEN`) to a token that has accepted the dataset's
/// terms-of-service. 401/403 surfaces a clear error with the accept-page URL.
public enum HFDatasetsServerDownloader {

    private static let logger = AppLogger(category: "HFDatasetsServerDownloader")
    private static let pageSize = 100
    private static let maxConcurrentDownloads = 16

    /// Errors specific to the datasets-server path.
    public enum Error: LocalizedError {
        case gatedDataset(dataset: String, acceptURL: String)
        case viewerDisabled(dataset: String)
        case parseFailed(message: String)
        case audioUrlMissing(rowIndex: Int)
        case downloadFailed(path: String, underlying: Swift.Error)

        public var errorDescription: String? {
            switch self {
            case .gatedDataset(let dataset, let acceptURL):
                return
                    "HuggingFace returned 401/403 for gated dataset \(dataset). "
                    + "Accept terms at \(acceptURL) and set HF_TOKEN to a token that has accepted them."
            case .viewerDisabled(let dataset):
                let script: String
                if dataset.contains("common_voice") {
                    script = "Scripts/prep_mcv17.py (downloads from fsicoli/common_voice_17_0 mirror)"
                } else if dataset.contains("multilingual_librispeech") {
                    script = "Scripts/prep_mls_flac.py (downloads original flac from data/mls_<lang>/test/)"
                } else {
                    script = "Scripts/prep_<dataset>.py"
                }
                return
                    "Dataset \(dataset) is not auto-downloadable via the HF datasets-server "
                    + "(viewer disabled or only lossy transcodes are exposed). "
                    + "Pre-populate the cache by running \(script)."
            case .parseFailed(let message):
                return "Failed to parse datasets-server response: \(message)"
            case .audioUrlMissing(let idx):
                return "Row \(idx) returned no audio src URL"
            case .downloadFailed(let path, let underlying):
                return "Failed to download \(path): \(underlying.localizedDescription)"
            }
        }
    }

    // MARK: - Public entry points

    /// Download MCV-17 test split for the given FLEURS-style language codes.
    public static func downloadMCV(
        languages: [String],
        cacheRoot: URL,
        samplesPerLanguage: Int
    ) async throws {
        try await download(
            dataset: .mcv,
            transcriptKey: "sentence",
            languages: languages,
            cacheRoot: cacheRoot,
            samplesPerLanguage: samplesPerLanguage
        )
    }

    /// Download MLS test split for the given FLEURS-style language codes.
    public static func downloadMLS(
        languages: [String],
        cacheRoot: URL,
        samplesPerLanguage: Int
    ) async throws {
        try await download(
            dataset: .mls,
            transcriptKey: "transcript",
            languages: languages,
            cacheRoot: cacheRoot,
            samplesPerLanguage: samplesPerLanguage
        )
    }

    // MARK: - Core implementation

    private static func download(
        dataset: MultilingualBenchmarkDataset,
        transcriptKey: String,
        languages: [String],
        cacheRoot: URL,
        samplesPerLanguage: Int
    ) async throws {
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)

        for lang in languages {
            guard let cfgName = dataset.hfConfigName(forFleursCode: lang) else {
                logger.warning(
                    "Skipping \(lang): not in \(dataset.rawValue) supported set (en_us, es_419, fr_fr, it_it, pt_br)"
                )
                continue
            }

            let langDir = cacheRoot.appendingPathComponent(lang)
            try FileManager.default.createDirectory(at: langDir, withIntermediateDirectories: true)

            let transFile = langDir.appendingPathComponent("\(lang).trans.txt")
            let limit = samplesPerLanguage == Int.max ? Int.max : samplesPerLanguage

            // Quick skip path: if transcript exists and we already have audio
            // on disk, treat as cached. For `--samples all` (limit=Int.max) we
            // trust the existing cache — re-fetching the whole split just to
            // confirm completeness would trigger datasets-server rate limiting.
            // For explicit limits we still require audioCount >= limit.
            // (Strict file validation happens at load time via AVAudioFile.)
            if FileManager.default.fileExists(atPath: transFile.path) {
                let existingAudio =
                    (try? FileManager.default.contentsOfDirectory(at: langDir, includingPropertiesForKeys: nil)) ?? []
                let audioCount = existingAudio.filter { $0.pathExtension == dataset.audioExtension }.count
                let satisfied = limit == Int.max ? audioCount > 0 : audioCount >= limit
                if satisfied && audioCount > 0 {
                    logger.info("\(dataset.rawValue)/\(lang): cached (\(audioCount) clips)")
                    continue
                }
            }

            // MCV-17 has the HF datasets-server viewer disabled, so we cannot
            // auto-download via the per-row API.
            //
            // MLS via datasets-server only exposes the lossy ogg/opus
            // transcode, which costs 1-3pp WER vs the original flac. We pin
            // MLS to flac (see MultilingualBenchmarkDataset.audioExtension)
            // and require the user to pre-populate via the prep script.
            //
            // Both paths require Scripts/prep_<dataset>.py to fill the cache
            // before benchmarks can run.
            if dataset == .mcv || dataset == .mls {
                throw Error.viewerDisabled(dataset: dataset.hfRepo)
            }

            logger.info("\(dataset.rawValue)/\(lang): downloading metadata from \(dataset.hfRepo) [\(cfgName)]")

            let rows = try await fetchAllRows(
                dataset: dataset,
                configName: cfgName,
                limit: limit
            )
            logger.info("\(dataset.rawValue)/\(lang): got \(rows.count) row metadata entries")

            try await downloadClips(
                dataset: dataset,
                language: lang,
                rows: rows,
                transcriptKey: transcriptKey,
                langDir: langDir,
                transFile: transFile
            )
        }
    }

    // MARK: - Pagination

    private struct RowMetadata {
        let rowIndex: Int
        let audioURL: URL
        let transcript: String
    }

    /// Fetch up to `limit` rows from the dataset's `test` split by paginating
    /// the datasets-server `/rows` endpoint. The `info` endpoint is used to
    /// discover the split's total row count.
    private static func fetchAllRows(
        dataset: MultilingualBenchmarkDataset,
        configName: String,
        limit: Int
    ) async throws -> [(rowIdx: Int, row: [String: Any])] {
        var out: [(Int, [String: Any])] = []
        var offset = 0
        // The datasets-server caps `length` to 100 per request. Loop until we
        // hit an empty page or the requested limit.
        while out.count < limit {
            let length = min(pageSize, limit - out.count)
            let urlString =
                "https://datasets-server.huggingface.co/rows"
                + "?dataset=\(percentEncode(dataset.hfRepo))"
                + "&config=\(percentEncode(configName))"
                + "&split=test"
                + "&offset=\(offset)"
                + "&length=\(length)"
            guard let url = URL(string: urlString) else {
                throw Error.parseFailed(message: "bad URL: \(urlString)")
            }

            let (data, response) = try await DownloadUtils.fetchWithAuth(from: url)
            if let http = response as? HTTPURLResponse {
                if http.statusCode == 401 || http.statusCode == 403 {
                    throw Error.gatedDataset(dataset: dataset.hfRepo, acceptURL: dataset.acceptTermsURL)
                }
                guard (200..<300).contains(http.statusCode) else {
                    let body = String(data: data, encoding: .utf8)?.prefix(200) ?? ""
                    throw Error.parseFailed(
                        message: "HTTP \(http.statusCode) from datasets-server: \(body)")
                }
            }

            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let rows = json["rows"] as? [[String: Any]]
            else {
                let body = String(data: data, encoding: .utf8)?.prefix(200) ?? ""
                throw Error.parseFailed(message: "missing 'rows' array: \(body)")
            }
            if rows.isEmpty {
                break  // end of split
            }
            for entry in rows {
                guard let idx = entry["row_idx"] as? Int,
                    let row = entry["row"] as? [String: Any]
                else { continue }
                out.append((idx, row))
                if out.count >= limit { break }
            }
            offset += rows.count
            // Defensive cap: a misbehaving server returning a non-empty page
            // forever would loop indefinitely.
            if rows.count < length { break }
        }
        return out
    }

    // MARK: - Concurrent clip downloads

    private static func downloadClips(
        dataset: MultilingualBenchmarkDataset,
        language: String,
        rows: [(rowIdx: Int, row: [String: Any])],
        transcriptKey: String,
        langDir: URL,
        transFile: URL
    ) async throws {
        // Extract (sampleId, audioURL, transcript) for each row up front so
        // we have a final ordered list to feed into the concurrent group.
        var planned: [RowMetadata] = []
        planned.reserveCapacity(rows.count)
        for (offset, (rowIdx, row)) in rows.enumerated() {
            guard
                let audio = row["audio"] as? [[String: Any]],
                let first = audio.first,
                let src = first["src"] as? String,
                let url = URL(string: src)
            else {
                // Some datasets nest the URL as a single dict instead of array.
                if let audio = row["audio"] as? [String: Any],
                    let src = audio["src"] as? String,
                    let url = URL(string: src)
                {
                    let transcript = (row[transcriptKey] as? String) ?? ""
                    planned.append(RowMetadata(rowIndex: offset, audioURL: url, transcript: transcript))
                    continue
                }
                logger.warning("Row \(rowIdx) has no audio.src; skipping")
                continue
            }
            let transcript = (row[transcriptKey] as? String) ?? ""
            planned.append(RowMetadata(rowIndex: offset, audioURL: url, transcript: transcript))
        }

        // Write transcript file up front so loadSamples sees it even if the
        // audio download is interrupted partway through.
        let transcriptLines = planned.map { meta -> String in
            let id = sampleId(language: language, index: meta.rowIndex)
            // LibriSpeech-style: id<SPACE>text<NEWLINE> (single-space separator
            // matches FLEURSBenchmark's parser which uses split(maxSplits: 1)).
            return "\(id) \(meta.transcript)"
        }
        try transcriptLines.joined(separator: "\n").write(to: transFile, atomically: true, encoding: .utf8)

        // Concurrent download in fixed-size batches. Batching (rather than a
        // sliding window) avoids Swift 6 sending-closure capture issues that
        // arise when the task group's submission loop mutates shared state.
        var downloaded = 0
        var skipped = 0
        let ext = dataset.audioExtension
        let datasetTag = dataset.rawValue
        let total = planned.count

        for batchStart in stride(from: 0, to: planned.count, by: maxConcurrentDownloads) {
            let batchEnd = min(batchStart + maxConcurrentDownloads, planned.count)
            let batch = Array(planned[batchStart..<batchEnd])

            let results: [(Int, Bool)] = try await withThrowingTaskGroup(of: (Int, Bool).self) { group in
                for meta in batch {
                    let id = sampleId(language: language, index: meta.rowIndex)
                    let dst = langDir.appendingPathComponent("\(id).\(ext)")
                    let audioURL = meta.audioURL
                    let rowIndex = meta.rowIndex
                    let description = "\(datasetTag)/\(language)/\(id)"
                    group.addTask {
                        // Skip if exists. Validation deferred to load time
                        // (AVAudioFile.forReading: catches corrupt mp3/flac).
                        if FileManager.default.fileExists(atPath: dst.path) {
                            return (rowIndex, false)  // not newly downloaded
                        }
                        do {
                            let data = try await DownloadUtils.fetchHuggingFaceFile(
                                from: audioURL,
                                description: description
                            )
                            try data.write(to: dst, options: .atomic)
                            return (rowIndex, true)
                        } catch {
                            throw Error.downloadFailed(path: id, underlying: error)
                        }
                    }
                }
                var out: [(Int, Bool)] = []
                for try await result in group { out.append(result) }
                return out
            }

            for (_, isNew) in results {
                if isNew { downloaded += 1 } else { skipped += 1 }
            }
            let done = downloaded + skipped
            if done % 100 < maxConcurrentDownloads || done == total {
                logger.info(
                    "\(datasetTag)/\(language): \(done)/\(total) "
                        + "(\(downloaded) new, \(skipped) cached)")
            }
        }

        logger.info(
            "\(dataset.rawValue)/\(language): done (\(downloaded) downloaded, \(skipped) already cached, \(total) total)"
        )
    }

    // MARK: - Helpers

    /// Stable sample ID for transcript + filename. Zero-padded index keeps
    /// `sort` order intact and matches the FLEURS / JSUT naming convention.
    private static func sampleId(language: String, index: Int) -> String {
        let padded = String(format: "%05d", index)
        return "\(language)_\(padded)"
    }

    private static func percentEncode(_ s: String) -> String {
        s.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? s
    }
}
#endif
