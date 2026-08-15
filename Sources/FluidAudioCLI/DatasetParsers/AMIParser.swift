#if os(macOS)
import FluidAudio
import Foundation

/// Errors thrown when AMI reference annotations cannot be loaded.
///
/// Benchmarks must fail loudly on these instead of scoring against a synthetic
/// reference: a transient annotation download failure once produced a bogus
/// 80.8% DER report scored against placeholder ground truth (issue #752).
enum AMIParserError: Error, LocalizedError {
    case annotationsNotFound(subdirectory: String)

    var errorDescription: String? {
        switch self {
        case .annotationsNotFound(let subdirectory):
            return
                "AMI annotations not found in any expected location. "
                + "Expected structure: [path]/\(subdirectory)/ AND [path]/corpusResources/meetings.xml. "
                + "Run with --auto-download or download manually from https://groups.inf.ed.ac.uk/ami/download/"
        }
    }
}

/// AMI annotation parser and ground truth handling
struct AMIParser {
    private static let logger = AppLogger(category: "AMIParser")
    private static let defaultMergeGapSeconds = 0.5
    private static let defaultReferenceFrameStepSeconds = 0.01

    /// Get ground truth speaker count from AMI meetings.xml
    static func getGroundTruthSpeakerCount(for meetingId: String) -> Int {
        for location in possibleAnnotationRoots() {
            let meetingsFile = location.appendingPathComponent("corpusResources/meetings.xml")
            if FileManager.default.fileExists(atPath: meetingsFile.path) {
                do {
                    let xmlData = try Data(contentsOf: meetingsFile)
                    let xmlString = String(data: xmlData, encoding: .utf8) ?? ""

                    // Find the meeting entry for this meetingId
                    if let meetingRange = xmlString.range(of: "observation=\"\(meetingId)\"") {
                        let afterObservation = xmlString[meetingRange.upperBound...]

                        // Count speaker elements within this meeting
                        if let meetingEndRange = afterObservation.range(of: "</meeting>") {
                            let meetingContent = String(
                                afterObservation[..<meetingEndRange.lowerBound])
                            let speakerCount =
                                meetingContent.components(separatedBy: "<speaker ").count - 1
                            return speakerCount
                        }
                    }
                } catch {
                    continue
                }
            }
        }

        // Default fallback for unknown meetings
        return 4  // AMI meetings typically have 4 speakers
    }

    /// Load AMI ground truth annotations for a specific meeting.
    ///
    /// Throws if annotations are missing or unparsable — never substitutes a
    /// placeholder reference, so callers cannot silently score against fake data.
    static func loadAMIGroundTruth(
        for meetingId: String,
        duration: Float,
        searchRoots: [URL]? = nil
    ) throws -> [TimedSpeakerSegment] {
        guard
            let validAmiDir = findAnnotationRoot(
                requiringSubdirectory: "segments", searchRoots: searchRoots)
        else {
            throw AMIParserError.annotationsNotFound(subdirectory: "segments")
        }

        logger.info("   📖 Loading AMI annotations for meeting: \(meetingId)")

        let allSegments = try loadAMIGroundTruth(
            for: meetingId,
            in: validAmiDir,
            duration: duration
        )
        logger.info("      Total segments loaded: \(allSegments.count)")
        return allSegments
    }

    /// Internal hook for tests and benchmark helpers that need deterministic parsing
    /// from a specific AMI annotation root.
    static func loadAMIGroundTruth(
        for meetingId: String,
        in amiDirectory: URL,
        duration: Float
    ) throws -> [TimedSpeakerSegment] {
        _ = duration
        return try loadOfficialGroundTruth(
            for: meetingId,
            in: amiDirectory,
            filterShortSegments: true
        )
    }

    private static func loadOfficialGroundTruth(
        for meetingId: String,
        in amiDirectory: URL,
        filterShortSegments: Bool
    ) throws -> [TimedSpeakerSegment] {
        let segmentsDir = amiDirectory.appendingPathComponent("segments")
        let meetingsFile = amiDirectory.appendingPathComponent("corpusResources/meetings.xml")
        let parser = AMIAnnotationParser()

        guard
            let speakerMapping = try parser.parseSpeakerMapping(
                for: meetingId,
                from: meetingsFile
            )
        else {
            throw NSError(
                domain: "AMIParser",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "No speaker mapping found for \(meetingId)"]
            )
        }

        logger.info(
            "      Speaker mapping: A=\(speakerMapping.speakerA), B=\(speakerMapping.speakerB), C=\(speakerMapping.speakerC), D=\(speakerMapping.speakerD)"
        )

        var allSegments: [TimedSpeakerSegment] = []

        for speakerCode in ["A", "B", "C", "D"] {
            let segmentFile = segmentsDir.appendingPathComponent("\(meetingId).\(speakerCode).segments.xml")
            guard FileManager.default.fileExists(atPath: segmentFile.path) else { continue }
            guard let participantId = speakerMapping.participantId(for: speakerCode) else { continue }

            let segments = try parser.parseSegmentsFile(segmentFile)
            for segment in segments where segment.duration > 0 {
                if filterShortSegments && segment.duration < 0.5 {
                    continue
                }
                allSegments.append(
                    TimedSpeakerSegment(
                        speakerId: participantId,
                        embedding: generatePlaceholderEmbedding(for: participantId),
                        startTimeSeconds: Float(segment.startTime),
                        endTimeSeconds: Float(segment.endTime),
                        qualityScore: 1.0
                    )
                )
            }

            logger.info(
                "      Loaded \(segments.count) segments for speaker \(speakerCode) (\(participantId))"
            )
        }

        allSegments.sort {
            if $0.startTimeSeconds == $1.startTimeSeconds {
                if $0.endTimeSeconds == $1.endTimeSeconds {
                    return $0.speakerId < $1.speakerId
                }
                return $0.endTimeSeconds < $1.endTimeSeconds
            }
            return $0.startTimeSeconds < $1.startTimeSeconds
        }
        return allSegments
    }

    /// Load AMI annotations as a 10 ms frame-quantized DER reference, matching the
    /// original Kaldi-style label construction used by the LS-EEND repo.
    static func loadFrameAlignedDERReference(
        for meetingId: String,
        duration: Float,
        frameStep: Double = defaultReferenceFrameStepSeconds,
        searchRoots: [URL]? = nil
    ) throws -> [DERSpeakerSegment] {
        guard
            let validAmiDir = findAnnotationRoot(
                requiringSubdirectory: "segments", searchRoots: searchRoots)
        else {
            throw AMIParserError.annotationsNotFound(subdirectory: "segments")
        }

        return try loadFrameAlignedDERReference(
            for: meetingId,
            in: validAmiDir,
            duration: duration,
            frameStep: frameStep
        )
    }

    static func loadFrameAlignedDERReference(
        for meetingId: String,
        in amiDirectory: URL,
        duration: Float,
        frameStep: Double = defaultReferenceFrameStepSeconds
    ) throws -> [DERSpeakerSegment] {
        _ = duration
        let segments = try loadOfficialGroundTruth(
            for: meetingId,
            in: amiDirectory,
            filterShortSegments: false
        )
        return frameAlignedDERReference(from: segments, frameStep: frameStep)
    }

    /// Load AMI word-aligned ground truth annotations for a specific meeting.
    ///
    /// Uses forced-alignment `{meeting}.{A|B|C|D}.words.xml` files and merges
    /// adjacent same-speaker words with gaps up to `mergeGap`.
    static func loadWordAlignedGroundTruth(
        for meetingId: String,
        duration: Float,
        mergeGap: Double = defaultMergeGapSeconds,
        searchRoots: [URL]? = nil
    ) throws -> [TimedSpeakerSegment] {
        guard
            let validAmiDir = findAnnotationRoot(
                requiringSubdirectory: "words", searchRoots: searchRoots)
        else {
            throw AMIParserError.annotationsNotFound(subdirectory: "words")
        }

        return try loadWordAlignedGroundTruth(
            for: meetingId,
            in: validAmiDir,
            duration: duration,
            mergeGap: mergeGap
        )
    }

    /// Internal hook for tests and benchmark helpers that need deterministic parsing
    /// from a specific AMI annotation root.
    static func loadWordAlignedGroundTruth(
        for meetingId: String,
        in amiDirectory: URL,
        duration: Float,
        mergeGap: Double = defaultMergeGapSeconds
    ) throws -> [TimedSpeakerSegment] {
        let wordsDir = amiDirectory.appendingPathComponent("words")
        let meetingsFile = amiDirectory.appendingPathComponent("corpusResources/meetings.xml")

        let parser = AMIAnnotationParser()
        guard
            let speakerMapping = try parser.parseSpeakerMapping(
                for: meetingId,
                from: meetingsFile
            )
        else {
            throw NSError(
                domain: "AMIParser",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "No speaker mapping found for \(meetingId)"]
            )
        }

        var allSegments: [TimedSpeakerSegment] = []
        for speakerCode in ["A", "B", "C", "D"] {
            let wordsFile = wordsDir.appendingPathComponent("\(meetingId).\(speakerCode).words.xml")
            guard FileManager.default.fileExists(atPath: wordsFile.path) else { continue }
            guard let participantId = speakerMapping.participantId(for: speakerCode) else { continue }

            let words = try parser.parseWordsFile(wordsFile)
            for segment in mergeSegments(words, mergeGap: mergeGap) {
                allSegments.append(
                    TimedSpeakerSegment(
                        speakerId: participantId,
                        embedding: generatePlaceholderEmbedding(for: participantId),
                        startTimeSeconds: Float(segment.startTime),
                        endTimeSeconds: Float(segment.endTime),
                        qualityScore: 1.0
                    )
                )
            }
        }

        allSegments.sort { $0.startTimeSeconds < $1.startTimeSeconds }
        return allSegments
    }

    static func loadWordAlignedDERReference(
        for meetingId: String,
        duration: Float,
        mergeGap: Double = defaultMergeGapSeconds,
        searchRoots: [URL]? = nil
    ) throws -> [DERSpeakerSegment] {
        let segments = try loadWordAlignedGroundTruth(
            for: meetingId,
            duration: duration,
            mergeGap: mergeGap,
            searchRoots: searchRoots
        )
        return segments.map {
            DERSpeakerSegment(
                speaker: $0.speakerId,
                start: Double($0.startTimeSeconds),
                end: Double($0.endTimeSeconds)
            )
        }
    }

    static func loadWordAlignedDERReference(
        for meetingId: String,
        in amiDirectory: URL,
        duration: Float,
        mergeGap: Double = defaultMergeGapSeconds
    ) throws -> [DERSpeakerSegment] {
        let segments = try loadWordAlignedGroundTruth(
            for: meetingId,
            in: amiDirectory,
            duration: duration,
            mergeGap: mergeGap
        )
        return segments.map {
            DERSpeakerSegment(
                speaker: $0.speakerId,
                start: Double($0.startTimeSeconds),
                end: Double($0.endTimeSeconds)
            )
        }
    }

    /// Generate consistent placeholder embeddings for each speaker
    static func generatePlaceholderEmbedding(for participantId: String) -> [Float] {
        // Generate a consistent embedding based on participant ID
        let hash = participantId.hashValue
        let seed = abs(hash) % 1000

        var embedding: [Float] = []
        for i in 0..<512 {  // Match expected embedding size
            let value = Float(sin(Double(seed + i * 37))) * 0.5 + 0.5
            embedding.append(value)
        }
        return embedding
    }

    private static func possibleAnnotationRoots() -> [URL] {
        [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Datasets/ami_public_1.6.2"),
            URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Datasets/ami_public_1.6.2"),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Tests/ami_public_1.6.2"),
            URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Tests/ami_public_1.6.2"),
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
                "code/FluidAudio/Tests/ami_public_1.6.2"
            ),
        ]
    }

    private static func findAnnotationRoot(
        requiringSubdirectory subdirectory: String,
        searchRoots: [URL]? = nil
    ) -> URL? {
        for path in searchRoots ?? possibleAnnotationRoots() {
            let requiredDir = path.appendingPathComponent(subdirectory)
            let meetingsFile = path.appendingPathComponent("corpusResources/meetings.xml")
            let hasRequiredDir = FileManager.default.fileExists(atPath: requiredDir.path)
            let hasMeetings = FileManager.default.fileExists(atPath: meetingsFile.path)
            if hasRequiredDir && hasMeetings {
                logger.info("      - 🎯 SELECTED: \(path.path)")
                return path
            }
        }
        return nil
    }

    private static func mergeSegments(
        _ segments: [AMISpeakerSegment],
        mergeGap: Double
    ) -> [AMISpeakerSegment] {
        let sorted = segments.sorted { $0.startTime < $1.startTime }
        guard var current = sorted.first else { return [] }

        var merged: [AMISpeakerSegment] = []
        for next in sorted.dropFirst() {
            if next.startTime - current.endTime <= mergeGap {
                current = AMISpeakerSegment(
                    segmentId: current.segmentId,
                    participantId: current.participantId,
                    startTime: current.startTime,
                    endTime: max(current.endTime, next.endTime)
                )
                continue
            }
            merged.append(current)
            current = next
        }

        merged.append(current)
        return merged
    }

    private static func frameAlignedDERReference(
        from segments: [TimedSpeakerSegment],
        frameStep: Double
    ) -> [DERSpeakerSegment] {
        precondition(frameStep > 0)

        var intervalsBySpeaker: [String: [(startFrame: Int, endFrame: Int)]] = [:]
        for segment in segments {
            let startFrame = Int(
                (Double(segment.startTimeSeconds) / frameStep).rounded(.toNearestOrEven)
            )
            let endFrame = Int(
                (Double(segment.endTimeSeconds) / frameStep).rounded(.toNearestOrEven)
            )
            guard endFrame > startFrame else { continue }
            intervalsBySpeaker[segment.speakerId, default: []].append((startFrame, endFrame))
        }

        var derSegments: [DERSpeakerSegment] = []
        for (speaker, intervals) in intervalsBySpeaker {
            let sortedIntervals = intervals.sorted {
                if $0.startFrame == $1.startFrame {
                    return $0.endFrame < $1.endFrame
                }
                return $0.startFrame < $1.startFrame
            }
            guard var current = sortedIntervals.first else { continue }

            for next in sortedIntervals.dropFirst() {
                guard next.startFrame > current.endFrame else {
                    current.endFrame = max(current.endFrame, next.endFrame)
                    continue
                }

                derSegments.append(
                    DERSpeakerSegment(
                        speaker: speaker,
                        start: Double(current.startFrame) * frameStep,
                        end: Double(current.endFrame) * frameStep
                    )
                )
                current = next
            }

            derSegments.append(
                DERSpeakerSegment(
                    speaker: speaker,
                    start: Double(current.startFrame) * frameStep,
                    end: Double(current.endFrame) * frameStep
                )
            )
        }

        derSegments.sort {
            if $0.start == $1.start {
                if $0.end == $1.end {
                    return $0.speaker < $1.speaker
                }
                return $0.end < $1.end
            }
            return $0.start < $1.start
        }
        return derSegments
    }
}

// MARK: - AMI Annotation Data Structures

/// Represents a single AMI speaker segment from NXT format
struct AMISpeakerSegment {
    let segmentId: String  // e.g., "EN2001a.sync.4"
    let participantId: String  // e.g., "FEE005" (mapped from A/B/C/D)
    let startTime: Double  // Start time in seconds
    let endTime: Double  // End time in seconds

    var duration: Double {
        return endTime - startTime
    }
}

/// Maps AMI speaker codes (A/B/C/D) to real participant IDs
struct AMISpeakerMapping {
    let meetingId: String
    let speakerA: String  // e.g., "MEE006"
    let speakerB: String  // e.g., "FEE005"
    let speakerC: String  // e.g., "MEE007"
    let speakerD: String  // e.g., "MEE008"

    func participantId(for speakerCode: String) -> String? {
        switch speakerCode.uppercased() {
        case "A": return speakerA
        case "B": return speakerB
        case "C": return speakerC
        case "D": return speakerD
        default: return nil
        }
    }
}

/// Parser for AMI NXT XML annotation files
class AMIAnnotationParser: NSObject {

    /// Parse segments.xml file and return speaker segments
    func parseSegmentsFile(_ xmlFile: URL) throws -> [AMISpeakerSegment] {
        let data = try Data(contentsOf: xmlFile)

        // Extract speaker code from filename (e.g., "EN2001a.A.segments.xml" -> "A")
        let speakerCode = extractSpeakerCodeFromFilename(xmlFile.lastPathComponent)

        let parser = XMLParser(data: data)
        let delegate = AMISegmentsXMLDelegate(speakerCode: speakerCode)
        parser.delegate = delegate

        guard parser.parse() else {
            throw NSError(
                domain: "AMIParser", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Failed to parse XML file: \(xmlFile.lastPathComponent)"
                ])
        }

        if let error = delegate.parsingError {
            throw error
        }

        return delegate.segments
    }

    /// Parse words.xml file and return word-level speaker segments.
    func parseWordsFile(_ xmlFile: URL) throws -> [AMISpeakerSegment] {
        let data = try Data(contentsOf: xmlFile)
        let speakerCode = extractSpeakerCodeFromFilename(xmlFile.lastPathComponent)

        let parser = XMLParser(data: data)
        let delegate = AMIWordsXMLDelegate(speakerCode: speakerCode)
        parser.delegate = delegate

        guard parser.parse() else {
            throw NSError(
                domain: "AMIParser",
                code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Failed to parse XML file: \(xmlFile.lastPathComponent)"
                ]
            )
        }

        if let error = delegate.parsingError {
            throw error
        }

        return delegate.segments
    }

    /// Extract speaker code from AMI filename
    private func extractSpeakerCodeFromFilename(_ filename: String) -> String {
        // Filename format: "EN2001a.A.segments.xml" -> extract "A"
        let components = filename.components(separatedBy: ".")
        if components.count >= 3 {
            return components[1]  // The speaker code is the second component
        }
        return "UNKNOWN"
    }

    /// Parse meetings.xml to get speaker mappings for a specific meeting
    func parseSpeakerMapping(
        for meetingId: String, from meetingsFile: URL
    ) throws
        -> AMISpeakerMapping?
    {
        let data = try Data(contentsOf: meetingsFile)

        let parser = XMLParser(data: data)
        let delegate = AMIMeetingsXMLDelegate(targetMeetingId: meetingId)
        parser.delegate = delegate

        guard parser.parse() else {
            throw NSError(
                domain: "AMIParser", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to parse meetings.xml"])
        }

        if let error = delegate.parsingError {
            throw error
        }

        return delegate.speakerMapping
    }
}

/// XML parser delegate for AMI words files
private class AMIWordsXMLDelegate: NSObject, XMLParserDelegate {
    var segments: [AMISpeakerSegment] = []
    var parsingError: Error?

    private let speakerCode: String

    init(speakerCode: String) {
        self.speakerCode = speakerCode
    }

    func parser(
        _ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?,
        qualifiedName qName: String?, attributes attributeDict: [String: String] = [:]
    ) {
        let tag = elementName.split(separator: ":").last.map(String.init) ?? elementName
        guard tag == "w",
            attributeDict["punc"] != "true",
            let startTimeString = attributeDict["starttime"],
            let endTimeString = attributeDict["endtime"],
            let startTime = Double(startTimeString),
            let endTime = Double(endTimeString),
            endTime > startTime
        else {
            return
        }

        segments.append(
            AMISpeakerSegment(
                segmentId: attributeDict["nite:id"] ?? UUID().uuidString,
                participantId: speakerCode,
                startTime: startTime,
                endTime: endTime
            )
        )
    }

    func parser(_ parser: XMLParser, parseErrorOccurred parseError: Error) {
        parsingError = parseError
    }
}

/// XML parser delegate for AMI segments files
private class AMISegmentsXMLDelegate: NSObject, XMLParserDelegate {
    var segments: [AMISpeakerSegment] = []
    var parsingError: Error?

    private let speakerCode: String

    init(speakerCode: String) {
        self.speakerCode = speakerCode
    }

    func parser(
        _ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?,
        qualifiedName qName: String?, attributes attributeDict: [String: String] = [:]
    ) {

        if elementName == "segment" {
            // Extract segment attributes
            guard let segmentId = attributeDict["nite:id"],
                let startTimeStr = attributeDict["transcriber_start"],
                let endTimeStr = attributeDict["transcriber_end"],
                let startTime = Double(startTimeStr),
                let endTime = Double(endTimeStr)
            else {
                return  // Skip invalid segments
            }

            let segment = AMISpeakerSegment(
                segmentId: segmentId,
                participantId: speakerCode,  // Use speaker code from filename
                startTime: startTime,
                endTime: endTime
            )

            segments.append(segment)
        }
    }

    func parser(_ parser: XMLParser, parseErrorOccurred parseError: Error) {
        parsingError = parseError
    }
}

/// XML parser delegate for AMI meetings.xml file
private class AMIMeetingsXMLDelegate: NSObject, XMLParserDelegate {
    let targetMeetingId: String
    var speakerMapping: AMISpeakerMapping?
    var parsingError: Error?

    private var currentMeetingId: String?
    private var speakersInCurrentMeeting: [String: String] = [:]  // agent code -> global_name
    private var isInTargetMeeting = false

    init(targetMeetingId: String) {
        self.targetMeetingId = targetMeetingId
    }

    func parser(
        _ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?,
        qualifiedName qName: String?, attributes attributeDict: [String: String] = [:]
    ) {

        if elementName == "meeting" {
            currentMeetingId = attributeDict["observation"]
            isInTargetMeeting = (currentMeetingId == targetMeetingId)
            speakersInCurrentMeeting.removeAll()
        }

        if elementName == "speaker" && isInTargetMeeting {
            guard let nxtAgent = attributeDict["nxt_agent"],
                let globalName = attributeDict["global_name"]
            else {
                return
            }
            speakersInCurrentMeeting[nxtAgent] = globalName
        }
    }

    func parser(
        _ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?,
        qualifiedName qName: String?
    ) {
        if elementName == "meeting" && isInTargetMeeting {
            // Create the speaker mapping for this meeting
            if let meetingId = currentMeetingId {
                speakerMapping = AMISpeakerMapping(
                    meetingId: meetingId,
                    speakerA: speakersInCurrentMeeting["A"] ?? "UNKNOWN",
                    speakerB: speakersInCurrentMeeting["B"] ?? "UNKNOWN",
                    speakerC: speakersInCurrentMeeting["C"] ?? "UNKNOWN",
                    speakerD: speakersInCurrentMeeting["D"] ?? "UNKNOWN"
                )
            }
            isInTargetMeeting = false
        }
    }

    func parser(_ parser: XMLParser, parseErrorOccurred parseError: Error) {
        parsingError = parseError
    }
}

#endif
