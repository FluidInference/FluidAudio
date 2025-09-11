#if os(macOS)
import FluidAudio
import Foundation

/// AMI annotation parser and ground truth handling
struct AMIParser {
    private static let logger = AppLogger(category: "AMIParser")

    /// Get ground truth speaker count from AMI meetings.xml
    static func getGroundTruthSpeakerCount(for meetingId: String) -> Int {
        // Use the same path resolution logic as loadAMIGroundTruth for consistency
        let possiblePaths = [
            // Current working directory - NEW Datasets location (after PR #19)
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(
                    "Datasets/ami_public_1.6.2"),
            // Relative to source file - NEW Datasets location
            URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent(
                    "Datasets/ami_public_1.6.2"),
            // OLD: Current working directory - Tests location (backward compatibility)
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(
                    "Tests/ami_public_1.6.2"),
            // OLD: Relative to source file - Tests location (backward compatibility)
            URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Tests/ami_public_1.6.2"),
            // OLD: Home directory - Tests location (backward compatibility)
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
                "code/FluidAudio/Tests/ami_public_1.6.2"),
        ]

        for location in possiblePaths {
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

    /// Load AMI ground truth annotations for a specific meeting
    static func loadAMIGroundTruth(
        for meetingId: String, duration: Float
    ) async
        -> [TimedSpeakerSegment]
    {
        // Try to find the AMI annotations directory in several possible locations
        let possiblePaths = [
            // Current working directory - NEW Datasets location (after PR #19)
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(
                    "Datasets/ami_public_1.6.2"),
            // Relative to source file - NEW Datasets location
            URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent(
                    "Datasets/ami_public_1.6.2"),
            // OLD: Current working directory - Tests location (backward compatibility)
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(
                    "Tests/ami_public_1.6.2"),
            // OLD: Relative to source file - Tests location (backward compatibility)
            URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Tests/ami_public_1.6.2"),
            // OLD: Home directory - Tests location (backward compatibility)
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
                "code/FluidAudio/Tests/ami_public_1.6.2"),
        ]

        // Add comprehensive debug logging for path resolution
        var amiDir: URL?
        for (_, path) in possiblePaths.enumerated() {
            let segmentsDir = path.appendingPathComponent("segments")
            let meetingsFile = path.appendingPathComponent("corpusResources/meetings.xml")

            let segmentsExists = FileManager.default.fileExists(atPath: segmentsDir.path)
            let meetingsExists = FileManager.default.fileExists(atPath: meetingsFile.path)

            if segmentsExists && meetingsExists {
                logger.info("      - 🎯 SELECTED: This path will be used")
                amiDir = path
                break
            }
        }

        guard let validAmiDir = amiDir else {
            logger.warning("   AMI annotations not found in any expected location")
            logger.warning(
                "      📁 Expected structure: [path]/segments/ AND [path]/corpusResources/meetings.xml"
            )
            logger.warning(
                "      🔧 To download annotations: visit https://groups.inf.ed.ac.uk/ami/download/"
            )
            logger.warning(
                "      📋 Using simplified placeholder ground truth (causes poor DER performance)"
            )
            return generateSimplifiedGroundTruth(duration: duration, speakerCount: 4)
        }

        let segmentsDir = validAmiDir.appendingPathComponent("segments")
        let meetingsFile = validAmiDir.appendingPathComponent("corpusResources/meetings.xml")

        logger.info("   📖 Loading AMI annotations for meeting: \(meetingId)")

        do {
            let parser = AMIAnnotationParser()

            // Get speaker mapping for this meeting
            guard
                let speakerMapping = try parser.parseSpeakerMapping(
                    for: meetingId, from: meetingsFile)
            else {
                logger.warning(
                    "      ⚠️ No speaker mapping found for meeting: \(meetingId), using placeholder"
                )
                return generateSimplifiedGroundTruth(duration: duration, speakerCount: 4)
            }

            logger.info(
                "      Speaker mapping: A=\(speakerMapping.speakerA), B=\(speakerMapping.speakerB), C=\(speakerMapping.speakerC), D=\(speakerMapping.speakerD)"
            )

            var allSegments: [TimedSpeakerSegment] = []

            // Parse segments for each speaker (A, B, C, D)
            for speakerCode in ["A", "B", "C", "D"] {
                let segmentFile = segmentsDir.appendingPathComponent(
                    "\(meetingId).\(speakerCode).segments.xml")

                if FileManager.default.fileExists(atPath: segmentFile.path) {
                    let segments = try parser.parseSegmentsFile(segmentFile)

                    // Map to TimedSpeakerSegment with real participant ID
                    guard let participantId = speakerMapping.participantId(for: speakerCode)
                    else {
                        continue
                    }

                    for segment in segments {
                        // Filter out very short segments (< 0.5 seconds) as done in research
                        guard segment.duration >= 0.5 else { continue }

                        let timedSegment = TimedSpeakerSegment(
                            speakerId: participantId,  // Use real AMI participant ID
                            embedding: generatePlaceholderEmbedding(for: participantId),
                            startTimeSeconds: Float(segment.startTime),
                            endTimeSeconds: Float(segment.endTime),
                            qualityScore: 1.0
                        )

                        allSegments.append(timedSegment)
                    }

                    logger.info(
                        "      Loaded \(segments.count) segments for speaker \(speakerCode) (\(participantId))"
                    )
                }
            }

            // Sort by start time
            allSegments.sort { $0.startTimeSeconds < $1.startTimeSeconds }

            logger.info("      Total segments loaded: \(allSegments.count)")
            return allSegments

        } catch {
            logger.warning("      Failed to parse AMI annotations: \(error)")
            logger.warning("      Using simplified placeholder instead")
            return generateSimplifiedGroundTruth(duration: duration, speakerCount: 4)
        }
    }

    /// Generate simplified ground truth for testing
    static func generateSimplifiedGroundTruth(
        duration: Float, speakerCount: Int
    )
        -> [TimedSpeakerSegment]
    {
        let segmentDuration = duration / Float(speakerCount * 2)
        var segments: [TimedSpeakerSegment] = []
        let dummyEmbedding: [Float] = Array(repeating: 0.1, count: 512)

        for i in 0..<(speakerCount * 2) {
            let speakerId = "Speaker \((i % speakerCount) + 1)"
            let startTime = Float(i) * segmentDuration
            let endTime = min(startTime + segmentDuration, duration)

            segments.append(
                TimedSpeakerSegment(
                    speakerId: speakerId,
                    embedding: dummyEmbedding,
                    startTimeSeconds: startTime,
                    endTimeSeconds: endTime,
                    qualityScore: 1.0
                ))
        }

        return segments
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
