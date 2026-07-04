#if os(macOS)
import Foundation
import XCTest

@testable import FluidAudioCLI

final class AMIParserTests: XCTestCase {

    func testWordAlignedGroundTruthParsesAndMergesWords() throws {
        let fixture = try makeAMIFixture()
        let segments = try AMIParser.loadWordAlignedGroundTruth(
            for: "ES2004a",
            in: fixture,
            duration: 30
        )

        XCTAssertEqual(segments.count, 2)

        XCTAssertEqual(segments[0].speakerId, "SpeakerA")
        XCTAssertEqual(segments[0].startTimeSeconds, 0.1, accuracy: 0.0001)
        XCTAssertEqual(segments[0].endTimeSeconds, 0.7, accuracy: 0.0001)

        XCTAssertEqual(segments[1].speakerId, "SpeakerB")
        XCTAssertEqual(segments[1].startTimeSeconds, 1.5, accuracy: 0.0001)
        XCTAssertEqual(segments[1].endTimeSeconds, 1.9, accuracy: 0.0001)
    }

    func testWordAlignedDERReferenceUsesMappedParticipantIDs() async throws {
        let fixture = try makeAMIFixture()
        let segments = try AMIParser.loadWordAlignedDERReference(
            for: "ES2004a",
            in: fixture,
            duration: 30
        )
        XCTAssertEqual(segments.map(\.speaker), ["SpeakerA", "SpeakerB"])
    }

    func testLegacyOfficialGroundTruthStillFiltersShortSegments() throws {
        let fixture = try makeAMIFixture()
        let segments = try AMIParser.loadAMIGroundTruth(
            for: "ES2004a",
            in: fixture,
            duration: 30
        )

        XCTAssertTrue(segments.isEmpty)
    }

    func testFrameAlignedDERReferenceQuantizesToKaldiStyle10msFrames() throws {
        let fixture = try makeAMIFixture()
        let segments = try AMIParser.loadFrameAlignedDERReference(
            for: "ES2004a",
            in: fixture,
            duration: 30
        )

        XCTAssertEqual(segments.count, 3)

        XCTAssertEqual(segments[0].speaker, "SpeakerA")
        XCTAssertEqual(segments[0].start, 0.00, accuracy: 0.0001)
        XCTAssertEqual(segments[0].end, 0.25, accuracy: 0.0001)

        XCTAssertEqual(segments[1].speaker, "SpeakerA")
        XCTAssertEqual(segments[1].start, 0.60, accuracy: 0.0001)
        XCTAssertEqual(segments[1].end, 0.80, accuracy: 0.0001)

        XCTAssertEqual(segments[2].speaker, "SpeakerB")
        XCTAssertEqual(segments[2].start, 1.00, accuracy: 0.0001)
        XCTAssertEqual(segments[2].end, 1.02, accuracy: 0.0001)
    }

    func testLoadAMIGroundTruthThrowsWhenAnnotationsMissing() throws {
        let missingRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        XCTAssertThrowsError(
            try AMIParser.loadAMIGroundTruth(
                for: "ES2004a",
                duration: 30,
                searchRoots: [missingRoot]
            )
        ) { error in
            guard case AMIParserError.annotationsNotFound(let subdirectory) = error else {
                return XCTFail("Expected annotationsNotFound, got \(error)")
            }
            XCTAssertEqual(subdirectory, "segments")
        }
    }

    func testLoadWordAlignedGroundTruthThrowsWhenAnnotationsMissing() throws {
        let missingRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        XCTAssertThrowsError(
            try AMIParser.loadWordAlignedGroundTruth(
                for: "ES2004a",
                duration: 30,
                searchRoots: [missingRoot]
            )
        ) { error in
            guard case AMIParserError.annotationsNotFound(let subdirectory) = error else {
                return XCTFail("Expected annotationsNotFound, got \(error)")
            }
            XCTAssertEqual(subdirectory, "words")
        }
    }

    func testSearchBasedLoadersResolveFixtureRoot() throws {
        let fixture = try makeAMIFixture()

        let segments = try AMIParser.loadWordAlignedGroundTruth(
            for: "ES2004a",
            duration: 30,
            searchRoots: [fixture]
        )
        XCTAssertEqual(segments.count, 2)
    }

    private func makeAMIFixture() throws -> URL {
        let baseURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let wordsURL = baseURL.appendingPathComponent("words", isDirectory: true)
        let segmentsURL = baseURL.appendingPathComponent("segments", isDirectory: true)
        let corpusURL = baseURL.appendingPathComponent("corpusResources", isDirectory: true)

        try FileManager.default.createDirectory(at: wordsURL, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: segmentsURL, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: corpusURL, withIntermediateDirectories: true)

        let meetingsXML = """
            <meetings>
              <meeting observation="ES2004a">
                <speaker nxt_agent="A" global_name="SpeakerA"/>
                <speaker nxt_agent="B" global_name="SpeakerB"/>
                <speaker nxt_agent="C" global_name="SpeakerC"/>
                <speaker nxt_agent="D" global_name="SpeakerD"/>
              </meeting>
            </meetings>
            """
        try meetingsXML.write(
            to: corpusURL.appendingPathComponent("meetings.xml"),
            atomically: true,
            encoding: .utf8
        )

        let speakerAWords = """
            <nite:root xmlns:nite="http://nite.sourceforge.net/">
              <w nite:id="a1" starttime="0.10" endtime="0.40">hello</w>
              <w nite:id="a2" starttime="0.45" endtime="0.70">world</w>
              <w nite:id="a3" starttime="0.71" endtime="1.00" punc="true">.</w>
              <pause starttime="1.00" endtime="1.20"/>
            </nite:root>
            """
        try speakerAWords.write(
            to: wordsURL.appendingPathComponent("ES2004a.A.words.xml"),
            atomically: true,
            encoding: .utf8
        )

        let speakerBWords = """
            <nite:root xmlns:nite="http://nite.sourceforge.net/">
              <w nite:id="b1" starttime="1.50" endtime="1.70">second</w>
              <w nite:id="b2" starttime="1.71" endtime="1.90">speaker</w>
            </nite:root>
            """
        try speakerBWords.write(
            to: wordsURL.appendingPathComponent("ES2004a.B.words.xml"),
            atomically: true,
            encoding: .utf8
        )

        let emptyWords = "<nite:root xmlns:nite=\"http://nite.sourceforge.net/\"/>"
        try emptyWords.write(
            to: wordsURL.appendingPathComponent("ES2004a.C.words.xml"),
            atomically: true,
            encoding: .utf8
        )
        try emptyWords.write(
            to: wordsURL.appendingPathComponent("ES2004a.D.words.xml"),
            atomically: true,
            encoding: .utf8
        )

        let speakerASegments = """
            <nite:root xmlns:nite="http://nite.sourceforge.net/">
              <segment nite:id="a1" transcriber_start="0.004" transcriber_end="0.126"/>
              <segment nite:id="a2" transcriber_start="0.129" transcriber_end="0.254"/>
              <segment nite:id="a3" transcriber_start="0.601" transcriber_end="0.799"/>
            </nite:root>
            """
        try speakerASegments.write(
            to: segmentsURL.appendingPathComponent("ES2004a.A.segments.xml"),
            atomically: true,
            encoding: .utf8
        )

        let speakerBSegments = """
            <nite:root xmlns:nite="http://nite.sourceforge.net/">
              <segment nite:id="b1" transcriber_start="1.001" transcriber_end="1.019"/>
            </nite:root>
            """
        try speakerBSegments.write(
            to: segmentsURL.appendingPathComponent("ES2004a.B.segments.xml"),
            atomically: true,
            encoding: .utf8
        )

        let emptySegments = "<nite:root xmlns:nite=\"http://nite.sourceforge.net/\"/>"
        try emptySegments.write(
            to: segmentsURL.appendingPathComponent("ES2004a.C.segments.xml"),
            atomically: true,
            encoding: .utf8
        )
        try emptySegments.write(
            to: segmentsURL.appendingPathComponent("ES2004a.D.segments.xml"),
            atomically: true,
            encoding: .utf8
        )

        return baseURL
    }
}
#endif
