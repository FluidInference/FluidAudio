import Foundation
import XCTest

/// Swift 6.4 `-O` inverts an unapplied `CharacterSet.<set>.contains` method reference, such as
/// `allSatisfy(CharacterSet.whitespacesAndNewlines.contains)`, once a module holds more than one of
/// them. Release builds then trimmed the first and last word of every phoneme chunk. Debug test runs
/// can't see a release miscompile, so this keeps the pattern out of `Sources/`: pass a closure
/// (`{ set.contains($0) }`) instead.
final class CharacterSetMethodReferenceTests: XCTestCase {
    func testSourcesPassNoCharacterSetContainsAsAFunction() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Shared
            .deletingLastPathComponent()  // FluidAudioTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
            .appendingPathComponent("Sources")
        let reference = try NSRegularExpression(pattern: #"CharacterSet\.\w+(\.inverted)?\.contains\s*[),\]]"#)
        var scanned = 0
        var offenders: [String] = []
        let files = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)
        while let file = files?.nextObject() as? URL {
            guard file.pathExtension == "swift" else { continue }
            scanned += 1
            let text = try String(contentsOf: file, encoding: .utf8)
            for (index, line) in text.components(separatedBy: "\n").enumerated() {
                let code = line.components(separatedBy: "//")[0]
                if reference.firstMatch(in: code, range: NSRange(code.startIndex..., in: code)) != nil {
                    offenders.append(
                        "\(file.lastPathComponent):\(index + 1): \(code.trimmingCharacters(in: .whitespaces))")
                }
            }
        }
        XCTAssertGreaterThan(scanned, 100, "found no sources under \(sources.path)")
        XCTAssertTrue(
            offenders.isEmpty,
            "Pass a closure, not the method reference:\n" + offenders.joined(separator: "\n"))
    }
}
