import CryptoKit
import Foundation

/// An invalid fixture, asset, or demo transition.
public struct DemoError: LocalizedError, Sendable {
    /// Human-readable failure reason.
    public let message: String
    /// Describe a recoverable demo error.
    public init(_ message: String) { self.message = message }
    /// Description displayed by the example application.
    public var errorDescription: String? { message }
}

/// One pre-extracted entity from the upstream document's candidate list.
public struct DocumentEntity: Identifiable, Sendable {
    /// The complete candidate string, also used as a stable identity.
    public let id: String
    /// Field name supplied by the document extractor.
    public let name: String
    /// Value supplied by the document extractor.
    public let value: String
}

/// A form element and its current local UI state.
public struct DemoControl: Identifiable, Sendable {
    /// Original row number in the pinned upstream demo.
    public let id: Int
    /// Upstream accessibility role.
    public let role: String
    /// Visible control label.
    public let title: String
    /// Editable field contents.
    public var value = ""
    /// Checkbox state.
    public var isChecked = false
    /// Original task and form description, without the element line.
    public let prefix: String

    /// Describe the current control to the decision model.
    public var context: String {
        let state = role == "CheckBox" ? (isChecked ? "checked" : "unchecked") : "value=\"\(value)\""
        return "\(prefix)\nELEMENT \(role) \"\(title)\" \(state)"
    }
}

/// A form and the document options supplied with it; contains no answer labels.
public struct DemoScenario: Identifiable, Sendable {
    /// Upstream page identifier.
    public let id: String
    /// Full form title from the original context.
    public let title: String
    /// Compact title for the scenario picker.
    public let shortTitle: String
    /// Name of the upstream source-document type.
    public let documentTitle: String
    /// All supplied candidates, including distractors and fixed actions.
    public let options: [String]
    /// Extracted document entities, without choosing targets for them.
    public let entities: [DocumentEntity]
    /// Initial form controls in upstream order.
    public let controls: [DemoControl]
}

/// Reads the unmodified, hash-pinned upstream demo. UI construction never uses answer labels.
public enum DemoCatalog {
    /// SHA-256 of the bundled original JSONL bytes, including CRLF line endings.
    public static let datasetSHA256 = "4f43b442e79ba2e2ce731e27e9b8e340c2b5dfcaffc92d8ff564c34f115ff1ca"

    private struct Row: Decodable {
        struct Metadata: Decodable { let page: String }
        let context: String
        let options: [String]
        let meta: Metadata
    }

    /// Load and validate the original demo bytes.
    public static func fixtureData() throws -> Data {
        // SwiftPM places bundles beside a CLI executable; a signed macOS app keeps
        // them in Contents/Resources. Prefer the app location when packaged.
        let packaged = Bundle.main.url(forResource: "CuaS1FormsDemo_CuaDemoCore", withExtension: "bundle")
            .flatMap { Bundle(url: $0) }
        let resources = packaged ?? Bundle.module
        guard let url = resources.url(forResource: "demo", withExtension: "jsonl") else {
            throw DemoError("The bundled upstream demo is missing.")
        }
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == datasetSHA256 else { throw DemoError("The upstream demo checksum does not match.") }
        return data
    }

    /// Build the three original forms from their first, empty UI states.
    public static func load() throws -> [DemoScenario] {
        let rows = try fixtureData().split(separator: 10).map {
            try JSONDecoder().decode(Row.self, from: Data($0))
        }
        let definitions = [
            ("patient-registration", "Patient registration", "Referral letter"),
            ("job-application", "Job application", "Applicant profile"),
            ("auto-claim", "Auto insurance claim", "Incident report"),
        ]
        return try definitions.map { page, shortTitle, documentTitle in
            let matching = rows.enumerated().filter { $0.element.meta.page == page }
            guard let first = matching.first else { throw DemoError("Missing form: \(page)") }
            let lines = first.element.context.components(separatedBy: "\n")
            guard lines.count == 3, lines[1].hasPrefix("FORM ") else { throw DemoError("Invalid form context.") }
            let prefix = lines.prefix(2).joined(separator: "\n")
            var controls: [DemoControl] = []
            for (index, row) in matching {
                // The initial form ends where the upstream browser chrome starts.
                if row.context.contains("ELEMENT Button \"Back\"") { break }
                let element = row.context.components(separatedBy: "\nELEMENT ")
                guard element.count == 2, let quote = element[1].firstIndex(of: "\"") else {
                    throw DemoError("Invalid element in row \(index).")
                }
                let role = element[1][..<quote].trimmingCharacters(in: .whitespaces)
                let suffix = element[1][element[1].index(after: quote)...]
                guard let end = suffix.firstIndex(of: "\""), ["Edit", "CheckBox", "Button"].contains(role) else {
                    throw DemoError("Unsupported element in row \(index).")
                }
                let control = DemoControl(id: index, role: role, title: String(suffix[..<end]), prefix: prefix)
                guard control.context == row.context else { throw DemoError("Initial state changed in row \(index).") }
                controls.append(control)
            }
            let options = first.element.options
            let entities = options.compactMap { option -> DocumentEntity? in
                guard let parts = fillParts(option) else { return nil }
                return DocumentEntity(id: option, name: parts.name, value: parts.value)
            }
            return DemoScenario(
                id: page, title: String(lines[1].dropFirst(5)), shortTitle: shortTitle,
                documentTitle: documentTitle, options: options, entities: entities, controls: controls)
        }
    }

    /// Split a supplied fill action at its first colon; preserve punctuation in the value.
    public static func fillParts(_ option: String) -> (name: String, value: String)? {
        guard option.hasPrefix("fill "), let separator = option.range(of: ": ") else { return nil }
        return (
            String(option[option.index(option.startIndex, offsetBy: 5)..<separator.lowerBound]),
            String(option[separator.upperBound...])
        )
    }
}
