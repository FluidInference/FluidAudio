import FluidAudio
import Foundation

/// One editable piece of information supplied by the demo user.
public struct ProfileDetail: Identifiable, Sendable {
    /// Stable identity while the label and value are edited.
    public let id: UUID
    /// A descriptive name such as Email or Date of birth.
    public var name: String
    /// The user's value; empty values are excluded from inference.
    public var value: String

    /// Create an editable detail without storing it outside this session.
    public init(name: String, value: String = "") {
        self.id = UUID()
        self.name = name
        self.value = value
    }
}

/// Constructs model choices exclusively from the details entered by the user.
public enum ProfileChoices {
    /// The initial blank profile; additional labeled details can be added.
    public static func emptyDetails() -> [ProfileDetail] {
        ["First name", "Last name", "Email", "Phone", "DOB", "Address", "City", "State", "ZIP"]
            .map { ProfileDetail(name: $0) }
    }

    /// Combine explicitly supplied first and last names when no full name was supplied.
    public static func combinedName(in details: [ProfileDetail]) -> String? {
        let supplied = details.filter { !$0.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let names = Dictionary(grouping: supplied) { $0.name.trimmingCharacters(in: .whitespaces).lowercased() }
        guard ["name", "full name", "patient", "insured"].allSatisfy({ names[$0] == nil }),
            let first = names["first name"]?.first?.value,
            let last = names["last name"]?.first?.value
        else { return nil }
        return "\(first.trimmingCharacters(in: .whitespaces)) \(last.trimmingCharacters(in: .whitespaces))"
    }

    /// Validate and encode nonempty details, plus the three fixed action choices.
    public static func make(from details: [ProfileDetail]) throws -> [String] {
        var options: [String] = []
        for detail in details {
            if detail.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
            let name = detail.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { throw DemoError("Give each nonempty detail a label, such as Email.") }
            guard !name.contains(":"), name.rangeOfCharacter(from: .newlines) == nil else {
                throw DemoError("Detail labels cannot contain colons or line breaks.")
            }
            options.append("fill \(name): \(detail.value)")
        }
        guard !options.isEmpty else { throw DemoError("Enter at least one detail, or choose Use example.") }
        if let name = combinedName(in: details) { options.append("fill Name: \(name)") }
        guard options.count + 3 <= CuaS1FormsManager.maximumOptions else {
            throw DemoError("Use at most 29 nonempty details, including the combined full name.")
        }
        guard options.allSatisfy({ $0.utf8.count <= CuaS1FormsManager.optionByteLimit }) else {
            throw DemoError("A detail is too long. Keep each label and value together under 90 UTF-8 bytes.")
        }
        return options + ["check", "click", "skip"]
    }
}
