import Foundation

/// The checkpoint uses UTF-8 byte values plus one, with zero reserved for padding.
struct CuaS1FormsInput {
    static let contextByteLimit = 224
    static let optionByteLimit = 96
    static let maximumOptions = 32

    let contextIDs: [Int32]
    let optionIDs: [Int32]
    let optionMask: [Int32]
    let contextWasTruncated: Bool
    let truncatedOptionIndices: [Int]

    init(context: String, options: [String]) throws {
        guard !context.isEmpty else { throw CuaS1FormsError.emptyContext }
        guard (2...Self.maximumOptions).contains(options.count) else {
            throw CuaS1FormsError.invalidOptionCount(options.count)
        }
        var contextIDs = [Int32](repeating: 0, count: Self.contextByteLimit)
        var optionIDs = [Int32](repeating: 0, count: Self.maximumOptions * Self.optionByteLimit)
        var optionMask = [Int32](repeating: 0, count: Self.maximumOptions)
        var truncatedIndices: [Int] = []
        let contextBytes = Array(context.utf8.prefix(Self.contextByteLimit + 1))
        for (index, byte) in contextBytes.prefix(Self.contextByteLimit).enumerated() {
            contextIDs[index] = Int32(byte) + 1
        }
        for (index, option) in options.enumerated() {
            guard !option.isEmpty else { throw CuaS1FormsError.emptyOption(index) }
            let bytes = Array(option.utf8.prefix(Self.optionByteLimit + 1))
            if bytes.count > Self.optionByteLimit { truncatedIndices.append(index) }
            for (offset, byte) in bytes.prefix(Self.optionByteLimit).enumerated() {
                optionIDs[index * Self.optionByteLimit + offset] = Int32(byte) + 1
            }
            optionMask[index] = 1
        }
        self.contextIDs = contextIDs
        self.optionIDs = optionIDs
        self.optionMask = optionMask
        self.contextWasTruncated = contextBytes.count > Self.contextByteLimit
        self.truncatedOptionIndices = truncatedIndices
    }
}
