import Foundation

/// Read-only French pronunciation lexicon (`word<TAB>ipa` per line, lowercase
/// keys), searched in place instead of being expanded into a dictionary: the
/// ~245k-entry ipa-dict `fr_FR` list stays one contiguous byte buffer plus a
/// line-offset index.
struct FrenchLexicon: Sendable {
    private let bytes: [UInt8]
    /// Start offset of each line, ordered by key bytes.
    private let lines: [Int]

    init(contentsOf url: URL) throws {
        try self.init(bytes: [UInt8](Data(contentsOf: url)))
    }

    init(tsv: String) {
        self.init(bytes: Array(tsv.utf8))
    }

    private init(bytes: [UInt8]) {
        var starts: [Int] = []
        var lineStart = 0
        for index in 0...bytes.count where index == bytes.count || bytes[index] == 0x0A {
            if index > lineStart { starts.append(lineStart) }
            lineStart = index + 1
        }
        self.bytes = bytes
        // The published asset is sorted by key bytes; sort defensively so a
        // hand-edited file cannot silently break the binary search.
        var sorted = true
        for i in starts.indices.dropFirst()
        where FrenchLexicon.compareKey(bytes, at: starts[i - 1], bytes, at: starts[i]) > 0 {
            sorted = false
            break
        }
        lines = sorted ? starts : starts.sorted { FrenchLexicon.compareKey(bytes, at: $0, bytes, at: $1) < 0 }
    }

    var count: Int { lines.count }

    func contains(_ word: String) -> Bool {
        find(word) != nil
    }

    /// Pronunciation of `word` (already lowercased), or nil.
    func lookup(_ word: String) -> String? {
        guard let start = find(word) else { return nil }
        let keyEnd = start + word.utf8.count  // at the tab
        var end = keyEnd + 1
        while end < bytes.count, bytes[end] != 0x0A, bytes[end] != 0x0D { end += 1 }
        return String(decoding: bytes[(keyEnd + 1)..<end], as: UTF8.self)
    }

    private func find(_ word: String) -> Int? {
        let key = Array(word.utf8)
        guard !key.isEmpty else { return nil }
        var low = 0
        var high = lines.count - 1
        while low <= high {
            let mid = (low + high) / 2
            let start = lines[mid]
            let order = FrenchLexicon.compareKey(bytes, at: start, key, at: 0)
            if order == 0 { return start }
            if order < 0 { low = mid + 1 } else { high = mid - 1 }
        }
        return nil
    }

    /// Compare the key starting at `a[i]` (terminated by tab/newline/end)
    /// with the key starting at `b[j]`, bytewise.
    private static func compareKey(_ a: [UInt8], at i: Int, _ b: [UInt8], at j: Int) -> Int {
        var i = i
        var j = j
        while true {
            let ca: UInt8? = i < a.count && a[i] != 0x09 && a[i] != 0x0A ? a[i] : nil
            let cb: UInt8? = j < b.count && b[j] != 0x09 && b[j] != 0x0A ? b[j] : nil
            switch (ca, cb) {
            case (nil, nil): return 0
            case (nil, _): return -1
            case (_, nil): return 1
            case (let x?, let y?):
                if x != y { return x < y ? -1 : 1 }
            }
            i += 1
            j += 1
        }
    }
}
