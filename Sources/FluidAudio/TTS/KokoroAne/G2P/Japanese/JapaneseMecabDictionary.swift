import Foundation

/// Read-only view of a MeCab binary dictionary set (`sys.dic`, `unk.dic`,
/// `char.bin`, `matrix.bin`) as shipped for `unidic-lite`, trimmed by
/// `mobius/models/tts/kokoro/coreml/g2p/convert_unidic_lite.py` so that each
/// entry's feature string is `pos1,pron,kana`. The double array, token table
/// and connection matrix are the standard MeCab layouts, memory-mapped.
///
/// Byte offsets are UTF-8 byte offsets into the analyzed text, as in MeCab.
final class JapaneseMecabDictionary: Sendable {
    struct Entry {
        let byteLength: Int
        let leftID: Int
        let rightID: Int
        let cost: Int
        let featureOffset: Int
    }

    struct Feature {
        let pos1: String
        let pron: String
        let kana: String
    }

    struct CharInfo {
        /// Bit `i` set when the character belongs to category `i`.
        let typeMask: UInt32
        let defaultType: Int
        let length: Int
        let group: Bool
        let invoke: Bool
    }

    /// One MeCab dictionary file: header + double array + tokens + features.
    final class Lexicon: Sendable {
        private let data: Data
        private let dartsOffset: Int
        private let dartsUnits: Int
        private let tokensOffset: Int
        let tokenCount: Int
        private let featuresOffset: Int
        private let featuresLength: Int
        let leftSize: Int
        let rightSize: Int

        init(url: URL) throws {
            let mapped = try Data(contentsOf: url, options: .mappedIfSafe)
            try JapaneseMecabDictionary.validateLexicon(mapped, name: url.lastPathComponent)
            func word(_ index: Int) -> Int {
                Int(mapped.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self) })
            }
            leftSize = word(4)
            rightSize = word(5)
            let dsize = word(6)
            let tsize = word(7)
            let fsize = word(8)
            data = mapped
            dartsOffset = 72
            dartsUnits = dsize / 8
            tokensOffset = 72 + dsize
            tokenCount = tsize / 16
            featuresOffset = tokensOffset + tsize
            featuresLength = fsize
        }

        private func unit(_ index: Int) -> (base: Int32, check: UInt32) {
            data.withUnsafeBytes { raw in
                let offset = dartsOffset + index * 8
                return (
                    raw.loadUnaligned(fromByteOffset: offset, as: Int32.self),
                    raw.loadUnaligned(fromByteOffset: offset + 4, as: UInt32.self)
                )
            }
        }

        /// Darts common-prefix search over `key[start...]`: every dictionary
        /// key that is a prefix of the remaining bytes, as `(value, byteLength)`.
        func commonPrefixSearch(_ key: UnsafeBufferPointer<UInt8>, from start: Int) -> [(value: Int, length: Int)] {
            var results: [(value: Int, length: Int)] = []
            guard dartsUnits > 0 else { return results }
            var b = Int(unit(0).base)
            var index = start
            while index < key.count {
                if b >= 0, b < dartsUnits {
                    let terminal = unit(b)
                    if terminal.check == UInt32(b), terminal.base < 0 {
                        results.append((Int(-terminal.base - 1), index - start))
                    }
                }
                let p = b + Int(key[index]) + 1
                guard p >= 0, p < dartsUnits else { return results }
                let next = unit(p)
                guard next.check == UInt32(b) else { return results }
                b = Int(next.base)
                index += 1
            }
            if b >= 0, b < dartsUnits {
                let terminal = unit(b)
                if terminal.check == UInt32(b), terminal.base < 0 {
                    results.append((Int(-terminal.base - 1), index - start))
                }
            }
            return results
        }

        func token(_ index: Int) -> (leftID: Int, rightID: Int, cost: Int, featureOffset: Int) {
            data.withUnsafeBytes { raw in
                let offset = tokensOffset + index * 16
                return (
                    Int(raw.loadUnaligned(fromByteOffset: offset, as: UInt16.self)),
                    Int(raw.loadUnaligned(fromByteOffset: offset + 2, as: UInt16.self)),
                    Int(raw.loadUnaligned(fromByteOffset: offset + 6, as: Int16.self)),
                    Int(raw.loadUnaligned(fromByteOffset: offset + 8, as: UInt32.self))
                )
            }
        }

        /// All entries whose surface is a prefix of `key[start...]`.
        func entries(_ key: UnsafeBufferPointer<UInt8>, from start: Int) -> [Entry] {
            var entries: [Entry] = []
            for (value, length) in commonPrefixSearch(key, from: start) {
                let first = value >> 8
                let count = value & 0xFF
                for k in 0..<count where first + k < tokenCount {
                    let t = token(first + k)
                    entries.append(
                        Entry(
                            byteLength: length, leftID: t.leftID, rightID: t.rightID, cost: t.cost,
                            featureOffset: t.featureOffset))
                }
            }
            return entries
        }

        /// Entries stored under the literal key `name` (unknown-word categories in `unk.dic`).
        func entries(forExactKey name: String) -> [Entry] {
            var bytes = Array(name.utf8)
            return bytes.withUnsafeMutableBufferPointer { buffer in
                entries(UnsafeBufferPointer(buffer), from: 0).filter { $0.byteLength == buffer.count }
            }
        }

        func feature(at offset: Int) -> Feature {
            guard offset < featuresLength else { return Feature(pos1: "", pron: "", kana: "") }
            let start = featuresOffset + offset
            var end = start
            while end < featuresOffset + featuresLength, data[end] != 0 { end += 1 }
            let text = String(decoding: data[start..<end], as: UTF8.self)
            let parts = text.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            func field(_ i: Int) -> String { i < parts.count && parts[i] != "*" ? parts[i] : "" }
            return Feature(pos1: field(0), pron: field(1), kana: field(2))
        }
    }

    let system: Lexicon
    let unknown: Lexicon
    private let charData: Data
    private let categoryNames: [String]
    private let charMapOffset: Int
    private let matrix: Data
    let leftSize: Int
    let rightSize: Int

    // MARK: - Validation

    /// Structural checks on the binary assets, run before any header field is
    /// read. Each file is exactly the size its header implies (the trimmed
    /// dictionaries are written that way), every section is nonempty, and the
    /// format version is MeCab's 102 — so a truncated, empty, or foreign file
    /// fails, and the downloader can refetch only the files that do. Cheap:
    /// mmapped, header bytes only.
    static let mecabDictionaryVersion = 102

    static func validateLexicon(_ mapped: Data, name: String) throws {
        guard mapped.count >= 72 else {
            throw KokoroAneError.modelNotLoaded("MeCab dictionary \(name) is truncated")
        }
        func word(_ index: Int) -> Int {
            Int(mapped.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self) })
        }
        guard word(1) == mecabDictionaryVersion else {
            throw KokoroAneError.modelNotLoaded("MeCab dictionary \(name) has version \(word(1)), expected 102")
        }
        let lexsize = word(3)
        let lsize = word(4)
        let rsize = word(5)
        let dsize = word(6)
        let tsize = word(7)
        let fsize = word(8)
        guard lexsize > 0, lsize > 0, rsize > 0, dsize > 0, tsize > 0, fsize > 0,
            dsize % 8 == 0, tsize % 16 == 0
        else {
            throw KokoroAneError.modelNotLoaded("MeCab dictionary \(name) has an empty section")
        }
        guard 72 + dsize + tsize + fsize == mapped.count else {
            throw KokoroAneError.modelNotLoaded(
                "MeCab dictionary \(name) is \(mapped.count) bytes, header implies \(72 + dsize + tsize + fsize)")
        }
    }

    static func validateCharCategories(_ chars: Data) throws {
        guard chars.count >= 4 else { throw KokoroAneError.modelNotLoaded("MeCab char.bin is truncated") }
        let count = Int(chars.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) })
        // MeCab's table has 0xFFFF entries (code points 0…0xFFFE).
        guard count > 0, chars.count == 4 + count * 32 + 0xFFFF * 4 else {
            throw KokoroAneError.modelNotLoaded("MeCab char.bin is \(chars.count) bytes with \(count) categories")
        }
    }

    static func validateConnectionMatrix(_ costs: Data) throws {
        guard costs.count >= 4 else { throw KokoroAneError.modelNotLoaded("MeCab matrix.bin is truncated") }
        let leftSize = Int(costs.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt16.self) })
        let rightSize = Int(costs.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 2, as: UInt16.self) })
        guard leftSize > 0, rightSize > 0, costs.count == 4 + leftSize * rightSize * 2 else {
            throw KokoroAneError.modelNotLoaded(
                "MeCab matrix.bin is \(costs.count) bytes for \(leftSize)×\(rightSize) contexts")
        }
    }

    /// Validates one asset file by name; a missing file throws too.
    static func validateAsset(named name: String, at url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        switch name {
        case KokoroAneConstants.japaneseSystemDictionaryFile, KokoroAneConstants.japaneseUnknownDictionaryFile:
            try validateLexicon(data, name: name)
        case KokoroAneConstants.japaneseCharCategoryFile:
            try validateCharCategories(data)
        case KokoroAneConstants.japaneseConnectionMatrixFile:
            try validateConnectionMatrix(data)
        default:
            guard !data.isEmpty else { throw KokoroAneError.modelNotLoaded("Japanese G2P asset \(name) is empty") }
        }
    }

    init(directory: URL) throws {
        system = try Lexicon(url: directory.appendingPathComponent(KokoroAneConstants.japaneseSystemDictionaryFile))
        unknown = try Lexicon(url: directory.appendingPathComponent(KokoroAneConstants.japaneseUnknownDictionaryFile))
        let chars = try Data(
            contentsOf: directory.appendingPathComponent(KokoroAneConstants.japaneseCharCategoryFile),
            options: .mappedIfSafe)
        try Self.validateCharCategories(chars)
        let count = Int(chars.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) })
        categoryNames = (0..<count).map { i in
            let start = 4 + i * 32
            let slice = chars[start..<start + 32].prefix { $0 != 0 }
            return String(decoding: slice, as: UTF8.self)
        }
        charMapOffset = 4 + count * 32
        charData = chars
        let costs = try Data(
            contentsOf: directory.appendingPathComponent(KokoroAneConstants.japaneseConnectionMatrixFile),
            options: .mappedIfSafe)
        try Self.validateConnectionMatrix(costs)
        leftSize = Int(costs.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt16.self) })
        rightSize = Int(costs.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 2, as: UInt16.self) })
        // The lexicons and the matrix must come from the same build.
        for lexicon in [system, unknown]
        where lexicon.leftSize != leftSize || lexicon.rightSize != rightSize {
            throw KokoroAneError.modelNotLoaded(
                "MeCab dictionary contexts \(lexicon.leftSize)×\(lexicon.rightSize) "
                    + "do not match matrix.bin \(leftSize)×\(rightSize)")
        }
        matrix = costs
    }

    func categoryName(_ id: Int) -> String {
        id < categoryNames.count ? categoryNames[id] : "DEFAULT"
    }

    func charInfo(_ scalar: Unicode.Scalar) -> CharInfo {
        // MeCab's table covers the BMP; anything above maps to DEFAULT (code point 0).
        let code = scalar.value < 0xFFFF ? Int(scalar.value) : 0
        let v = charData.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: charMapOffset + code * 4, as: UInt32.self) }
        return CharInfo(
            typeMask: v & 0x3FFFF, defaultType: Int((v >> 18) & 0xFF), length: Int((v >> 26) & 0xF),
            group: (v >> 30) & 1 == 1, invoke: (v >> 31) & 1 == 1)
    }

    /// `matrix[rightID of previous][leftID of next]`, MeCab's `matrix_[rcAttr + lsize * lcAttr]`.
    func connectionCost(previousRightID: Int, nextLeftID: Int) -> Int {
        guard previousRightID < leftSize, nextLeftID < rightSize else { return 0 }
        let index = 4 + (nextLeftID * leftSize + previousRightID) * 2
        return Int(matrix.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: index, as: Int16.self) })
    }
}
