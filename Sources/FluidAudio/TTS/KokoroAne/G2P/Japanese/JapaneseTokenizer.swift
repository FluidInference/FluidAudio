import Foundation

/// A MeCab-compatible Viterbi tokenizer over ``JapaneseMecabDictionary``.
///
/// Produces the same segmentation and readings as fugashi/MeCab with
/// `unidic-lite` (verified against fugashi on the reference sentences), which
/// is what Misaki's Cutlet frontend, and therefore Kokoro's Japanese training
/// data, was built on. Unknown words follow MeCab's character-category rules
/// (`char.bin` + `unk.dic`): categories with `group` set yield one candidate
/// spanning the run, categories with `length` yield candidates of 1…length
/// characters, and unknown processing runs when no dictionary entry starts at
/// the position or the category has `invoke` set.
struct JapaneseTokenizer {
    struct Word: Equatable {
        let surface: String
        /// Pronunciation kana (`pron`), empty for unknown words and symbols.
        let pron: String
        /// Orthographic reading kana (`kana`), Cutlet's fallback when `pron` is empty.
        let kana: String
        /// MeCab character category id of the surface (2 kanji, 3 symbol, 6 hiragana, 7 katakana, …).
        let charType: Int
        let isUnknown: Bool
    }

    private let dictionary: JapaneseMecabDictionary

    init(dictionary: JapaneseMecabDictionary) {
        self.dictionary = dictionary
    }

    private struct Node {
        let start: Int
        let end: Int
        let cost: Int
        let rightID: Int
        let featureOffset: Int
        let isUnknown: Bool
        let charType: Int
        let previous: Int  // index into `nodes`, -1 for BOS
    }

    private struct Candidate {
        let byteLength: Int
        let leftID: Int
        let rightID: Int
        let cost: Int
        let featureOffset: Int
        let isUnknown: Bool
    }

    func tokenize(_ text: String) -> [Word] {
        let bytes = Array(text.utf8)
        guard !bytes.isEmpty else { return [] }
        // Code point boundaries: (byte offset, byte length, scalar).
        var scalars: [(offset: Int, length: Int, scalar: Unicode.Scalar)] = []
        var offset = 0
        for scalar in text.unicodeScalars {
            let length = String(scalar).utf8.count
            scalars.append((offset, length, scalar))
            offset += length
        }
        var scalarIndex: [Int: Int] = [:]
        for (i, s) in scalars.enumerated() { scalarIndex[s.offset] = i }

        var nodes: [Node] = [
            Node(start: 0, end: 0, cost: 0, rightID: 0, featureOffset: -1, isUnknown: false, charType: 0, previous: -1)
        ]
        var endingAt: [Int: [Int]] = [0: [0]]

        bytes.withUnsafeBufferPointer { buffer in
            for (k, s) in scalars.enumerated() {
                guard let previousNodes = endingAt[s.offset], !previousNodes.isEmpty else { continue }
                // MeCab skips whitespace: no node is created and the nodes
                // ending before the space connect directly to the next word.
                if s.scalar.properties.isWhitespace {
                    endingAt[s.offset + s.length, default: []].append(contentsOf: previousNodes)
                    continue
                }
                var candidates: [Candidate] = dictionary.system.entries(buffer, from: s.offset).map {
                    Candidate(
                        byteLength: $0.byteLength, leftID: $0.leftID, rightID: $0.rightID, cost: $0.cost,
                        featureOffset: $0.featureOffset, isUnknown: false)
                }
                let info = dictionary.charInfo(s.scalar)
                if candidates.isEmpty || info.invoke {
                    let category = info.defaultType
                    let categoryBit: UInt32 = 1 << UInt32(category)
                    var lengths = Set<Int>()
                    if info.group {
                        var j = k
                        var total = 0
                        while j < scalars.count, dictionary.charInfo(scalars[j].scalar).typeMask & categoryBit != 0 {
                            total += scalars[j].length
                            j += 1
                        }
                        if total > 0 { lengths.insert(total) }
                    }
                    if info.length > 0 {
                        var total = 0
                        for m in 0..<info.length {
                            guard k + m < scalars.count,
                                dictionary.charInfo(scalars[k + m].scalar).typeMask & categoryBit != 0
                            else { break }
                            total += scalars[k + m].length
                            lengths.insert(total)
                        }
                    }
                    if lengths.isEmpty { lengths.insert(s.length) }
                    let unknownEntries = dictionary.unknown.entries(forExactKey: dictionary.categoryName(category))
                    for length in lengths {
                        for entry in unknownEntries {
                            candidates.append(
                                Candidate(
                                    byteLength: length, leftID: entry.leftID, rightID: entry.rightID, cost: entry.cost,
                                    featureOffset: entry.featureOffset, isUnknown: true))
                        }
                    }
                }
                for candidate in candidates {
                    var bestCost = Int.max
                    var bestPrevious = -1
                    for p in previousNodes {
                        let node = nodes[p]
                        let total =
                            node.cost + candidate.cost
                            + dictionary.connectionCost(previousRightID: node.rightID, nextLeftID: candidate.leftID)
                        if total < bestCost {
                            bestCost = total
                            bestPrevious = p
                        }
                    }
                    guard bestPrevious >= 0 else { continue }
                    let end = s.offset + candidate.byteLength
                    nodes.append(
                        Node(
                            start: s.offset, end: end, cost: bestCost, rightID: candidate.rightID,
                            featureOffset: candidate.featureOffset, isUnknown: candidate.isUnknown,
                            charType: info.defaultType, previous: bestPrevious))
                    endingAt[end, default: []].append(nodes.count - 1)
                }
            }
        }

        let finals = (endingAt[bytes.count] ?? []).filter { $0 != 0 }
        guard !finals.isEmpty else { return [] }
        var best = finals[0]
        var bestCost = Int.max
        for index in finals {
            let node = nodes[index]
            let total = node.cost + dictionary.connectionCost(previousRightID: node.rightID, nextLeftID: 0)
            if total < bestCost {
                bestCost = total
                best = index
            }
        }
        var path: [Node] = []
        var cursor = best
        while cursor > 0 {
            path.append(nodes[cursor])
            cursor = nodes[cursor].previous
        }
        path.reverse()
        return path.map { node in
            let lexicon = node.isUnknown ? dictionary.unknown : dictionary.system
            let feature = lexicon.feature(at: node.featureOffset)
            let surface = String(decoding: bytes[node.start..<node.end], as: UTF8.self)
            return Word(
                surface: surface, pron: node.isUnknown ? "" : feature.pron, kana: node.isUnknown ? "" : feature.kana,
                charType: node.charType, isUnknown: node.isUnknown)
        }
    }
}
