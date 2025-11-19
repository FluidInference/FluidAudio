import Foundation

/// Trie-based engine for custom vocabulary context biasing.
///
/// This engine operates at the word level (using normalized word strings derived from
/// the model vocabulary tokens). It builds a prefix trie over phrases in
/// `CustomVocabularyContext`, assigning per-node bias scores based on:
/// - `contextScore`
/// - `depthScaling`
/// - `scorePerPhrase`
/// - per-phrase `weight`
struct CustomVocabularyEngine {

    /// A single trie node keyed by next-word strings.
    struct Node {
        var children: [String: Int]
        var bias: Float
    }

    private(set) var nodes: [Node] = []

    init(context: CustomVocabularyContext) {
        // Root node
        nodes.append(Node(children: [:], bias: 0))

        let contextScore = context.contextScore
        let depthScaling = context.depthScaling
        let baseScorePerPhrase = context.scorePerPhrase

        for term in context.terms {
            let text = term.text
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard !text.isEmpty else { continue }

            let words = text.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
            guard !words.isEmpty else { continue }

            let phraseLength = Float(words.count)

            // Base contribution per token from contextScore and per-phrase score.
            // Per-term weights from custom_vocab.json are intentionally ignored here so that
            // all phrases share the same base strength; tuning is done via global knobs
            // (alpha, contextScore, depthScaling, scorePerPhrase).
            let baseTokenScore = contextScore
            let perPhraseShare: Float = phraseLength > 0 ? baseScorePerPhrase / phraseLength : 0

            var nodeIndex = 0
            for (depth, word) in words.enumerated() {
                let key = word
                let nextIndex: Int
                if let existing = nodes[nodeIndex].children[key] {
                    nextIndex = existing
                } else {
                    nextIndex = nodes.count
                    nodes[nodeIndex].children[key] = nextIndex
                    nodes.append(Node(children: [:], bias: 0))
                }

                nodeIndex = nextIndex

                // Depth-aware scaling: stronger bias deeper into the phrase.
                let depthFactor = pow(depthScaling, Float(depth))
                let delta = baseTokenScore * depthFactor + perPhraseShare
                if delta > 0 {
                    nodes[nodeIndex].bias += delta
                }
            }
        }
    }

    /// Root node index for new sessions.
    func rootIndex() -> Int { 0 }

    /// Advance from a given node using the next word.
    ///
    /// - Parameters:
    ///   - node: current trie node index (use 0 for root).
    ///   - word: normalized (lowercased, trimmed) word key.
    /// - Returns: tuple of next node index and bias at that node.
    func advance(from node: Int, with word: String) -> (next: Int, bias: Float) {
        guard !nodes.isEmpty else { return (0, 0) }

        // If the current node has this child, follow it.
        if let child = nodes[node].children[word] {
            let nodeData = nodes[child]
            return (child, nodeData.bias)
        }

        // Simple fallback: try from root.
        if let child = nodes[0].children[word] {
            let nodeData = nodes[child]
            return (child, nodeData.bias)
        }

        // No match, stay at root with zero bias.
        return (0, 0)
    }
}

/// Trie-based engine over token IDs, matching the model vocabulary indices.
/// This is used when `CustomVocabularyTerm.tokenIds` is provided.
struct TokenVocabularyEngine {

    struct Node {
        var children: [Int: Int]
        var bias: Float
    }

    private(set) var nodes: [Node] = []

    init(context: CustomVocabularyContext) {
        nodes.append(Node(children: [:], bias: 0))

        let contextScore = context.contextScore
        let depthScaling = context.depthScaling
        let baseScorePerPhrase = context.scorePerPhrase

        for term in context.terms {
            guard let ids = term.tokenIds, !ids.isEmpty else { continue }

            let phraseLength = Float(ids.count)
            let baseTokenScore = contextScore
            let perPhraseShare: Float = phraseLength > 0 ? baseScorePerPhrase / phraseLength : 0

            var nodeIndex = 0
            for (depth, tokenId) in ids.enumerated() {
                let nextIndex: Int
                if let existing = nodes[nodeIndex].children[tokenId] {
                    nextIndex = existing
                } else {
                    nextIndex = nodes.count
                    nodes[nodeIndex].children[tokenId] = nextIndex
                    nodes.append(Node(children: [:], bias: 0))
                }

                nodeIndex = nextIndex

                let depthFactor = pow(depthScaling, Float(depth))
                let delta = baseTokenScore * depthFactor + perPhraseShare
                if delta > 0 {
                    nodes[nodeIndex].bias += delta
                }
            }
        }
    }

    func rootIndex() -> Int { 0 }

    func advance(from node: Int, withTokenId tokenId: Int) -> (next: Int, bias: Float) {
        guard !nodes.isEmpty else { return (0, 0) }

        if let child = nodes[node].children[tokenId] {
            let nodeData = nodes[child]
            return (child, nodeData.bias)
        }

        if let child = nodes[0].children[tokenId] {
            let nodeData = nodes[child]
            return (child, nodeData.bias)
        }

        return (0, 0)
    }
}
