import Foundation

/// A trie (prefix tree) structure for efficient vocabulary lookup during decoding.
/// Enables O(1) prefix matching to determine if a token sequence could lead to a vocabulary term.
///
/// This structure is immutable after construction, making it inherently thread-safe.
public struct VocabularyTrie: Sendable {

    /// A node in the vocabulary trie (value type for Sendable conformance)
    struct Node: Sendable {
        var children: [Int: Node] = [:]
        var isTerminal: Bool = false
        var term: CustomVocabularyTerm?
        var depth: Int = 0
    }

    private let root: Node
    private let termCount: Int

    /// Initialize an empty trie
    public init() {
        self.root = Node()
        self.termCount = 0
    }

    /// Initialize with vocabulary terms (builds immutable trie)
    public init(vocabulary: CustomVocabularyContext) {
        var mutableRoot = Node()
        var count = 0

        for term in vocabulary.terms {
            guard let tokenIds = term.tokenIds, !tokenIds.isEmpty else { continue }

            var path: [Int] = []

            // Build path to insertion point
            for (depth, tokenId) in tokenIds.enumerated() {
                path.append(tokenId)

                // Navigate/create path
                if Self.getNode(from: mutableRoot, path: path) == nil {
                    Self.setNode(in: &mutableRoot, path: path, node: Node(depth: depth + 1))
                }
            }

            // Mark terminal and set term
            Self.updateNode(in: &mutableRoot, path: path) { node in
                node.isTerminal = true
                node.term = term
            }
            count += 1
        }

        self.root = mutableRoot
        self.termCount = count
    }

    // MARK: - Private helpers for building the trie

    private static func getNode(from root: Node, path: [Int]) -> Node? {
        var current = root
        for tokenId in path {
            guard let next = current.children[tokenId] else {
                return nil
            }
            current = next
        }
        return current
    }

    private static func setNode(in root: inout Node, path: [Int], node: Node) {
        guard !path.isEmpty else { return }

        if path.count == 1 {
            root.children[path[0]] = node
            return
        }

        let tokenId = path[0]
        var child = root.children[tokenId] ?? Node(depth: 1)
        setNode(in: &child, path: Array(path.dropFirst()), node: node)
        root.children[tokenId] = child
    }

    private static func updateNode(in root: inout Node, path: [Int], update: (inout Node) -> Void) {
        guard !path.isEmpty else {
            update(&root)
            return
        }

        if path.count == 1 {
            var child = root.children[path[0]] ?? Node()
            update(&child)
            root.children[path[0]] = child
            return
        }

        let tokenId = path[0]
        var child = root.children[tokenId] ?? Node()
        updateNode(in: &child, path: Array(path.dropFirst()), update: update)
        root.children[tokenId] = child
    }

    // MARK: - Public API

    /// Check if a token sequence is a prefix of any vocabulary term
    /// - Parameter tokens: The token sequence to check
    /// - Returns: Tuple indicating (isPrefix, isComplete, matchedTerm)
    public func lookup(_ tokens: [Int]) -> (isPrefix: Bool, isComplete: Bool, term: CustomVocabularyTerm?) {
        var current = root
        for tokenId in tokens {
            guard let next = current.children[tokenId] else {
                return (false, false, nil)
            }
            current = next
        }
        return (true, current.isTerminal, current.term)
    }

    /// Get all possible next tokens from a given prefix
    /// - Parameter tokens: The current token sequence
    /// - Returns: Set of token IDs that could continue a vocabulary term
    public func nextTokens(after tokens: [Int]) -> Set<Int> {
        var current = root
        for tokenId in tokens {
            guard let next = current.children[tokenId] else {
                return []
            }
            current = next
        }
        return Set(current.children.keys)
    }

    /// Check if adding a token would continue or complete a vocabulary match
    /// - Parameters:
    ///   - tokens: Current token sequence
    ///   - nextToken: The token being considered
    /// - Returns: Match result with scoring information
    public func checkMatch(tokens: [Int], nextToken: Int) -> VocabularyMatchResult {
        var current = root
        for tokenId in tokens {
            guard let next = current.children[tokenId] else {
                return .noMatch
            }
            current = next
        }

        guard let next = current.children[nextToken] else {
            return .noMatch
        }

        if next.isTerminal, let term = next.term {
            return .complete(term: term, depth: next.depth)
        } else {
            return .partial(depth: next.depth, possibleContinuations: next.children.count)
        }
    }

    /// Number of terms in the trie
    public var count: Int { termCount }

    /// Check if trie is empty
    public var isEmpty: Bool { termCount == 0 }
}

/// Result of checking a vocabulary match
public enum VocabularyMatchResult: Sendable {
    /// No vocabulary term matches this prefix
    case noMatch
    /// Partial match - could lead to a vocabulary term
    case partial(depth: Int, possibleContinuations: Int)
    /// Complete match - this sequence completes a vocabulary term
    case complete(term: CustomVocabularyTerm, depth: Int)

    /// Whether this represents any kind of match
    public var isMatch: Bool {
        switch self {
        case .noMatch: return false
        case .partial, .complete: return true
        }
    }

    /// The depth of the match (0 for no match)
    public var depth: Int {
        switch self {
        case .noMatch: return 0
        case .partial(let d, _): return d
        case .complete(_, let d): return d
        }
    }
}
