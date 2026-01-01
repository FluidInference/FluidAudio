import Foundation

/// A trie (prefix tree) structure for efficient vocabulary lookup during decoding.
/// Enables O(1) prefix matching to determine if a token sequence could lead to a vocabulary term.
///
/// This structure is immutable after construction, making it inherently thread-safe.
public struct VocabularyTrie: Sendable {

    /// A node in the vocabulary trie (class type for reference semantics in cursors)
    /// Note: Properties are mutated only during trie construction, then immutable.
    public final class Node: Sendable {
        public nonisolated(unsafe) var children: [Int: Node] = [:]
        public nonisolated(unsafe) var isTerminal: Bool = false
        public nonisolated(unsafe) var term: CustomVocabularyTerm?
        public nonisolated(unsafe) var depth: Int = 0

        public init(depth: Int = 0) {
            self.depth = depth
        }
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
        let rootNode = Node()
        var count = 0

        for term in vocabulary.terms {
            guard let tokenIds = term.tokenIds, !tokenIds.isEmpty else { continue }

            var current = rootNode
            for (depth, tokenId) in tokenIds.enumerated() {
                if let next = current.children[tokenId] {
                    current = next
                } else {
                    let newNode = Node(depth: depth + 1)
                    current.children[tokenId] = newNode
                    current = newNode
                }
            }

            current.isTerminal = true
            current.term = term
            count += 1
        }

        self.root = rootNode
        self.termCount = count
    }

    /// Cursor for stateful traversal of the trie
    public struct Cursor {
        private let root: Node
        private var current: Node

        fileprivate init(root: Node) {
            self.root = root
            self.current = root
        }

        /// Current depth in the trie
        public var depth: Int { current.depth }

        /// Reset cursor to root
        public mutating func reset() {
            current = root
        }

        /// Attempt to advance the cursor with a token
        /// - Returns: Match result for the move
        public mutating func advance(_ tokenId: Int) -> VocabularyMatchResult {
            // 1. Try to extend current path
            if let next = current.children[tokenId] {
                current = next
                if next.isTerminal, let term = next.term {
                    return .complete(term: term, depth: next.depth)
                }
                return .partial(depth: next.depth, possibleContinuations: next.children.count)
            }

            // 2. If failed, this path is dead. Reset to root.
            // Note: The caller might want to check if 'tokenId' starts a NEW match at root
            // but that logic is best handled by the caller or a specific 'advanceOrReset' method.
            // Here we just report no match for the *continuation*.
            return .noMatch
        }

        /// Check if a token would start a new match from root
        public func startsMatch(_ tokenId: Int) -> VocabularyMatchResult {
            if let next = root.children[tokenId] {
                if next.isTerminal, let term = next.term {
                    return .complete(term: term, depth: next.depth)
                }
                return .partial(depth: next.depth, possibleContinuations: next.children.count)
            }
            return .noMatch
        }

        /// Force the cursor to a specific state (e.g. after finding a new match from root)
        public mutating func forceAdvanceFromRoot(_ tokenId: Int) {
            if let next = root.children[tokenId] {
                current = next
            } else {
                current = root
            }
        }
    }

    /// Create a new cursor at the root
    public func makeCursor() -> Cursor {
        return Cursor(root: root)
    }

    // MARK: - Legacy/Stateless API (kept for compatibility but implemented via Cursor)

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
