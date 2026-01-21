import Foundation

/// A BK-tree (Burkhard-Keller tree) for efficient approximate string matching.
///
/// BK-trees organize strings by edit distance, enabling fast fuzzy searches.
/// Instead of comparing against all N strings (O(N)), a BK-tree query typically
/// examines only O(log N) strings for small distance thresholds.
///
/// This implementation uses immutable nodes for thread safety, making the tree
/// fully Sendable without requiring `@unchecked`.
///
/// Usage:
/// ```swift
/// let tree = BKTree(terms: vocabulary.terms)
/// let matches = tree.search(query: "nvidia", maxDistance: 2)
/// // Returns terms within edit distance 2 of "nvidia"
/// ```
public struct BKTree: Sendable {

    /// An immutable node in the BK-tree.
    /// Children are set at creation time and never modified.
    private struct Node: Sendable {
        let term: CustomVocabularyTerm
        let normalizedText: String
        let children: [Int: Node]  // distance -> child node (immutable after creation)
    }

    /// Result of a BK-tree search
    public struct SearchResult: Sendable {
        public let term: CustomVocabularyTerm
        public let normalizedText: String
        public let distance: Int
    }

    private let root: Node?
    private let termCount: Int

    /// Initialize a BK-tree from vocabulary terms.
    ///
    /// - Parameter terms: Vocabulary terms to index
    /// - Complexity: O(n²) worst case, O(n log n) average
    public init(terms: [CustomVocabularyTerm]) {
        self.termCount = terms.count

        // Build the tree using immutable nodes
        let normalizedTerms = terms.map { ($0, $0.text.lowercased()) }
        self.root = Self.buildTree(from: normalizedTerms)
    }

    /// Build an immutable tree recursively.
    ///
    /// This approach creates all nodes with their children set at creation time,
    /// ensuring the entire tree structure is immutable after construction.
    private static func buildTree(from terms: [(CustomVocabularyTerm, String)]) -> Node? {
        guard let first = terms.first else { return nil }

        // Group remaining terms by their distance from the first term
        var groups: [Int: [(CustomVocabularyTerm, String)]] = [:]
        for item in terms.dropFirst() {
            let dist = StringUtils.levenshteinDistance(item.1, first.1)
            groups[dist, default: []].append(item)
        }

        // Recursively build children (each group becomes a subtree)
        var children: [Int: Node] = [:]
        for (dist, group) in groups {
            if let child = buildTree(from: group) {
                children[dist] = child
            }
        }

        return Node(term: first.0, normalizedText: first.1, children: children)
    }

    /// Search for terms within a maximum edit distance of the query.
    ///
    /// - Parameters:
    ///   - query: The search string
    ///   - maxDistance: Maximum Levenshtein distance (inclusive)
    /// - Returns: All terms within the specified distance, sorted by distance
    /// - Complexity: O(log n) average for small maxDistance, O(n) worst case
    public func search(query: String, maxDistance: Int) -> [SearchResult] {
        guard let root = root else { return [] }

        let normalizedQuery = query.lowercased()
        var results: [SearchResult] = []
        var stack: [Node] = [root]

        while let node = stack.popLast() {
            let distance = StringUtils.levenshteinDistance(normalizedQuery, node.normalizedText)

            if distance <= maxDistance {
                results.append(
                    SearchResult(
                        term: node.term,
                        normalizedText: node.normalizedText,
                        distance: distance
                    ))
            }

            // BK-tree property: only traverse children where
            // |child_edge - distance| <= maxDistance
            let minEdge = max(0, distance - maxDistance)
            let maxEdge = distance + maxDistance

            for (edge, child) in node.children {
                if edge >= minEdge && edge <= maxEdge {
                    stack.append(child)
                }
            }
        }

        return results.sorted { $0.distance < $1.distance }
    }

    /// Check if the tree is empty.
    public var isEmpty: Bool {
        return root == nil
    }

    /// Number of terms in the tree.
    public var count: Int {
        return termCount
    }
}
