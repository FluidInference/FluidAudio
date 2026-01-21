import Foundation

/// A BK-tree (Burkhard-Keller tree) for efficient approximate string matching.
///
/// BK-trees organize strings by edit distance, enabling fast fuzzy searches.
/// Instead of comparing against all N strings (O(N)), a BK-tree query typically
/// examines only O(log N) strings for small distance thresholds.
///
/// Usage:
/// ```swift
/// let tree = BKTree(terms: vocabulary.terms)
/// let matches = tree.search(query: "nvidia", maxDistance: 2)
/// // Returns terms within edit distance 2 of "nvidia"
/// ```
public final class BKTree: @unchecked Sendable {

    /// A node in the BK-tree
    private final class Node {
        let term: CustomVocabularyTerm
        let normalizedText: String
        var children: [Int: Node] = [:]  // distance -> child node

        init(term: CustomVocabularyTerm, normalizedText: String) {
            self.term = term
            self.normalizedText = normalizedText
        }
    }

    /// Result of a BK-tree search
    public struct SearchResult: Sendable {
        public let term: CustomVocabularyTerm
        public let normalizedText: String
        public let distance: Int
    }

    private var root: Node?
    private let termCount: Int

    /// Initialize a BK-tree from vocabulary terms.
    ///
    /// - Parameter terms: Vocabulary terms to index
    /// - Complexity: O(n²) worst case, O(n log n) average
    public init(terms: [CustomVocabularyTerm]) {
        self.termCount = terms.count

        for term in terms {
            let normalized = term.text.lowercased()
            insert(term: term, normalizedText: normalized)
        }
    }

    /// Insert a term into the tree.
    private func insert(term: CustomVocabularyTerm, normalizedText: String) {
        let newNode = Node(term: term, normalizedText: normalizedText)

        guard let root = root else {
            self.root = newNode
            return
        }

        var current = root
        while true {
            let distance = StringUtils.levenshteinDistance(normalizedText, current.normalizedText)

            if let child = current.children[distance] {
                current = child
            } else {
                current.children[distance] = newNode
                break
            }
        }
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
