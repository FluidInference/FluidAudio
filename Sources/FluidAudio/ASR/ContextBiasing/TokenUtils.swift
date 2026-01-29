import Foundation

// MARK: - Token Word Boundary Utilities

/// Check if a token string indicates a word boundary.
///
/// SentencePiece and TDT tokenizers use prefixes to indicate word starts:
/// - `▁` (U+2581 LOWER ONE EIGHTH BLOCK) - SentencePiece convention
/// - ` ` (space) - TDT/some tokenizer formats
///
/// - Parameter token: The token string to check
/// - Returns: True if the token starts a new word
public func isWordBoundary(_ token: String) -> Bool {
    token.hasPrefix("▁") || token.hasPrefix(" ")
}

/// Strip word boundary prefix from a token.
///
/// Removes the leading `▁` or space character if present.
///
/// - Parameter token: The token string to process
/// - Returns: Token with word boundary prefix removed
public func stripWordBoundaryPrefix(_ token: String) -> String {
    if token.hasPrefix("▁") || token.hasPrefix(" ") {
        return String(token.dropFirst())
    }
    return token
}
