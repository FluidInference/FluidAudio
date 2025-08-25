#if os(macOS)
import Foundation
import Darwin

/// Terminal UI utilities using ANSI escape codes
enum TerminalUI {
    // ANSI Escape Codes
    static let ESC = "\u{001B}"
    static let CSI = ESC + "["

    // Cursor Control
    enum Cursor {
        static func up(_ lines: Int = 1) -> String { return CSI + "\(lines)A" }
        static func down(_ lines: Int = 1) -> String { return CSI + "\(lines)B" }
        static func forward(_ columns: Int = 1) -> String { return CSI + "\(columns)C" }
        static func back(_ columns: Int = 1) -> String { return CSI + "\(columns)D" }
        static func position(row: Int, column: Int) -> String { return CSI + "\(row);\(column)H" }
        static let save = CSI + "s"
        static let restore = CSI + "u"
        static let hide = CSI + "?25l"
        static let show = CSI + "?25h"
    }

    // Screen Control
    enum Screen {
        static let clear = CSI + "2J" + CSI + "H"
        static let clearLine = CSI + "2K"
        static let clearToEndOfLine = CSI + "K"
        static let clearToStartOfLine = CSI + "1K"
    }

    // Colors
    enum Color {
        static let reset = CSI + "0m"
        static let bold = CSI + "1m"
        static let dim = CSI + "2m"
        static let normal = CSI + "22m"

        // Foreground colors
        static let black = CSI + "30m"
        static let red = CSI + "31m"
        static let green = CSI + "32m"
        static let yellow = CSI + "33m"
        static let blue = CSI + "34m"
        static let magenta = CSI + "35m"
        static let cyan = CSI + "36m"
        static let white = CSI + "37m"
        static let gray = CSI + "90m"

        // Background colors
        static let bgBlack = CSI + "40m"
        static let bgRed = CSI + "41m"
        static let bgGreen = CSI + "42m"
        static let bgYellow = CSI + "43m"
        static let bgBlue = CSI + "44m"
        static let bgMagenta = CSI + "45m"
        static let bgCyan = CSI + "46m"
        static let bgWhite = CSI + "47m"
    }

    /// Check if terminal supports ANSI escape codes
    static var supportsANSI: Bool {
        guard let term = ProcessInfo.processInfo.environment["TERM"] else { return false }
        return !term.isEmpty && term != "dumb"
    }

    /// Print without newline and flush output
    static func print(_ text: String, terminator: String = "") {
        Swift.print(text, terminator: terminator)
        fflush(stdout)
    }

    /// Clear the screen and move cursor to top-left
    static func clearScreen() {
        print(Screen.clear)
    }

    /// Move cursor to specific position (1-indexed)
    static func moveTo(row: Int, column: Int) {
        print(Cursor.position(row: row, column: column))
    }

    /// Clear current line
    static func clearCurrentLine() {
        print(Screen.clearLine)
    }

    /// Hide cursor
    static func hideCursor() {
        print(Cursor.hide)
    }

    /// Show cursor
    static func showCursor() {
        print(Cursor.show)
    }

    /// Get terminal size (columns, rows)
    static func getTerminalSize() -> (columns: Int, rows: Int) {
        var winsize = winsize()
        let result = ioctl(STDOUT_FILENO, TIOCGWINSZ, &winsize)

        if result == 0 && winsize.ws_col > 0 && winsize.ws_row > 0 {
            return (Int(winsize.ws_col), Int(winsize.ws_row))
        } else {
            // Default fallback size
            return (80, 24)
        }
    }
}

/// Progress bar component
struct ProgressBar {
    let width: Int
    let fillChar: Character
    let emptyChar: Character

    init(width: Int = 40, fillChar: Character = "█", emptyChar: Character = "░") {
        self.width = width
        self.fillChar = fillChar
        self.emptyChar = emptyChar
    }

    /// Generate progress bar string
    func render(progress: Double) -> String {
        let clampedProgress = max(0, min(1, progress))
        let filledWidth = Int(Double(width) * clampedProgress)
        let emptyWidth = width - filledWidth

        let filled = String(repeating: String(fillChar), count: filledWidth)
        let empty = String(repeating: String(emptyChar), count: emptyWidth)

        return "[\(filled)\(empty)]"
    }
}

/// Box drawing characters for terminal UI
enum BoxChars {
    static let topLeft = "╔"
    static let topRight = "╗"
    static let bottomLeft = "╚"
    static let bottomRight = "╝"
    static let horizontal = "═"
    static let vertical = "║"
    static let topTee = "╦"
    static let bottomTee = "╩"
    static let leftTee = "╠"
    static let rightTee = "╣"
    static let cross = "╬"
}

/// Terminal box component
struct TerminalBox {
    let width: Int
    let title: String?

    init(width: Int = 60, title: String? = nil) {
        self.width = width
        self.title = title
    }

    /// Generate top border
    func topBorder() -> String {
        let innerWidth = width - 2
        if let title = title {
            let titleWithSpaces = " \(title) "
            let remainingWidth = innerWidth - titleWithSpaces.count
            let leftPadding = remainingWidth / 2
            let rightPadding = remainingWidth - leftPadding

            return BoxChars.topLeft + String(repeating: BoxChars.horizontal, count: leftPadding) + titleWithSpaces
                + String(repeating: BoxChars.horizontal, count: rightPadding) + BoxChars.topRight
        } else {
            return BoxChars.topLeft + String(repeating: BoxChars.horizontal, count: innerWidth) + BoxChars.topRight
        }
    }

    /// Generate middle border (divider)
    func middleBorder() -> String {
        return BoxChars.leftTee + String(repeating: BoxChars.horizontal, count: width - 2) + BoxChars.rightTee
    }

    /// Generate bottom border
    func bottomBorder() -> String {
        return BoxChars.bottomLeft + String(repeating: BoxChars.horizontal, count: width - 2) + BoxChars.bottomRight
    }

    /// Generate content line with borders
    func contentLine(_ content: String) -> String {
        let paddedContent = content.padding(toLength: width - 2, withPad: " ", startingAt: 0)
        return BoxChars.vertical + paddedContent + BoxChars.vertical
    }
}

/// String extensions for terminal formatting
extension String {
    func colored(_ color: String) -> String {
        return color + self + TerminalUI.Color.reset
    }

    var bold: String { colored(TerminalUI.Color.bold) }
    var dim: String { colored(TerminalUI.Color.dim) }
    var red: String { colored(TerminalUI.Color.red) }
    var green: String { colored(TerminalUI.Color.green) }
    var yellow: String { colored(TerminalUI.Color.yellow) }
    var blue: String { colored(TerminalUI.Color.blue) }
    var magenta: String { colored(TerminalUI.Color.magenta) }
    var cyan: String { colored(TerminalUI.Color.cyan) }
    var gray: String { colored(TerminalUI.Color.gray) }
}
#endif
