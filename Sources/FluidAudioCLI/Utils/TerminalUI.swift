//
//  TerminalUI.swift
//  FluidAudio
//
//  Terminal UI utilities using ANSI escape codes for sophisticated display
//

import Foundation

/// ANSI escape code utilities for terminal manipulation
public struct TerminalUI {

    // MARK: - ANSI Color Codes

    public struct Color {
        public static let reset = "\u{001B}[0m"
        public static let bold = "\u{001B}[1m"
        public static let dim = "\u{001B}[2m"
        public static let italic = "\u{001B}[3m"
        public static let underline = "\u{001B}[4m"

        // Foreground colors
        public static let black = "\u{001B}[30m"
        public static let red = "\u{001B}[31m"
        public static let green = "\u{001B}[32m"
        public static let yellow = "\u{001B}[33m"
        public static let blue = "\u{001B}[34m"
        public static let magenta = "\u{001B}[35m"
        public static let cyan = "\u{001B}[36m"
        public static let white = "\u{001B}[37m"

        // Bright foreground colors
        public static let brightBlack = "\u{001B}[90m"
        public static let brightRed = "\u{001B}[91m"
        public static let brightGreen = "\u{001B}[92m"
        public static let brightYellow = "\u{001B}[93m"
        public static let brightBlue = "\u{001B}[94m"
        public static let brightMagenta = "\u{001B}[95m"
        public static let brightCyan = "\u{001B}[96m"
        public static let brightWhite = "\u{001B}[97m"

        // Background colors
        public static let bgBlack = "\u{001B}[40m"
        public static let bgRed = "\u{001B}[41m"
        public static let bgGreen = "\u{001B}[42m"
        public static let bgYellow = "\u{001B}[43m"
        public static let bgBlue = "\u{001B}[44m"
        public static let bgMagenta = "\u{001B}[45m"
        public static let bgCyan = "\u{001B}[46m"
        public static let bgWhite = "\u{001B}[47m"
    }

    // MARK: - Cursor Control

    /// Move cursor up n lines
    public static func cursorUp(_ n: Int = 1) {
        print("\u{001B}[\(n)A", terminator: "")
    }

    /// Move cursor down n lines
    public static func cursorDown(_ n: Int = 1) {
        print("\u{001B}[\(n)B", terminator: "")
    }

    /// Move cursor forward n columns
    public static func cursorForward(_ n: Int = 1) {
        print("\u{001B}[\(n)C", terminator: "")
    }

    /// Move cursor backward n columns
    public static func cursorBackward(_ n: Int = 1) {
        print("\u{001B}[\(n)D", terminator: "")
    }

    /// Move cursor to specific position (1-based)
    public static func cursorTo(row: Int, column: Int) {
        print("\u{001B}[\(row);\(column)H", terminator: "")
    }

    /// Save cursor position
    public static func saveCursor() {
        print("\u{001B}[s", terminator: "")
    }

    /// Restore cursor position
    public static func restoreCursor() {
        print("\u{001B}[u", terminator: "")
    }

    /// Hide cursor
    public static func hideCursor() {
        print("\u{001B}[?25l", terminator: "")
    }

    /// Show cursor
    public static func showCursor() {
        print("\u{001B}[?25h", terminator: "")
    }

    // MARK: - Screen Control

    /// Clear entire screen
    public static func clearScreen() {
        print("\u{001B}[2J", terminator: "")
        cursorTo(row: 1, column: 1)
    }

    /// Clear from cursor to end of screen
    public static func clearToEndOfScreen() {
        print("\u{001B}[0J", terminator: "")
    }

    /// Clear from cursor to beginning of screen
    public static func clearToBeginningOfScreen() {
        print("\u{001B}[1J", terminator: "")
    }

    /// Clear current line
    public static func clearLine() {
        print("\u{001B}[2K", terminator: "")
    }

    /// Clear from cursor to end of line
    public static func clearToEndOfLine() {
        print("\u{001B}[0K", terminator: "")
    }

    /// Clear from cursor to beginning of line
    public static func clearToBeginningOfLine() {
        print("\u{001B}[1K", terminator: "")
    }

    // MARK: - Text Formatting

    /// Apply color to text
    public static func colored(_ text: String, color: String) -> String {
        return "\(color)\(text)\(Color.reset)"
    }

    /// Apply multiple styles to text
    public static func styled(_ text: String, styles: String...) -> String {
        let styleString = styles.joined()
        return "\(styleString)\(text)\(Color.reset)"
    }

    // MARK: - Terminal Info

    /// Get terminal size
    public static func getTerminalSize() -> (width: Int, height: Int) {
        var size = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 {
            return (Int(size.ws_col), Int(size.ws_row))
        }
        // Default fallback
        return (80, 24)
    }

    // MARK: - Drawing

    /// Draw a horizontal line
    public static func drawHorizontalLine(width: Int? = nil, character: Character = "─") {
        let termWidth = width ?? getTerminalSize().width
        print(String(repeating: String(character), count: termWidth))
    }

    /// Draw a double horizontal line
    public static func drawDoubleHorizontalLine(width: Int? = nil) {
        drawHorizontalLine(width: width, character: "═")
    }

    /// Draw a box around text
    public static func drawBox(title: String, content: String, width: Int? = nil) {
        let boxWidth = width ?? getTerminalSize().width
        let contentWidth = boxWidth - 4  // Account for borders and padding

        // Top border
        print("╔" + String(repeating: "═", count: boxWidth - 2) + "╗")

        // Title
        if !title.isEmpty {
            let paddedTitle = " \(title) "
            let leftPadding = (boxWidth - paddedTitle.count - 2) / 2
            let rightPadding = boxWidth - paddedTitle.count - leftPadding - 2
            print(
                "║" + String(repeating: " ", count: leftPadding) + paddedTitle
                    + String(repeating: " ", count: rightPadding) + "║")
            print("╟" + String(repeating: "─", count: boxWidth - 2) + "╢")
        }

        // Content (word wrap if needed)
        let lines = wordWrap(text: content, width: contentWidth)
        for line in lines {
            let padding = contentWidth - line.count
            print("║ \(line)\(String(repeating: " ", count: padding)) ║")
        }

        // Bottom border
        print("╚" + String(repeating: "═", count: boxWidth - 2) + "╝")
    }

    /// Word wrap text to fit within specified width
    public static func wordWrap(text: String, width: Int) -> [String] {
        var lines: [String] = []
        var currentLine = ""

        for word in text.split(separator: " ") {
            if currentLine.isEmpty {
                currentLine = String(word)
            } else if currentLine.count + word.count + 1 <= width {
                currentLine += " " + String(word)
            } else {
                lines.append(currentLine)
                currentLine = String(word)
            }
        }

        if !currentLine.isEmpty {
            lines.append(currentLine)
        }

        return lines.isEmpty ? [""] : lines
    }

    // MARK: - Progress Bar

    /// Draw a progress bar
    public static func drawProgressBar(current: Int, total: Int, width: Int = 50, label: String = "") {
        let progress = Float(current) / Float(total)
        let filledWidth = Int(progress * Float(width))
        let emptyWidth = width - filledWidth

        let bar = String(repeating: "█", count: filledWidth) + String(repeating: "░", count: emptyWidth)
        let percentage = Int(progress * 100)

        if !label.isEmpty {
            print("\r\(label): [\(bar)] \(percentage)%", terminator: "")
        } else {
            print("\r[\(bar)] \(percentage)%", terminator: "")
        }

        if current >= total {
            print()  // New line when complete
        }
    }

    // MARK: - Animation

    private static let spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    private static var spinnerIndex = 0

    /// Show a spinner animation (call repeatedly)
    public static func showSpinner(message: String = "") {
        let frame = spinnerFrames[spinnerIndex % spinnerFrames.count]
        spinnerIndex += 1

        clearLine()
        print("\r\(frame) \(message)", terminator: "")
        fflush(stdout)
    }

    // MARK: - Utility

    /// Flush output immediately
    public static func flush() {
        fflush(stdout)
    }

    /// Reset terminal to default state
    public static func reset() {
        print(Color.reset, terminator: "")
        showCursor()
        flush()
    }
}

// MARK: - Terminal Size Structure (for ioctl)

#if os(macOS) || os(Linux)
import Darwin.POSIX.termios

private struct winsize {
    var ws_row: UInt16 = 0
    var ws_col: UInt16 = 0
    var ws_xpixel: UInt16 = 0
    var ws_ypixel: UInt16 = 0
}

private let TIOCGWINSZ: UInt = 0x4008_7468
#endif
