//
//  StreamingDisplay.swift
//  FluidAudio
//
//  Manages the streaming transcription display with real-time updates
//

import FluidAudio
import Foundation

/// Manages the display of streaming transcription with volatile and confirmed text
@available(macOS 13.0, iOS 16.0, *)
public actor StreamingDisplay {

    // Display state
    private var confirmedText: String = ""
    private var volatileText: String = ""
    private var lastVolatileText: String = ""
    private var statistics: Statistics
    private let config: DisplayConfig
    private var isActive: Bool = false
    private var lastRenderTime: Date = Date()

    // Terminal dimensions
    private var terminalWidth: Int = 80
    private var terminalHeight: Int = 24

    public struct DisplayConfig {
        let showStatistics: Bool
        let showConfidence: Bool
        let highlightNewText: Bool
        let maxVolatileLines: Int
        let maxConfirmedLines: Int
        let updateInterval: TimeInterval  // Minimum time between renders

        public static let `default` = DisplayConfig(
            showStatistics: true,
            showConfidence: true,
            highlightNewText: true,
            maxVolatileLines: 5,
            maxConfirmedLines: 20,  // Increased to show more confirmed text
            updateInterval: 0.05  // 20 FPS max
        )

        public init(
            showStatistics: Bool = true,
            showConfidence: Bool = true,
            highlightNewText: Bool = true,
            maxVolatileLines: Int = 5,
            maxConfirmedLines: Int = 10,
            updateInterval: TimeInterval = 0.05
        ) {
            self.showStatistics = showStatistics
            self.showConfidence = showConfidence
            self.highlightNewText = highlightNewText
            self.maxVolatileLines = maxVolatileLines
            self.maxConfirmedLines = maxConfirmedLines
            self.updateInterval = updateInterval
        }
    }

    struct Statistics {
        var totalChunks: Int = 0
        var confirmedChunks: Int = 0
        var volatileChunks: Int = 0
        var currentConfidence: Float = 0.0
        var averageConfidence: Float = 0.0
        var confidenceThreshold: Float = 0.5
        var processingTime: TimeInterval = 0
        var audioLength: TimeInterval = 0
        var rtfx: Float = 0.0

        var confidenceHistory: [Float] = []
        let maxHistory = 20

        mutating func updateConfidence(_ confidence: Float) {
            currentConfidence = confidence
            confidenceHistory.append(confidence)
            if confidenceHistory.count > maxHistory {
                confidenceHistory.removeFirst()
            }
            averageConfidence = confidenceHistory.reduce(0, +) / Float(confidenceHistory.count)
        }
    }

    public init(config: DisplayConfig = .default) {
        self.config = config
        self.statistics = Statistics()
        // Terminal size will be updated on start
        self.terminalWidth = 80
        self.terminalHeight = 24
    }

    // MARK: - Public Methods

    /// Start the display
    public func start(confidenceThreshold: Float) {
        isActive = true
        statistics.confidenceThreshold = confidenceThreshold
        updateTerminalSize()
        TerminalUI.clearScreen()
        TerminalUI.hideCursor()
        render()
    }

    /// Stop the display
    public func stop() {
        isActive = false
        TerminalUI.showCursor()
        TerminalUI.cursorTo(row: terminalHeight, column: 1)
        print()  // New line
    }

    /// Show completion status while keeping display active
    public func showCompletion() {
        // Update the display to show completion status
        // This will be shown in the header or footer
        render()
    }

    /// Update with new transcription data
    public func update(
        confirmed: String?,
        volatile: String?,
        confidence: Float,
        isNewChunk: Bool
    ) {
        if let confirmed = confirmed {
            confirmedText = confirmed
            if isNewChunk {
                statistics.confirmedChunks += 1
            }
        }

        if let volatile = volatile {
            // Detect if volatile text has changed
            if volatile != lastVolatileText {
                volatileText = volatile
                lastVolatileText = volatile
                if isNewChunk {
                    statistics.volatileChunks += 1
                }
            }
        }

        statistics.totalChunks = statistics.confirmedChunks + statistics.volatileChunks
        statistics.updateConfidence(confidence)

        // Throttle rendering to avoid flickering
        let now = Date()
        if now.timeIntervalSince(lastRenderTime) >= config.updateInterval {
            render()
            lastRenderTime = now
        }
    }

    /// Update processing statistics
    public func updateStatistics(
        processingTime: TimeInterval,
        audioLength: TimeInterval
    ) {
        statistics.processingTime = processingTime
        statistics.audioLength = audioLength
        statistics.rtfx = audioLength > 0 ? Float(audioLength / processingTime) : 0
        render()
    }

    // MARK: - Private Methods

    private func updateTerminalSize() {
        let size = TerminalUI.getTerminalSize()
        terminalWidth = size.width
        terminalHeight = size.height
    }

    private func render() {
        guard isActive else { return }

        TerminalUI.cursorTo(row: 1, column: 1)
        TerminalUI.clearToEndOfScreen()

        // Header
        renderHeader()

        // Confirmed text section
        renderConfirmedText()

        // Volatile text section
        renderVolatileText()

        // Statistics section
        if config.showStatistics {
            renderStatistics()
        }

        // Progress indicator
        renderProgressIndicator()

        TerminalUI.flush()
    }

    private func renderHeader() {
        let headerLine = String(repeating: "═", count: terminalWidth)
        print(TerminalUI.colored(headerLine, color: TerminalUI.Color.brightCyan))

        let title = "🎙️  LIVE TRANSCRIPTION"
        let timeInfo = String(
            format: "[Processing: %.1fs / %.1fs]",
            statistics.processingTime,
            statistics.audioLength)

        let padding = max(0, terminalWidth - title.count - timeInfo.count - 2)
        let paddingStr = String(repeating: " ", count: padding)

        print(
            TerminalUI.styled(title, styles: TerminalUI.Color.bold, TerminalUI.Color.brightWhite)
                + paddingStr
                + TerminalUI.colored(timeInfo, color: TerminalUI.Color.brightYellow))

        print(TerminalUI.colored(headerLine, color: TerminalUI.Color.brightCyan))
        print()
    }

    private func renderConfirmedText() {
        // Section header
        let avgConfStr = String(format: "%.1f%%", statistics.averageConfidence * 100)
        let header = "📝 CONFIRMED TEXT (\(avgConfStr) avg confidence):"
        print(TerminalUI.styled(header, styles: TerminalUI.Color.bold, TerminalUI.Color.green))

        let divider = String(repeating: "─", count: min(terminalWidth, 80))
        print(TerminalUI.colored(divider, color: TerminalUI.Color.dim))

        // Display confirmed text with word wrapping
        if confirmedText.isEmpty {
            print(TerminalUI.colored("  (No confirmed text yet)", color: TerminalUI.Color.dim))
        } else {
            let lines = wrapText(confirmedText, width: terminalWidth - 4)
            let linesToShow = min(lines.count, config.maxConfirmedLines)
            let startLine = max(0, lines.count - linesToShow)

            for i in startLine..<(startLine + linesToShow) {
                print("  " + TerminalUI.colored(lines[i], color: TerminalUI.Color.brightGreen))
            }

            if lines.count > config.maxConfirmedLines {
                let hiddenLines = lines.count - config.maxConfirmedLines
                print(TerminalUI.colored("  ... (\(hiddenLines) more lines)", color: TerminalUI.Color.dim))
            }
        }
        print()
    }

    private func renderVolatileText() {
        // Section header with animated indicator
        let confidenceStr = String(format: "%.1f%%", statistics.currentConfidence * 100)
        let confidenceColor = getConfidenceColor(statistics.currentConfidence)

        let processingIndicator = isProcessing() ? " ⚡" : ""
        let header = "💭 VOLATILE TEXT (\(confidenceStr) confidence)\(processingIndicator):"

        print(TerminalUI.styled(header, styles: TerminalUI.Color.bold, confidenceColor))

        let divider = String(repeating: "─", count: min(terminalWidth, 80))
        print(TerminalUI.colored(divider, color: TerminalUI.Color.dim))

        // Display volatile text with word wrapping
        if volatileText.isEmpty {
            print(TerminalUI.colored("  (Processing...)", color: TerminalUI.Color.dim))
        } else {
            let lines = wrapText(volatileText, width: terminalWidth - 4)
            let linesToShow = min(lines.count, config.maxVolatileLines)

            for i in 0..<linesToShow {
                // Highlight new text if enabled
                if config.highlightNewText && i == lines.count - 1 {
                    print(
                        "  "
                            + TerminalUI.styled(
                                lines[i],
                                styles: TerminalUI.Color.italic,
                                confidenceColor))
                } else {
                    print("  " + TerminalUI.colored(lines[i], color: confidenceColor))
                }
            }

            if lines.count > config.maxVolatileLines {
                let hiddenLines = lines.count - config.maxVolatileLines
                print(TerminalUI.colored("  ... (\(hiddenLines) more lines)", color: TerminalUI.Color.dim))
            }
        }
        print()
    }

    private func renderStatistics() {
        print(TerminalUI.styled("📊 Statistics:", styles: TerminalUI.Color.bold, TerminalUI.Color.brightCyan))

        // Create statistics lines
        var stats: [String] = []

        stats.append(
            "• Chunks: \(statistics.totalChunks) total | " + "\(statistics.confirmedChunks) confirmed | "
                + "\(statistics.volatileChunks) volatile")

        stats.append(
            String(
                format: "• Confidence: Current %.1f%% | Avg %.1f%% | Threshold %.1f%%",
                statistics.currentConfidence * 100,
                statistics.averageConfidence * 100,
                statistics.confidenceThreshold * 100))

        if statistics.rtfx > 0 {
            stats.append(
                String(
                    format: "• Performance: RTFx %.1fx | Processing %.2fs",
                    statistics.rtfx,
                    statistics.processingTime))
        }

        for stat in stats {
            print(TerminalUI.colored(stat, color: TerminalUI.Color.cyan))
        }
        print()
    }

    private func renderProgressIndicator() {
        // Show confidence trend as a mini bar chart
        if config.showConfidence && !statistics.confidenceHistory.isEmpty {
            print(TerminalUI.colored("Confidence Trend:", color: TerminalUI.Color.dim))
            let chart = createMiniChart(statistics.confidenceHistory, width: min(40, terminalWidth - 4))
            print("  " + chart)
        }

        print()
        print(TerminalUI.colored("[Press Ctrl+C to stop]", color: TerminalUI.Color.dim))
    }

    // MARK: - Helper Methods

    private func wrapText(_ text: String, width: Int) -> [String] {
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

        return lines
    }

    private func getConfidenceColor(_ confidence: Float) -> String {
        switch confidence {
        case 0.8...:
            return TerminalUI.Color.brightGreen
        case 0.6..<0.8:
            return TerminalUI.Color.green
        case 0.4..<0.6:
            return TerminalUI.Color.yellow
        case 0.2..<0.4:
            return TerminalUI.Color.brightYellow
        default:
            return TerminalUI.Color.red
        }
    }

    private func isProcessing() -> Bool {
        // Simple animation based on time
        return Int(Date().timeIntervalSince1970 * 2) % 2 == 0
    }

    private func createMiniChart(_ values: [Float], width: Int) -> String {
        guard !values.isEmpty else { return "" }

        let bars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        var chart = ""

        // Sample values if we have more than width
        let step = max(1, values.count / width)
        let sampledValues = stride(from: 0, to: values.count, by: step).map { values[$0] }

        for value in sampledValues.prefix(width) {
            let barIndex = Int(value * Float(bars.count - 1))
            let safeIndex = min(max(0, barIndex), bars.count - 1)
            let color = getConfidenceColor(value)
            chart += TerminalUI.colored(bars[safeIndex], color: color)
        }

        return chart
    }
}
