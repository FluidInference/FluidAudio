#if os(macOS)
import Foundation
import FluidAudio

/// Manager for streaming transcription terminal UI
@available(macOS 13.0, *)
actor StreamingUI {
    private var box: TerminalBox
    private var progressBar: ProgressBar
    private var terminalSize: (columns: Int, rows: Int)
    private var currentTranscription: String = ""
    private var volatileText: String = ""
    private var confirmedText: String = ""
    private var stats: StreamingStats = StreamingStats()
    private var isVisible: Bool = false
    private let supportsANSI: Bool

    struct StreamingStats {
        var audioDuration: Double = 0
        var elapsedTime: Double = 0
        var chunksProcessed: Int = 0
        var totalChunks: Int = 0
        var rtfx: Double = 0

        var progressPercentage: Double {
            guard totalChunks > 0 else { return 0 }
            return Double(chunksProcessed) / Double(totalChunks)
        }
    }

    init() {
        self.terminalSize = TerminalUI.getTerminalSize()
        let boxWidth = min(max(terminalSize.columns - 4, 60), 120)  // At least 60, at most 120, with 2-char margin on each side
        self.box = TerminalBox(width: boxWidth, title: "🎙️  FluidAudio Streaming Transcription")
        self.progressBar = ProgressBar(width: boxWidth - 20)
        self.supportsANSI = TerminalUI.supportsANSI
    }

    /// Initialize the UI display
    func start(audioDuration: Double, totalChunks: Int) {
        stats.audioDuration = audioDuration
        stats.totalChunks = totalChunks

        if supportsANSI {
            TerminalUI.clearScreen()
            TerminalUI.hideCursor()
            render()
            isVisible = true
        } else {
            // Fallback to simple text output
            print("🎙️ FluidAudio Streaming Transcription")
            print("Audio Duration: \(String(format: "%.1f", audioDuration))s")
            print("Simulating real-time audio with 1-second chunks...")
            print("")
        }
    }

    /// Clean up the UI
    func finish() {
        if supportsANSI && isVisible {
            TerminalUI.showCursor()
        }
    }

    /// Update progress with chunk information
    func updateProgress(chunksProcessed: Int, elapsedTime: Double) {
        stats.chunksProcessed = chunksProcessed
        stats.elapsedTime = elapsedTime

        // Calculate RTFx (Real-Time Factor)
        let audioProcessed = Double(chunksProcessed) / Double(stats.totalChunks) * stats.audioDuration
        stats.rtfx = elapsedTime > 0 ? audioProcessed / elapsedTime : 0

        if supportsANSI && isVisible {
            render()
        } else {
            // Simple progress output
            let percentage = Int(stats.progressPercentage * 100)
            print(
                "Progress: \(percentage)% (\(chunksProcessed)/\(stats.totalChunks) chunks) - \(String(format: "%.1f", stats.rtfx))x RTF"
            )
        }
    }

    /// Update transcription text
    func updateTranscription(confirmed: String, volatile: String) {
        confirmedText = confirmed
        volatileText = volatile
        currentTranscription = confirmed + (volatile.isEmpty ? "" : " " + volatile)

        if supportsANSI && isVisible {
            render()
        } else {
            // Simple text output
            if !volatile.isEmpty {
                print("Volatile: \(volatile)")
            }
            if !confirmed.isEmpty {
                print("Confirmed: \(confirmed)")
            }
        }
    }

    /// Add confirmed text update
    func addConfirmedUpdate(_ text: String) {
        if !confirmedText.isEmpty && !text.isEmpty {
            confirmedText += " "
        }
        confirmedText += text
        currentTranscription = confirmedText + (volatileText.isEmpty ? "" : " " + volatileText)

        if supportsANSI && isVisible {
            render()
        } else {
            print("Confirmed: \(text)")
        }
    }

    /// Update volatile text
    func updateVolatileText(_ text: String) {
        volatileText = text
        currentTranscription = confirmedText + (volatileText.isEmpty ? "" : " " + volatileText)

        if supportsANSI && isVisible {
            render()
        }
    }

    /// Show final results
    func showFinalResults(finalText: String, totalTime: Double) {
        let finalRtfx = totalTime > 0 ? stats.audioDuration / totalTime : 0

        if supportsANSI && isVisible {
            // Move cursor below the box and show final results
            let finalResultsRow = 12 + max(min(terminalSize.rows - 12, 6), 3) + 2
            TerminalUI.moveTo(row: finalResultsRow, column: 1)
            TerminalUI.showCursor()
            Swift.print("" + String(repeating: "═", count: 50))
            Swift.print("FINAL TRANSCRIPTION RESULTS")
            Swift.print(String(repeating: "═", count: 50))
            Swift.print("Final transcription:")
            Swift.print(finalText)
            Swift.print("Performance:")
            Swift.print("  Audio duration: \(String(format: "%.2f", stats.audioDuration))s")
            Swift.print("  Processing time: \(String(format: "%.2f", totalTime))s")
            Swift.print("  RTFx: \(String(format: "%.2f", finalRtfx))x")
        } else {
            Swift.print("" + String(repeating: "=", count: 50))
            Swift.print("FINAL TRANSCRIPTION RESULTS")
            Swift.print(String(repeating: "=", count: 50))
            Swift.print("Final transcription:")
            Swift.print(finalText)
            Swift.print("Performance:")
            Swift.print("  Audio duration: \(String(format: "%.2f", stats.audioDuration))s")
            Swift.print("  Processing time: \(String(format: "%.2f", totalTime))s")
            Swift.print("  RTFx: \(String(format: "%.2f", finalRtfx))x")
        }
    }

    /// Render the complete UI
    private func render() {
        guard supportsANSI else { return }

        // Move to top and clear
        TerminalUI.moveTo(row: 1, column: 1)

        // Header
        TerminalUI.print(box.topBorder())
        TerminalUI.print("")

        // Subtitle
        TerminalUI.print(box.contentLine(" Simulating real-time audio with 1-second chunks"))
        TerminalUI.print("")

        // Divider
        TerminalUI.print(box.middleBorder())
        TerminalUI.print("")

        // Progress section
        let progressPercent = Int(stats.progressPercentage * 100)
        let progressText =
            "\(progressBar.render(progress: stats.progressPercentage)) \(progressPercent)% (\(String(format: "%.1f", Double(stats.chunksProcessed) / Double(stats.totalChunks) * stats.audioDuration))s / \(String(format: "%.1f", stats.audioDuration))s)"
        TerminalUI.print(box.contentLine(" Progress: " + progressText))
        TerminalUI.print("")

        // Stats section
        let statsText =
            " Speed: \(String(format: "%.1f", stats.rtfx))x RTF | Chunks: \(stats.chunksProcessed)/\(stats.totalChunks) | Elapsed: \(String(format: "%.1f", stats.elapsedTime))s"
        TerminalUI.print(box.contentLine(statsText))
        TerminalUI.print("")

        // Divider
        TerminalUI.print(box.middleBorder())
        TerminalUI.print("")

        // Transcription header
        TerminalUI.print(box.contentLine(" Transcription:"))
        TerminalUI.print("")

        // Empty line
        TerminalUI.print(box.contentLine(""))
        TerminalUI.print("")

        // Transcription content - word wrap for long text
        let transcriptionLines = wrapText(currentTranscription, maxWidth: box.width - 4)

        // Calculate how many lines we can show based on terminal height
        // Terminal layout: header(1) + subtitle(1) + divider(1) + progress(1) + stats(1) + divider(1) + transcription_header(1) + empty(1) + transcription_lines + empty(1) + bottom(1) = 10 + transcription_lines
        let availableHeight = terminalSize.rows - 12  // Leave some space for final results
        let transcriptionLineCount = max(min(availableHeight, max(transcriptionLines.count, 6)), 3)  // At least 3 lines, at most availableHeight

        // Show the most recent lines (bottom of transcription) if there are more lines than we can display
        let startIndex = max(0, transcriptionLines.count - transcriptionLineCount)

        for i in 0..<transcriptionLineCount {
            let lineIndex = startIndex + i
            if lineIndex < transcriptionLines.count {
                let line = transcriptionLines[lineIndex]
                // Highlight confirmed vs volatile text
                let formattedLine = formatTranscriptionLine(line)
                TerminalUI.print(box.contentLine(" " + formattedLine))
            } else {
                TerminalUI.print(box.contentLine(""))
            }
            TerminalUI.print("")
        }

        // Empty line
        TerminalUI.print(box.contentLine(""))
        TerminalUI.print("")

        // Bottom border
        TerminalUI.print(box.bottomBorder())
    }

    /// Wrap text to fit within specified width
    private func wrapText(_ text: String, maxWidth: Int) -> [String] {
        guard !text.isEmpty else { return [""] }

        let words = text.split(separator: " ")
        var lines: [String] = []
        var currentLine = ""

        for word in words {
            let wordStr = String(word)
            if currentLine.isEmpty {
                currentLine = wordStr
            } else if currentLine.count + 1 + wordStr.count <= maxWidth {
                currentLine += " " + wordStr
            } else {
                lines.append(currentLine)
                currentLine = wordStr
            }
        }

        if !currentLine.isEmpty {
            lines.append(currentLine)
        }

        return lines.isEmpty ? [""] : lines
    }

    /// Format transcription line with colors for confirmed vs volatile text
    private func formatTranscriptionLine(_ line: String) -> String {
        guard supportsANSI else { return line }

        // Simple approach: if the line contains text from volatile part, show it dimmed
        if volatileText.isEmpty {
            return line  // All confirmed text
        }

        // Find where volatile text starts in the current line
        if line.contains(volatileText) {
            if let range = line.range(of: volatileText) {
                let before = String(line[..<range.lowerBound])
                let volatile = String(line[range])
                let after = String(line[range.upperBound...])
                return before + volatile.dim + after
            }
        }

        return line  // Default to normal formatting
    }
}

/// Simple print function for non-ANSI fallback
private func print(_ text: String) {
    Swift.print(text, terminator: "")
    fflush(stdout)
}
#endif
