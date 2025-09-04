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
    private var finalizedText: String = ""
    private var stats: StreamingStats = StreamingStats()
    private var isVisible: Bool = false
    private var isInitialized: Bool = false
    private var isCompleted: Bool = false
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

    /// Show initialization UI immediately
    func showInitialization() {
        if supportsANSI {
            TerminalUI.clearScreen()
            TerminalUI.hideCursor()
            render()
            isVisible = true
        } else {
            print("FluidAudio Streaming Transcription")
            print("Initializing models and audio processing...")
            print("")
        }
    }

    /// Initialize the UI display after models are loaded
    func start(audioDuration: Double, totalChunks: Int) {
        stats.audioDuration = audioDuration
        stats.totalChunks = totalChunks
        isInitialized = true

        if supportsANSI {
            TerminalUI.clearScreen()
            TerminalUI.hideCursor()
            render()
            isVisible = true
        } else {
            // Fallback to simple text output
            print("FluidAudio Streaming Transcription")
            print("Audio Duration: \(String(format: "%.1f", audioDuration))s")
            print("Simulating real-time audio with 1-second chunks...")
            print("")
        }
    }

    /// Clean up the UI with optional wait for user input
    func finish() {
        if supportsANSI && isVisible {
            // Check if we're in CI environment
            if ProcessInfo.processInfo.environment["CI"] != nil {
                // In CI, exit immediately
                TerminalUI.showCursor()
                return
            }

            // Wait for user input before exiting (non-CI mode)
            if isCompleted {
                // Cursor is already positioned, just wait for Enter
                _ = readLine()
            }

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
    func updateTranscription(finalized: String, volatile: String) {
        finalizedText = finalized
        volatileText = volatile
        currentTranscription = finalized + (volatile.isEmpty ? "" : " " + volatile)

        if supportsANSI && isVisible {
            render()
        } else {
            // Simple text output
            if !volatile.isEmpty {
                print("Volatile: \(volatile)")
            }
            if !finalized.isEmpty {
                print("Finalized: \(finalized)")
            }
        }
    }

    func addFinalizedUpdate(_ text: String) {
        if !finalizedText.isEmpty && !text.isEmpty {
            finalizedText += " "
        }
        finalizedText += text
        currentTranscription = finalizedText + (volatileText.isEmpty ? "" : " " + volatileText)

        if supportsANSI && isVisible {
            render()
        } else {
            print("Finalized: \(text)")
        }
    }

    /// Update volatile text
    func updateVolatileText(_ text: String) {
        volatileText = text
        currentTranscription = finalizedText + (volatileText.isEmpty ? "" : " " + volatileText)

        if supportsANSI && isVisible {
            render()
        }
    }

    func showFinalResults(finalText: String, totalTime: Double) {
        finalizedText = finalText
        volatileText = ""
        currentTranscription = finalText
        isCompleted = true

        if supportsANSI && isVisible {
            render()
        } else {
            Swift.print("\n" + String(repeating: "=", count: 50))
            Swift.print("FINAL TRANSCRIPTION RESULTS")
            Swift.print(String(repeating: "=", count: 50))
            Swift.print("\nFinal transcription:")
            Swift.print(finalText)
        }
    }

    /// Render the complete UI
    private func render() {
        guard supportsANSI else { return }

        // Move to top and clear
        TerminalUI.moveTo(row: 1, column: 1)

        // Header
        TerminalUI.print(box.topBorder())
        TerminalUI.print("\n")

        // Subtitle
        if isCompleted {
            TerminalUI.print(box.contentLine(" ✅ Transcription Complete - Press Enter to exit"))
        } else if isInitialized {
            if stats.audioDuration < 0 {
                // Microphone mode
                TerminalUI.print(box.contentLine(" 🎙️ Live microphone transcription - Press Enter to stop"))
            } else {
                // File mode
                TerminalUI.print(box.contentLine(" Audio is throttled to simulate real-time streaming..."))
            }
        } else {
            TerminalUI.print(box.contentLine(" Initializing models and loading audio..."))
        }
        TerminalUI.print("\n")

        // Divider
        TerminalUI.print(box.middleBorder())
        TerminalUI.print("\n")

        // Progress section - show elapsed time vs total duration
        if isInitialized && !isCompleted {
            if stats.audioDuration < 0 {
                // Microphone mode - just show elapsed time
                let elapsedText = String(format: "%.1f", stats.elapsedTime)
                TerminalUI.print(box.contentLine(" Status: Recording... (\(elapsedText)s elapsed)"))
            }
            TerminalUI.print("\n")
        } else if isCompleted {
            // Show completion status
            let progressText = "\(progressBar.render(progress: 1.0)) 100% - Transcription completed"
            TerminalUI.print(box.contentLine(" Status: " + progressText))
            TerminalUI.print("\n")
        } else {
            TerminalUI.print(box.contentLine(" Status: Initializing models and loading audio..."))
            TerminalUI.print("\n")
        }

        // Divider
        TerminalUI.print(box.middleBorder())
        TerminalUI.print("\n")

        // Transcription header
        TerminalUI.print(box.contentLine(" Transcription:"))
        TerminalUI.print("\n")

        // Empty line
        TerminalUI.print(box.contentLine(""))
        TerminalUI.print("\n")

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
                // Highlight finalized vs volatile text
                let formattedLine = formatTranscriptionLine(line)
                TerminalUI.print(box.contentLine(" " + formattedLine))
            } else {
                TerminalUI.print(box.contentLine(""))
            }
            TerminalUI.print("\n")
        }

        // Empty line
        TerminalUI.print(box.contentLine(""))
        TerminalUI.print("\n")

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

    /// Format transcription line with colors for finalized vs volatile text
    private func formatTranscriptionLine(_ line: String) -> String {
        guard supportsANSI else { return line }

        // Simple approach: if the line contains text from volatile part, show it dimmed
        if volatileText.isEmpty {
            return line  // All finalized text
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
