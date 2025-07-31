#if os(macOS)
import Foundation
import FluidAudio

/// Terminal display manager for realtime transcription output
actor RealtimeDisplay {
    private var streams: [UUID: StreamDisplay] = [:]
    private let refreshRate: TimeInterval = 0.1 // 10 FPS
    private var lastRenderTime: Date = Date()
    
    struct StreamDisplay {
        var lines: [TranscriptionLine] = []
        var metrics: StreamMetrics
        let name: String
    }
    
    struct TranscriptionLine {
        let timeRange: ClosedRange<TimeInterval>
        let text: String
        let type: TranscriptionUpdate.UpdateType
        let timestamp: Date
    }
    
    /// ANSI escape codes for terminal formatting
    private enum ANSI {
        static let clear = "\u{001B}[2J\u{001B}[H"
        static let reset = "\u{001B}[0m"
        static let bold = "\u{001B}[1m"
        static let dim = "\u{001B}[2m"
        static let gray = "\u{001B}[90m"
        static let green = "\u{001B}[32m"
        static let yellow = "\u{001B}[33m"
        static let blue = "\u{001B}[34m"
        static let cyan = "\u{001B}[36m"
    }
    
    /// Update transcription for a stream
    func updateTranscription(_ update: TranscriptionUpdate, streamName: String? = nil) {
        // Initialize stream display if needed
        if streams[update.streamId] == nil {
            streams[update.streamId] = StreamDisplay(
                lines: [],
                metrics: StreamMetrics(),
                name: streamName ?? "Stream \(streams.count + 1)"
            )
        }
        
        // Add or update line
        let line = TranscriptionLine(
            timeRange: update.timeRange,
            text: update.text,
            type: update.type,
            timestamp: update.timestamp
        )
        
        if var display = streams[update.streamId] {
            // Check if we should update an existing line or add a new one
            if let existingIndex = display.lines.firstIndex(where: { 
                $0.timeRange.lowerBound == update.timeRange.lowerBound 
            }) {
                display.lines[existingIndex] = line
            } else {
                display.lines.append(line)
            }
            
            // Keep lines sorted by time
            display.lines.sort { $0.timeRange.lowerBound < $1.timeRange.lowerBound }
            
            // Limit to last 20 lines for display
            if display.lines.count > 20 {
                display.lines.removeFirst(display.lines.count - 20)
            }
            
            streams[update.streamId] = display
        }
    }
    
    /// Update metrics for a stream
    func updateMetrics(_ metrics: StreamMetrics, streamId: UUID) {
        if var display = streams[streamId] {
            display.metrics = metrics
            streams[streamId] = display
        }
    }
    
    /// Render all streams to terminal
    func render(audioFile: String? = nil) {
        // Rate limit rendering
        let now = Date()
        guard now.timeIntervalSince(lastRenderTime) >= refreshRate else { return }
        lastRenderTime = now
        
        // Clear screen
        print(ANSI.clear, terminator: "")
        
        // Header
        print("\(ANSI.bold)🎤 Realtime Transcription\(ANSI.reset)")
        if let file = audioFile {
            print("   File: \(ANSI.cyan)\(file)\(ANSI.reset)")
        }
        print(String(repeating: "━", count: 80))
        print()
        
        // Render each stream
        for (streamId, display) in streams.sorted(by: { $0.value.name < $1.value.name }) {
            renderStream(streamId: streamId, display: display)
            print()
        }
        
        // Force flush output
        fflush(stdout)
    }
    
    /// Render a single stream
    private func renderStream(streamId: UUID, display: StreamDisplay) {
        // Stream header
        if streams.count > 1 {
            print("\(ANSI.bold)\(ANSI.blue)📡 \(display.name)\(ANSI.reset)")
        }
        
        // Transcription lines
        for line in display.lines {
            let timeStr = formatTimeRange(line.timeRange)
            let textColor = getColorForType(line.type)
            let suffix = line.type == .pending ? " [pending...]" : ""
            
            print("\(ANSI.dim)[\(timeStr)]\(ANSI.reset) \(textColor)\(line.text)\(suffix)\(ANSI.reset)")
        }
        
        // Metrics
        if display.metrics.chunkCount > 0 {
            print()
            var metricsLine = ""
            
            // Time to first word
            if let ttfw = display.metrics.timeToFirstWord {
                metricsLine += "⏱️  TTFW: \(ANSI.green)\(String(format: "%.2f", ttfw))s\(ANSI.reset) "
            }
            
            // RTFx
            if display.metrics.rtfx > 0 {
                let rtfxColor = display.metrics.rtfx >= 1.0 ? ANSI.green : ANSI.yellow
                metricsLine += "│ 📊 RTFx: \(rtfxColor)\(String(format: "%.1f", display.metrics.rtfx))x\(ANSI.reset) "
            }
            
            // Average latency
            if display.metrics.averageLatency > 0 {
                metricsLine += "│ ⚡ Latency: \(ANSI.cyan)\(String(format: "%.3f", display.metrics.averageLatency))s\(ANSI.reset)"
            }
            
            print(metricsLine)
        }
    }
    
    /// Get color for update type
    private func getColorForType(_ type: TranscriptionUpdate.UpdateType) -> String {
        switch type {
        case .pending:
            return ANSI.gray
        case .partial:
            return ANSI.yellow
        case .confirmed:
            return ""  // Default color
        case .finalized:
            return ANSI.green
        }
    }
    
    /// Format time range for display
    private func formatTimeRange(_ range: ClosedRange<TimeInterval>) -> String {
        let start = String(format: "%.2f", range.lowerBound)
        let end = String(format: "%.2f", range.upperBound)
        return "\(start) - \(end)"
    }
    
    /// Clear the display
    func clear() {
        streams.removeAll()
        print(ANSI.clear)
    }
    
    /// Print final summary
    func printSummary() {
        print()
        print(String(repeating: "━", count: 80))
        print("\(ANSI.bold)📊 Final Summary\(ANSI.reset)")
        
        for (_, display) in streams.sorted(by: { $0.value.name < $1.value.name }) {
            if streams.count > 1 {
                print("\n\(ANSI.bold)\(display.name):\(ANSI.reset)")
            }
            
            print("  Total chunks processed: \(display.metrics.chunkCount)")
            print("  Total audio duration: \(String(format: "%.1f", display.metrics.totalAudioDuration))s")
            print("  Total processing time: \(String(format: "%.2f", display.metrics.totalProcessingTime))s")
            
            if let ttfw = display.metrics.timeToFirstWord {
                print("  Time to first word: \(String(format: "%.2f", ttfw))s")
            }
            
            if display.metrics.rtfx > 0 {
                let capability = display.metrics.rtfx >= 1.0 ? "✅ realtime capable" : "⚠️  below realtime"
                print("  Real-time factor: \(String(format: "%.2f", display.metrics.rtfx))x \(capability)")
            }
            
            if display.metrics.averageLatency > 0 {
                print("  Average latency: \(String(format: "%.3f", display.metrics.averageLatency))s")
            }
        }
    }
}
#endif