#if os(macOS)
import Foundation
import OSLog

/// Log level enum for console output filtering
public enum LogLevel: String, CaseIterable, Comparable {
    case debug = "DEBUG"
    case info = "INFO"
    case warning = "WARNING"
    case error = "ERROR"

    /// Initialize from string, case insensitive
    public init?(string: String) {
        if let level = LogLevel.allCases.first(where: { $0.rawValue.lowercased() == string.lowercased() }) {
            self = level
        } else {
            return nil
        }
    }

    /// Mapping from OSLog.Logger levels to our console levels
    public static func from(osLogLevel: OSLogEntryLog.Level) -> LogLevel {
        switch osLogLevel {
        case .debug:
            return .debug
        case .info:
            return .info
        case .notice:
            return .info
        case .error:
            return .error
        case .fault:
            return .error
        default:
            return .info
        }
    }

    /// Comparable implementation for filtering
    public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
        let order: [LogLevel] = [.debug, .info, .warning, .error]
        guard let lhsIndex = order.firstIndex(of: lhs),
            let rhsIndex = order.firstIndex(of: rhs)
        else {
            return false
        }
        return lhsIndex < rhsIndex
    }
}

/// Console logger that captures OSLog messages and outputs them to stdout
@available(macOS 13.0, *)
public final class ConsoleLogger {

    /// Shared instance for convenience
    public static let shared = ConsoleLogger()

    /// Current log level threshold
    private var logLevel: LogLevel

    /// OSLog store for reading log entries
    private let logStore: OSLogStore

    /// Task for polling log messages
    private var pollingTask: Task<Void, Never>?

    /// Last position in the log store to avoid re-reading messages
    private var lastPosition: OSLogPosition?

    /// Subsystems to monitor (FluidAudio related)
    private let monitoredSubsystems = [
        "com.fluidinfluence.asr",
        "com.fluidinfluence.diarizer",
        "com.fluidinfluence.vad",
        "com.fluidinference.fluidaudio",
        "FluidAudio",
    ]

    /// Date formatter for log timestamps
    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()

    /// Initialize with log level from environment or default
    public init(logLevel: LogLevel? = nil) {
        // Determine log level from environment variable or parameter
        if let logLevel = logLevel {
            self.logLevel = logLevel
        } else if let envLevel = ProcessInfo.processInfo.environment["LOG_LEVEL"],
            let parsedLevel = LogLevel(string: envLevel)
        {
            self.logLevel = parsedLevel
        } else {
            self.logLevel = .info  // Default level
        }

        do {
            self.logStore = try OSLogStore(scope: .currentProcessIdentifier)
        } catch {
            // Fallback: create a dummy store, logging will be disabled
            fatalError("Failed to create OSLogStore: \(error)")
        }
    }

    /// Start monitoring OSLog messages
    public func startMonitoring() {
        guard pollingTask == nil else { return }

        // Set initial position to current time
        lastPosition = logStore.position(date: Date())

        pollingTask = Task {
            await pollLogMessages()
        }
    }

    /// Stop monitoring OSLog messages
    public func stopMonitoring() {
        pollingTask?.cancel()
        pollingTask = nil
    }

    /// Set the log level threshold
    public func setLogLevel(_ level: LogLevel) {
        logLevel = level
    }

    /// Poll for new log messages
    private func pollLogMessages() async {
        while !Task.isCancelled {
            do {
                // Get entries since last position
                let entries = try logStore.getEntries(
                    at: lastPosition,
                    matching: NSPredicate(
                        format: "subsystem BEGINSWITH 'com.fluidinfluence' OR subsystem == 'FluidAudio'")
                )

                // Process new entries
                var newEntries: [OSLogEntryLog] = []
                var lastDate = Date()

                for entry in entries {
                    guard let logEntry = entry as? OSLogEntryLog else { continue }

                    // Filter by subsystem
                    if monitoredSubsystems.contains(where: { logEntry.subsystem.hasPrefix($0) }) {
                        newEntries.append(logEntry)
                        lastDate = logEntry.date
                    }
                }

                // Print new entries
                for entry in newEntries {
                    printLogEntry(entry)
                }

                // Update position for next poll
                if !newEntries.isEmpty {
                    lastPosition = logStore.position(date: lastDate.addingTimeInterval(0.001))
                }

            } catch {
                // Log store access failed, continue silently
                break
            }

            // Poll every 100ms for near real-time output
            try? await Task.sleep(nanoseconds: 100_000_000)  // 100ms
        }
    }

    /// Print a log entry to stdout if it meets the level threshold
    private func printLogEntry(_ entry: OSLogEntryLog) {
        let entryLevel = LogLevel.from(osLogLevel: entry.level)

        // Filter by log level
        guard entryLevel >= logLevel else { return }

        // Format timestamp
        let timestamp = dateFormatter.string(from: entry.date)

        // Get color for log level
        let levelColor = colorForLevel(entryLevel)
        let resetColor = "\u{001B}[0m"

        // Format the log message
        let formattedMessage =
            "[\(timestamp)] [\(levelColor)\(entryLevel.rawValue)\(resetColor)] [\(entry.category)] \(entry.composedMessage)"

        // Print to stdout
        print(formattedMessage)
    }

    /// Get ANSI color code for log level
    private func colorForLevel(_ level: LogLevel) -> String {
        // Check if terminal supports colors
        guard ProcessInfo.processInfo.environment["TERM"] != nil else {
            return ""
        }

        switch level {
        case .debug:
            return "\u{001B}[90m"  // Bright black (gray)
        case .info:
            return ""  // Default color
        case .warning:
            return "\u{001B}[33m"  // Yellow
        case .error:
            return "\u{001B}[31m"  // Red
        }
    }

    /// Convenience method to start monitoring with specific level
    public static func startMonitoring(level: LogLevel = .info) {
        let logger = ConsoleLogger.shared
        logger.setLogLevel(level)
        logger.startMonitoring()
    }

    /// Convenience method to stop monitoring
    public static func stopMonitoring() {
        ConsoleLogger.shared.stopMonitoring()
    }
}

#endif
