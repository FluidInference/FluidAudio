import Foundation
import OSLog

/// A logging wrapper that bridges OSLog to console output for CLI usage
public class ConsoleLogger {
    private let logger: Logger
    private let category: String
    private let subsystem: String

    /// Log levels for filtering
    public enum LogLevel: String, CaseIterable, Comparable {
        case debug = "DEBUG"
        case info = "INFO"
        case warning = "WARNING"
        case error = "ERROR"
        case fault = "FAULT"

        var priority: Int {
            switch self {
            case .debug: return 0
            case .info: return 1
            case .warning: return 2
            case .error: return 3
            case .fault: return 4
            }
        }

        public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
            return lhs.priority < rhs.priority
        }
    }

    /// Current log level based on environment variable
    private static let currentLogLevel: LogLevel = {
        guard let levelString = ProcessInfo.processInfo.environment["LOG_LEVEL"],
            let level = LogLevel(rawValue: levelString.uppercased())
        else {
            return .warning  // Default to WARNING level if not specified
        }
        return level
    }()

    /// Whether console output is enabled - always true for CLI, defaults to WARNING level
    private static let consoleOutputEnabled = true

    public init(subsystem: String, category: String) {
        self.subsystem = subsystem
        self.category = category
        self.logger = Logger(subsystem: subsystem, category: category)
    }

    // MARK: - Logging Methods

    public func debug(_ message: String) {
        logger.debug("\(message)")
        if Self.consoleOutputEnabled && LogLevel.debug >= Self.currentLogLevel {
            print("[\(timestamp())] [DEBUG] [\(category)] \(message)")
        }
    }

    public func info(_ message: String) {
        logger.info("\(message)")
        if Self.consoleOutputEnabled && LogLevel.info >= Self.currentLogLevel {
            print("[\(timestamp())] [INFO] [\(category)] \(message)")
        }
    }

    public func warning(_ message: String) {
        logger.warning("\(message)")
        if Self.consoleOutputEnabled && LogLevel.warning >= Self.currentLogLevel {
            print("[\(timestamp())] [WARNING] [\(category)] \(message)")
        }
    }

    public func error(_ message: String) {
        logger.error("\(message)")
        if Self.consoleOutputEnabled && LogLevel.error >= Self.currentLogLevel {
            print("[\(timestamp())] [ERROR] [\(category)] \(message)")
        }
    }

    public func fault(_ message: String) {
        logger.fault("\(message)")
        if Self.consoleOutputEnabled && LogLevel.fault >= Self.currentLogLevel {
            print("[\(timestamp())] [FAULT] [\(category)] \(message)")
        }
    }

    // MARK: - Helper Methods

    private func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: Date())
    }

    /// Check if a specific log level would be output to console
    public func isEnabled(for level: LogLevel) -> Bool {
        return Self.consoleOutputEnabled && level >= Self.currentLogLevel
    }

    /// Get current log level for debugging
    public static func getCurrentLogLevel() -> LogLevel {
        return currentLogLevel
    }

    /// Check if console output is enabled
    public static func isConsoleOutputEnabled() -> Bool {
        return consoleOutputEnabled
    }
}
