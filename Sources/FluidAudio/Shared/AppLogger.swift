import Foundation
import OSLog

/// Lightweight logger that writes to Unified Logging and, optionally, to console.
/// Use this instead of `OSLog.Logger` so CLI runs can surface logs without `print`.
public struct AppLogger {
    /// Default subsystem for all loggers in FluidAudio.
    /// Keep this consistent; categories should vary per component.
    public static var defaultSubsystem: String = "com.fluidinference"

    public enum Level: Int {
        case debug = 0
        case info
        case notice
        case warning
        case error
        case fault
    }

    private let osLogger: Logger
    private let subsystem: String
    private let category: String

    /// Designated initializer allowing a custom subsystem if needed.
    public init(subsystem: String, category: String) {
        self.osLogger = Logger(subsystem: subsystem, category: category)
        self.subsystem = subsystem
        self.category = category
    }

    /// Convenience initializer that uses the shared default subsystem.
    public init(category: String) {
        self.init(subsystem: AppLogger.defaultSubsystem, category: category)
    }

    // MARK: - Public API

    public func debug(_ message: String) {
        log(.debug, message)
    }

    public func info(_ message: String) {
        log(.info, message)
    }

    public func notice(_ message: String) {
        log(.notice, message)
    }

    public func warning(_ message: String) {
        log(.warning, message)
    }

    public func error(_ message: String) {
        log(.error, message)
    }

    public func fault(_ message: String) {
        log(.fault, message)
    }

    // MARK: - Console Mirroring
    private func log(_ level: Level, _ message: String) {
        #if DEBUG
        logToConsole(level, message)
        #else
        switch level {
        case .debug:
            osLogger.debug("\(message)")
        case .info:
            osLogger.info("\(message)")
        case .notice:
            osLogger.notice("\(message)")
        case .warning:
            osLogger.warning("\(message)")
        case .error:
            osLogger.error("\(message)")
        case .fault:
            osLogger.fault("\(message)")
        }
        #endif
    }

    private func logToConsole(_ level: Level, _ message: String) {
        Task.detached(priority: .utility) {
            await LogConsole.shared.write(level: level, category: category, message: message)
        }
    }
}

// MARK: - Console Sink (thread-safe)

actor LogConsole {
    static let shared = LogConsole()

    private let dateFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss.SSS"
        return df
    }()

    func write(level: AppLogger.Level, category: String, message: String) {
        let timestamp = dateFormatter.string(from: Date())
        let line = "[\(timestamp)] [\(label(for: level))] [FluidAudio.\(category)] \(message)\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }

    private func label(for level: AppLogger.Level) -> String {
        switch level {
        case .debug: return "DEBUG"
        case .info: return "INFO"
        case .notice: return "NOTICE"
        case .warning: return "WARN"
        case .error: return "ERROR"
        case .fault: return "FAULT"
        }
    }
}
