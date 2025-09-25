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

    public static func enableConsoleOutput(_ enabled: Bool = true, minimumLevel: Level = .debug) {
        Task { await LogConsole.shared.update(enabled: enabled, minimumLevel: minimumLevel) }
    }

    public func debug(_ message: String) {
        osLogger.debug("\(message)")
        logToConsole(.debug, message)
    }

    public func info(_ message: String) {
        osLogger.info("\(message)")
        logToConsole(.info, message)
    }

    public func notice(_ message: String) {
        osLogger.notice("\(message)")
        logToConsole(.notice, message)
    }

    public func warning(_ message: String) {
        osLogger.warning("\(message)")
        logToConsole(.warning, message)
    }

    public func error(_ message: String) {
        osLogger.error("\(message)")
        logToConsole(.error, message)
    }

    public func fault(_ message: String) {
        osLogger.fault("\(message)")
        logToConsole(.fault, message)
    }

    // MARK: - Console Mirroring
    private func logToConsole(_ level: Level, _ message: String) {
        Task.detached(priority: .utility) {
            await LogConsole.shared.write(level: level, category: category, message: message)
        }
    }
}

// MARK: - Console Sink (thread-safe)

actor LogConsole {
    static let shared = LogConsole()

    private var enabled: Bool = {
        #if DEBUG
        // Enable console output for debug builds by default
        return true
        #else
        // Allow environment variable to toggle without code changes for non-debug builds
        if let env = ProcessInfo.processInfo.environment["FLUIDAUDIO_LOG_TO_CONSOLE"],
           env == "1" || env.lowercased() == "true"
        {
            return true
        }
        return false
        #endif
    }()

    private var minimumLevel: AppLogger.Level = .info
    private let dateFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss.SSS"
        return df
    }()

    func update(enabled: Bool, minimumLevel: AppLogger.Level) {
        self.enabled = enabled
        self.minimumLevel = minimumLevel
    }

    func write(level: AppLogger.Level, category: String, message: String) {
        guard enabled, level.rawValue >= minimumLevel.rawValue else { return }
        let timestamp = dateFormatter.string(from: Date())
        let line = "[\(timestamp)] [\(label(for: level))] [\(category)] \(message)\n"
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
