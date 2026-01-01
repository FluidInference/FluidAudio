import Foundation
import OSLog

/// Lightweight logger that writes to Unified Logging and, optionally, to console.
/// Use this instead of `OSLog.Logger` so CLI runs can surface logs without `print`.
public struct AppLogger: Sendable {
    /// Default subsystem for all loggers in FluidAudio.
    /// Keep this consistent; categories should vary per component.
    /// Note: Set this before creating any logger instances.
    nonisolated(unsafe) public static var defaultSubsystem: String = "com.fluidinference"

    public enum Level: Int, Sendable {
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
            logToConsole(level, message)  // Also log warnings to console in release
        case .error:
            osLogger.error("\(message)")
            logToConsole(level, message)  // Also log errors to console in release
        case .fault:
            osLogger.fault("\(message)")
            logToConsole(level, message)  // Also log faults to console in release
        }
        #endif
    }

    private func logToConsole(_ level: Level, _ message: String) {
        let capturedLevel = level
        let capturedCategory = category
        let capturedMessage = message

        // For errors/warnings/faults in release mode, log synchronously to ensure
        // the message appears before potential exit() calls
        #if !DEBUG
        if level.rawValue >= Level.warning.rawValue {
            logToConsoleSynchronously(level: capturedLevel, category: capturedCategory, message: capturedMessage)
            return
        }
        #endif

        Task.detached(priority: .utility) { [capturedLevel, capturedCategory, capturedMessage] in
            await LogConsole.shared.write(level: capturedLevel, category: capturedCategory, message: capturedMessage)
        }
    }

    private func logToConsoleSynchronously(level: Level, category: String, message: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss.SSS"
        let timestamp = dateFormatter.string(from: Date())
        let levelLabel: String
        switch level {
        case .debug: levelLabel = "DEBUG"
        case .info: levelLabel = "INFO"
        case .notice: levelLabel = "NOTICE"
        case .warning: levelLabel = "WARN"
        case .error: levelLabel = "ERROR"
        case .fault: levelLabel = "FAULT"
        }
        let line = "[\(timestamp)] [\(levelLabel)] [FluidAudio.\(category)] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
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
            do {
                try FileHandle.standardError.write(contentsOf: data)
            } catch {
                // If an I/O error occurs (e.g., the disk is full, the pipe is closed),
                // the program won't crash. Instead, you handle the error here.
                print("Fatal: Failed to write to standard error. Underlying error: \(error.localizedDescription)")
            }
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
