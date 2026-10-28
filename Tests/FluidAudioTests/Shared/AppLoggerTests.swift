import XCTest

@testable import FluidAudio

final class AppLoggerTests: XCTestCase {

    private var savedMinimumLevel: AppLogger.Level = .debug
    private var savedMirrorsToConsole = true

    override func setUp() {
        super.setUp()
        savedMinimumLevel = AppLogger.minimumLevel
        savedMirrorsToConsole = AppLogger.mirrorsToConsole
    }

    override func tearDown() {
        AppLogger.minimumLevel = savedMinimumLevel
        AppLogger.mirrorsToConsole = savedMirrorsToConsole
        super.tearDown()
    }

    func testLevelOrdering() {
        XCTAssertLessThan(AppLogger.Level.debug, .info)
        XCTAssertLessThan(AppLogger.Level.info, .notice)
        XCTAssertLessThan(AppLogger.Level.notice, .warning)
        XCTAssertLessThan(AppLogger.Level.warning, .error)
        XCTAssertLessThan(AppLogger.Level.error, .fault)
    }

    func testMinimumLevelDropsLowerLevelsFromAllSinks() {
        AppLogger.minimumLevel = .warning
        let dropped = AppLogger.Route(osLog: false, console: false)
        XCTAssertEqual(AppLogger.route(for: .debug), dropped)
        XCTAssertEqual(AppLogger.route(for: .info), dropped)
        XCTAssertEqual(AppLogger.route(for: .notice), dropped)
        XCTAssertNotEqual(AppLogger.route(for: .warning), dropped)
        XCTAssertNotEqual(AppLogger.route(for: .fault), dropped)
    }

    func testMirrorsToConsoleOffNeverWritesConsole() {
        AppLogger.mirrorsToConsole = false
        for level: AppLogger.Level in [.debug, .info, .notice, .warning, .error, .fault] {
            let route = AppLogger.route(for: level)
            XCTAssertFalse(route.console, "\(level) reached console")
            XCTAssertTrue(route.osLog, "\(level) missing from os_log")
        }
    }

    func testDefaultRouting() {
        AppLogger.minimumLevel = .debug
        AppLogger.mirrorsToConsole = true
        #if DEBUG
        XCTAssertEqual(AppLogger.route(for: .debug), AppLogger.Route(osLog: false, console: true))
        XCTAssertEqual(AppLogger.route(for: .error), AppLogger.Route(osLog: false, console: true))
        #else
        XCTAssertEqual(AppLogger.route(for: .debug), AppLogger.Route(osLog: true, console: false))
        XCTAssertEqual(AppLogger.route(for: .error), AppLogger.Route(osLog: true, console: true))
        #endif
    }
}
