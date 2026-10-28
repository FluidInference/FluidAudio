import AppKit
import SwiftUI

@main
struct LocalVQEDemoApp: App {
    @StateObject private var model = DemoModel()

    init() {
        // Line-buffer stdout so `swift run … | tee log` shows progress live.
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a
        // regular windowed app with a menu bar and Dock presence.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("LocalVQE Demo") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 860, minHeight: 620)
        }
        .windowResizability(.contentSize)
    }
}
