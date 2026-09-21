import AppKit
import SwiftUI

@main
struct LayaTetrisDemoApp: App {
    @StateObject private var model = GameModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a
        // regular windowed app with a menu bar and Dock presence.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("laya plays Tetris") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 980, minHeight: 640)
                .onAppear { model.applyEnvironment() }
        }
        .windowResizability(.contentSize)
    }
}
