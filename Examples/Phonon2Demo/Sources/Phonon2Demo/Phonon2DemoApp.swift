import AppKit
import SwiftUI

@main
struct Phonon2DemoApp: App {
    @StateObject private var model = DemoModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a
        // regular windowed app with a menu bar and Dock presence.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Phonon-2 on FluidAudio") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 820, minHeight: 760)
        }
    }
}
