import SwiftUI

@main
struct FluidAudioExampleApp: App {
    @StateObject private var viewModel = ExampleViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .frame(minWidth: 720, minHeight: 520)
        }
    }
}
