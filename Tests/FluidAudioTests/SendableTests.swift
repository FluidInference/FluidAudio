import XCTest

@testable import FluidAudio

final class SendableTests: XCTestCase {

    func testAsrModelsIsSendable() {
        // This test verifies that AsrModels conforms to Sendable
        // If it doesn't, this won't compile
        func requiresSendable<T: Sendable>(_: T.Type) {}
        requiresSendable(AsrModels.self)
    }

    func testDiarizerModelsIsSendable() {
        // This test verifies that DiarizerModels conforms to Sendable
        // If it doesn't, this won't compile
        func requiresSendable<T: Sendable>(_: T.Type) {}
        requiresSendable(DiarizerModels.self)
    }

    func testErrorTypesAreSendable() {
        // Verify error types are also Sendable
        func requiresSendable<T: Sendable>(_: T.Type) {}
        requiresSendable(AsrModelsError.self)
        requiresSendable(DiarizerError.self)
    }
}
