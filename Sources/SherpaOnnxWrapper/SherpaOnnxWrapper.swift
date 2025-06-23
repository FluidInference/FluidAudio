import Foundation

// This is a Swift wrapper for the SherpaOnnx C API
// The actual C functions are defined in the c-api.h header file

// Re-export the C functions so they can be used from Swift
@_exported import SherpaOnnxWrapperC

// You can add Swift convenience functions here if needed
public struct SherpaOnnxWrapper {
    // Placeholder for Swift wrapper functionality
    public static func version() -> String {
        return "1.0.0"
    }
}
