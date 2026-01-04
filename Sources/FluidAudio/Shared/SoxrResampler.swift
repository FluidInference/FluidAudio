import Accelerate
#if canImport(CSoxr)
import CSoxr
#endif
import Foundation

/// High-quality audio resampler using libsoxr.
/// Provides identical output to Python's librosa.resample() for maximum compatibility.
public final class SoxrResampler {
    
    /// Check if libsoxr is available
    public static var isAvailable: Bool {
        #if canImport(CSoxr)
        return true
        #else
        return false
        #endif
    }
    
    /// Resample audio from one sample rate to another using high-quality soxr algorithm.
    /// - Parameters:
    ///   - input: Input audio samples (Float32)
    ///   - inputRate: Input sample rate (e.g., 48000)
    ///   - outputRate: Output sample rate (e.g., 16000)
    /// - Returns: Resampled audio samples at outputRate
    /// - Throws: SoxrError if resampling fails or library is not installed
    public static func resample(
        _ input: [Float],
        from inputRate: Double,
        to outputRate: Double
    ) throws -> [Float] {
        guard isAvailable else {
            throw SoxrError.libraryNotInstalled
        }

        #if canImport(CSoxr)
        guard !input.isEmpty else { return [] }
        guard inputRate > 0 && outputRate > 0 else {
            throw SoxrError.invalidSampleRate
        }
        
        // If rates are the same, just return a copy
        if inputRate == outputRate {
            return input
        }
        
        // Calculate output length
        let ratio = outputRate / inputRate
        let outputLength = Int(ceil(Double(input.count) * ratio))
        
        // Create output buffer
        var output = [Float](repeating: 0, count: outputLength)
        
        var actualOutputLength = 0
        
        // Create HQ quality spec to match Python's librosa/soxr default
        // Note: soxr_oneshot defaults to LQ, but we want HQ for ML compatibility
        // SOXR_HQ = SOXR_20_BITQ = 4
        var qualitySpec = soxr_quality_spec(UInt(SOXR_HQ), 0)
        
        let result = input.withUnsafeBufferPointer { inputPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                soxr_oneshot(
                    inputRate,           // input rate
                    outputRate,          // output rate
                    1,                   // number of channels (mono)
                    inputPtr.baseAddress,  // input buffer
                    input.count,         // input length
                    nil,                 // input consumed (not needed for oneshot)
                    outputPtr.baseAddress, // output buffer
                    outputLength,        // output buffer capacity
                    &actualOutputLength, // actual output length
                    nil,                 // io_spec (use default for Float32)
                    &qualitySpec,        // quality_spec: use HQ
                    nil                  // runtime_spec (use default)
                )
            }
        }
        
        if let error = result, error != nil {
            let errorString = String(cString: error)
            throw SoxrError.resamplingFailed(errorString)
        }
        
        // Trim to actual output length
        if actualOutputLength < outputLength {
            output.removeLast(outputLength - actualOutputLength)
        }
        
        return output
        #else
        throw SoxrError.libraryNotInstalled
        #endif
    }
}

/// Errors that can occur during soxr resampling
public enum SoxrError: LocalizedError {
    case invalidSampleRate
    case resamplingFailed(String)
    case libraryNotInstalled
    
    public var errorDescription: String? {
        switch self {
        case .invalidSampleRate:
            return "Invalid sample rate (must be positive)"
        case .resamplingFailed(let message):
            return "Soxr resampling failed: \(message)"
        case .libraryNotInstalled:
            return "libsoxr is not installed. Please install it via 'brew install libsoxr' to use high-quality resampling."
        }
    }
}
