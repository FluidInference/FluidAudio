import Foundation
import CoreML
import Accelerate

public typealias LSEENDVariant = ModelNames.LSEEND.Variant
public typealias LSEENDStepSize = ModelNames.LSEEND.StepSize

// MARK: - Metadata

public struct LSEENDMetadata: Codable {
    /// Number of output frames to process per model call
    public let chunkSize: Int

    /// Duration of output frames in seconds
    public let frameDurationSeconds: Float

    /// Max output speakers
    public let maxSpeakers: Int

    /// Required audio sample rate
    public let sampleRate: Int

    /// Number of attractors. Not output speakers
    public let maxNspks: Int

    /// Mel hop length
    public let hopLength: Int

    /// Mel window length
    public let winLength: Int

    /// Number of mel features
    public let nMels: Int

    /// Number of mel context frames per output frame
    public let contextSize: Int

    /// Mel -> prediction subsampling
    public let subsampling: Int

    /// Number of right context output frames
    public let convDelay: Int

    // Model structure
    public let nUnits: Int
    public let nHeads: Int
    public let encNLayers: Int
    public let decNLayers: Int
    public let convKernelSize: Int
    public var headDim: Int { nUnits / nHeads }

    /// Mel frames per chunk
    public var melFrames: Int { (chunkSize - 1) * subsampling + 2 * contextSize + 1 }

    public var nFFT: Int {
        1 << (Int.bitWidth - (winLength - 1).leadingZeroBitCount)
    }
}

// MARK: - Recurrent State

public struct LSEENDState: ~Copyable {
    public var encRetKv: MLMultiArray
    public var encRetScale: MLMultiArray
    public var encConvCache: MLMultiArray
    public var cnnWindow: MLMultiArray
    public var decRetKv: MLMultiArray
    public var decRetScale: MLMultiArray

    public init(
        encRetKv: MLMultiArray,
        encRetScale: MLMultiArray,
        encConvCache: MLMultiArray,
        cnnWindow: MLMultiArray,
        decRetKv: MLMultiArray,
        decRetScale: MLMultiArray
    ) {
        self.encRetKv = encRetKv
        self.encRetScale = encRetScale
        self.encConvCache = encConvCache
        self.cnnWindow = cnnWindow
        self.decRetKv = decRetKv
        self.decRetScale = decRetScale
    }

    public init(from metadata: borrowing LSEENDMetadata) throws {
        let Lenc = NSNumber(value: metadata.encNLayers)
        let Ldec = NSNumber(value: metadata.decNLayers)
        let H = NSNumber(value: metadata.nHeads)
        let hd = NSNumber(value: metadata.headDim)
        let D = NSNumber(value: metadata.nUnits)
        let K = NSNumber(value: metadata.convKernelSize)
        let Kcnn = NSNumber(value: 2 * metadata.convDelay)
        let nSpk = NSNumber(value: metadata.maxNspks)

        func makeArray(shape: [NSNumber]) throws -> MLMultiArray {
            try ANEMemoryUtils.createAlignedArray(shape: shape, dataType: .float32)
        }

        self.init(
            encRetKv: try makeArray(shape: [Lenc, 1, H, hd, hd]),
            encRetScale: try makeArray(shape: [Lenc, 1]),
            encConvCache: try makeArray(shape: [Lenc, 1, K, D]),
            cnnWindow: try makeArray(shape: [1, D, Kcnn]),
            decRetKv: try makeArray(shape: [Ldec, nSpk, H, hd, hd]),
            decRetScale: try makeArray(shape: [Ldec, 1])
        )

        self.reset()
    }

    public func copy() throws -> LSEENDState {
        func clone(_ src: MLMultiArray) throws -> MLMultiArray {
            let dst = try ANEMemoryUtils.createAlignedArray(
                shape: src.shape, dataType: src.dataType
            )
            ANEMemoryUtils.strideAwareCopy(from: src, to: dst)
            return dst
        }
        return LSEENDState(
            encRetKv: try clone(encRetKv),
            encRetScale: try clone(encRetScale),
            encConvCache: try clone(encConvCache),
            cnnWindow: try clone(cnnWindow),
            decRetKv: try clone(decRetKv),
            decRetScale: try clone(decRetScale)
        )
    }

    public func copy(to dst: borrowing LSEENDState) {
        ANEMemoryUtils.strideAwareCopy(from: encRetKv, to: dst.encRetKv)
        ANEMemoryUtils.strideAwareCopy(from: encRetScale, to: dst.encRetScale)
        ANEMemoryUtils.strideAwareCopy(from: encConvCache, to: dst.encConvCache)
        ANEMemoryUtils.strideAwareCopy(from: cnnWindow, to: dst.cnnWindow)
        ANEMemoryUtils.strideAwareCopy(from: decRetKv, to: dst.decRetKv)
        ANEMemoryUtils.strideAwareCopy(from: decRetScale, to: dst.decRetScale)
    }

    public func reset() {
        func clear(_ buffer: MLMultiArray) {
            buffer.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
                guard let base = buf.baseAddress else { return }
                memset(base, 0, buf.count * MemoryLayout<Float>.stride)
            }
        }

        clear(encRetKv)
        clear(encRetScale)
        clear(encConvCache)
        clear(cnnWindow)
        clear(decRetKv)
        clear(decRetScale)
    }
}

// MARK: - Errors

public enum LSEENDError: Error, LocalizedError {
    case initializationFailed(String)
    case inferenceFailed(String)
    case invalidInputSize(String)
    case notInitialized
}
