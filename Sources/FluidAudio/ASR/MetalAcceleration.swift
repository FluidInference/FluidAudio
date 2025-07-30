import Metal
import MetalPerformanceShaders
import CoreML
import Accelerate
import os

/// GPU-accelerated operations using Metal Performance Shaders
@available(macOS 13.0, iOS 16.0, *)
public class MetalAcceleration {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "MetalAcceleration")
    
    /// Singleton instance for shared GPU resources
    public static let shared: MetalAcceleration? = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Logger(subsystem: "com.fluidinfluence.asr", category: "MetalAcceleration")
                .error("Failed to create Metal device")
            return nil
        }
        return MetalAcceleration(device: device)
    }()
    
    private init?(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            logger.error("Failed to create command queue")
            return nil
        }
        self.commandQueue = queue
    }
    
    /// Convert MLMultiArray to Metal buffer (zero-copy when possible)
    private func createMetalBuffer(from array: MLMultiArray) -> MTLBuffer? {
        guard array.dataType == .float32 else {
            logger.error("Only float32 arrays supported")
            return nil
        }
        
        let dataPointer = array.dataPointer
        let byteLength = array.count * MemoryLayout<Float>.stride
        
        // Try to create buffer without copying (using the existing memory)
        if let buffer = device.makeBuffer(
            bytesNoCopy: dataPointer,
            length: byteLength,
            options: [.storageModeShared],
            deallocator: nil
        ) {
            return buffer
        }
        
        // Fallback: copy data
        return device.makeBuffer(bytes: dataPointer, length: byteLength, options: [.storageModeShared])
    }
    
    /// GPU-accelerated softmax using MPS
    public func softmax(_ input: MLMultiArray, temperature: Float = 1.0) -> MLMultiArray? {
        guard let inputBuffer = createMetalBuffer(from: input),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            logger.error("Failed to create Metal resources for softmax")
            return nil
        }
        
        // Create output array
        guard let output = try? MLMultiArray(shape: input.shape, dataType: .float32),
              let outputBuffer = createMetalBuffer(from: output) else {
            return nil
        }
        
        // Create MPS softmax
        let softmax = MPSCNNSoftMax(device: device)
        
        // Create temporary image descriptors for MPS
        let width = input.count
        let imageDescriptor = MPSImageDescriptor(
            channelFormat: .float32,
            width: width,
            height: 1,
            featureChannels: 1
        )
        
        let inputImage = MPSImage(device: device, imageDescriptor: imageDescriptor)
        let outputImage = MPSImage(device: device, imageDescriptor: imageDescriptor)
        
        // Copy data to images
        inputImage.texture.replace(
            region: MTLRegionMake2D(0, 0, width, 1),
            mipmapLevel: 0,
            withBytes: inputBuffer.contents(),
            bytesPerRow: width * MemoryLayout<Float>.stride
        )
        
        // Apply temperature scaling if needed
        if temperature != 1.0 {
            let scaledBuffer = inputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
            vDSP_vsdiv(scaledBuffer, 1, [temperature], scaledBuffer, 1, vDSP_Length(input.count))
        }
        
        // Encode softmax operation
        softmax.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Copy result back
        outputImage.texture.getBytes(
            outputBuffer.contents(),
            bytesPerRow: width * MemoryLayout<Float>.stride,
            from: MTLRegionMake2D(0, 0, width, 1),
            mipmapLevel: 0
        )
        
        // Copy to output MLMultiArray
        memcpy(output.dataPointer, outputBuffer.contents(), output.count * MemoryLayout<Float>.stride)
        
        return output
    }
    
    /// GPU-accelerated argmax
    public func argmax(_ input: MLMultiArray) -> (index: Int, value: Float)? {
        guard input.dataType == .float32 else {
            logger.error("Only float32 arrays supported for argmax")
            return nil
        }
        
        // For small arrays, CPU might be faster
        if input.count < 1024 {
            return argmaxCPU(input)
        }
        
        guard let inputBuffer = createMetalBuffer(from: input),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return argmaxCPU(input)
        }
        
        // Use MPS reduction for finding max
        let reduction = MPSMatrixFindTopK(
            device: device,
            numberOfTopKValues: 1
        )
        
        let matrixDescriptor = MPSMatrixDescriptor(
            rows: 1,
            columns: input.count,
            rowBytes: input.count * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        
        let inputMatrix = MPSMatrix(buffer: inputBuffer, descriptor: matrixDescriptor)
        
        // Create output buffers
        guard let valueBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared),
              let indexBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            return argmaxCPU(input)
        }
        
        let valueMatrix = MPSMatrix(
            buffer: valueBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: 1,
                columns: 1,
                rowBytes: MemoryLayout<Float>.stride,
                dataType: .float32
            )
        )
        
        let indexMatrix = MPSMatrix(
            buffer: indexBuffer,
            descriptor: MPSMatrixDescriptor(
                rows: 1,
                columns: 1,
                rowBytes: MemoryLayout<UInt32>.stride,
                dataType: .uInt32
            )
        )
        
        reduction.encode(
            commandBuffer: commandBuffer,
            inputMatrix: inputMatrix,
            resultIndexMatrix: indexMatrix,
            resultValueMatrix: valueMatrix
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let index = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        let value = valueBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
        
        return (index: Int(index), value: value)
    }
    
    /// CPU fallback for argmax
    private func argmaxCPU(_ input: MLMultiArray) -> (index: Int, value: Float)? {
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        
        input.dataPointer.withMemoryRebound(to: Float.self, capacity: input.count) { ptr in
            vDSP_maxvi(ptr, 1, &maxValue, &maxIndex, vDSP_Length(input.count))
        }
        
        return (index: Int(maxIndex), value: maxValue)
    }
}

/// Extension for MLMultiArray to Metal buffer conversions
@available(macOS 13.0, iOS 16.0, *)
extension MLMultiArray {
    /// Create a Metal buffer sharing memory with this MLMultiArray (zero-copy when possible)
    func makeMetalBuffer(device: MTLDevice) -> MTLBuffer? {
        guard dataType == .float32 else { return nil }
        
        let byteLength = count * MemoryLayout<Float>.stride
        
        // Try zero-copy buffer creation
        if let buffer = device.makeBuffer(
            bytesNoCopy: dataPointer,
            length: byteLength,
            options: [.storageModeShared],
            deallocator: nil
        ) {
            return buffer
        }
        
        // Fallback to copying
        return device.makeBuffer(bytes: dataPointer, length: byteLength, options: [.storageModeShared])
    }
}