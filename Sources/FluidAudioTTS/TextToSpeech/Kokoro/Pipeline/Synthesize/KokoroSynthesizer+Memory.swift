import Accelerate
import CoreML
import FluidAudio
import Foundation

extension KokoroSynthesizer {
    struct MultiArrayKey: Hashable {
        let dataTypeRawValue: Int
        let shape: [Int]

        init(dataType: MLMultiArrayDataType, shape: [NSNumber]) {
            dataTypeRawValue = dataType.rawValue
            self.shape = shape.map { $0.intValue }
        }

        func hash(into hasher: inout Hasher) {
            hasher.combine(dataTypeRawValue)
            for dimension in shape {
                hasher.combine(dimension)
            }
        }
    }

    actor MultiArrayPool {
        private var storage: [MultiArrayKey: [MLMultiArray]] = [:]

        func rent(
            shape: [NSNumber],
            dataType: MLMultiArrayDataType,
            zeroFill: Bool
        ) async throws -> MLMultiArray {
            let key = MultiArrayKey(dataType: dataType, shape: shape)
            let array: MLMultiArray
            if var cached = storage[key], let candidate = cached.popLast() {
                storage[key] = cached
                array = candidate
            } else {
                array = try MLMultiArray(shape: shape, dataType: dataType)
            }

            if zeroFill {
                zero(array)
            }
            return array
        }

        func recycle(_ array: MLMultiArray, zeroFill: Bool) {
            if zeroFill {
                zero(array)
            }
            let key = MultiArrayKey(dataType: array.dataType, shape: array.shape)
            storage[key, default: []].append(array)
        }

        func preallocate(
            shape: [NSNumber],
            dataType: MLMultiArrayDataType,
            count: Int,
            zeroFill: Bool
        ) async throws {
            guard count > 0 else { return }
            let key = MultiArrayKey(dataType: dataType, shape: shape)
            var pool = storage[key] ?? []
            if pool.count >= count {
                storage[key] = pool
                return
            }

            let additional = count - pool.count
            pool.reserveCapacity(count)
            for _ in 0..<additional {
                let array = try MLMultiArray(shape: shape, dataType: dataType)
                if zeroFill {
                    zero(array)
                }
                pool.append(array)
            }
            storage[key] = pool
        }

        private func zero(_ array: MLMultiArray) {
            let elementCount = array.count
            guard elementCount > 0 else { return }

            switch array.dataType {
            case .int32:
                let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: elementCount)
                pointer.initialize(repeating: 0, count: elementCount)
            case .float32:
                let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: elementCount)
                vDSP_vclr(pointer, 1, vDSP_Length(elementCount))
            case .double:
                let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: elementCount)
                vDSP_vclrD(pointer, 1, vDSP_Length(elementCount))
            case .float16:
                let pointer = array.dataPointer.bindMemory(to: UInt16.self, capacity: elementCount)
                pointer.initialize(repeating: 0, count: elementCount)
            #if swift(>=6.2)
            case .int8:
                array.dataPointer.initializeMemory(as: Int8.self, repeating: 0, count: elementCount)
            @unknown default:
                memset(array.dataPointer, 0, elementCount * MemoryLayout<Float>.stride)
            #else
            default:
                memset(array.dataPointer, 0, elementCount * MemoryLayout<Float>.stride)
            #endif
            }
        }
    }
}
