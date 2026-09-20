import CoreML
import Foundation

extension MLMultiArray {
    /// Reset all elements in the array to the given value.
    func reset(to value: NSNumber) {
        resetData(to: value)
    }

    /// Fills every element with `value`.
    ///
    /// Contiguous storage fills in bulk: zero is one `memset` for every data type, other values
    /// fill through a typed pointer for float32, float64 and int32. Padded strides and other data
    /// types fill element by element, so nothing past the last element is ever written. `value` is
    /// compared as an `NSNumber`, so `-0.0` takes the zero path and lands as `+0.0`.
    func resetData(to value: NSNumber) {
        let elementSize = ANEMemoryUtils.getElementSize(for: dataType)
        let filled = withUnsafeMutableBytes { bytes, _ -> Bool in
            guard bytes.count == count * elementSize, let base = bytes.baseAddress else {
                return false
            }
            if value == 0 {
                memset(base, 0, bytes.count)
                return true
            }
            switch dataType {
            case .float32:
                bytes.bindMemory(to: Float.self).update(repeating: value.floatValue)
            case .float64:
                bytes.bindMemory(to: Double.self).update(repeating: value.doubleValue)
            case .int32:
                bytes.bindMemory(to: Int32.self).update(repeating: value.int32Value)
            default:
                return false
            }
            return true
        }
        if filled {
            return
        }
        for i in 0..<count {
            self[i] = value
        }
    }

    /// Copies every element from `source`.
    ///
    /// Identical contiguous layouts copy the storage in bulk; that copy is overlap-safe, so two
    /// views of one allocation may overlap. Any other pair copies element by element and assumes
    /// the two arrays do not share storage.
    func copyData(from source: MLMultiArray) {
        let elementSize = ANEMemoryUtils.getElementSize(for: dataType)
        if dataType == source.dataType, shape == source.shape, strides == source.strides {
            let copied = withUnsafeMutableBytes { destination, _ -> Bool in
                guard destination.count == count * elementSize else {
                    return false
                }
                source.withUnsafeBytes { destination.copyMemory(from: $0) }
                return true
            }
            if copied {
                return
            }
        }
        for i in 0..<count {
            self[i] = source[i]
        }
    }
}
