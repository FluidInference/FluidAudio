import CoreML

extension MLMultiArray {
    /// Reset all elements in the array to the given value.
    func reset(to value: NSNumber) {
        let count = self.count
        if self.dataType == .float32 {
            let ptr = self.dataPointer.bindMemory(to: Float.self, capacity: count)
            ptr.update(repeating: value.floatValue, count: count)
        } else if self.dataType == .int32 {
            let intPtr = self.dataPointer.bindMemory(to: Int32.self, capacity: count)
            intPtr.update(repeating: value.int32Value, count: count)
        }
    }
}
