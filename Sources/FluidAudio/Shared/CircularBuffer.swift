import Foundation

/// A fixed-growth circular buffer with amortized O(1) append and drop operations.
/// Storage is backed by a contiguous array and automatically expands as needed.
struct CircularBuffer<Element: AdditiveArithmetic & ExpressibleByIntegerLiteral & Sendable> {
    private var storage: [Element]
    private var capacity: Int
    private var head: Int
    private var tail: Int
    private(set) var count: Int

    init(initialCapacity: Int = 0) {
        let capacity = Self.normalizedCapacity(initialCapacity)
        self.capacity = capacity
        let zero = Element.zero
        self.storage = capacity > 0 ? Array(repeating: zero, count: capacity) : []
        self.head = 0
        self.tail = 0
        self.count = 0
    }

    var isEmpty: Bool {
        count == 0
    }

    mutating func removeAll(keepingCapacity: Bool) {
        count = 0
        head = 0
        tail = 0

        if !keepingCapacity {
            storage.removeAll(keepingCapacity: false)
            capacity = 0
        }
    }

    mutating func append(_ element: Element) {
        ensureCapacity(for: 1)
        storage[tail] = element
        tail = incrementedIndex(from: tail)
        count += 1
    }

    mutating func append(contentsOf elements: [Element]) {
        guard !elements.isEmpty else { return }
        ensureCapacity(for: elements.count)
        for element in elements {
            storage[tail] = element
            tail = incrementedIndex(from: tail)
            count += 1
        }
    }

    func copyPrefix(_ length: Int) -> [Element] {
        guard length > 0 else { return [] }
        precondition(length <= count, "Requested prefix exceeds buffer count.")
        return copyRangeInternal(start: 0, length: length)
    }

    func copyRange(_ range: Range<Int>) -> [Element] {
        precondition(range.lowerBound >= 0, "Range lower bound must be non-negative.")
        precondition(range.upperBound <= count, "Range upper bound exceeds buffer count.")
        return copyRangeInternal(start: range.lowerBound, length: range.count)
    }

    mutating func popFirst(_ length: Int) -> [Element] {
        let result = copyPrefix(length)
        dropFirst(length)
        return result
    }

    mutating func dropFirst(_ length: Int) {
        guard length > 0 else { return }
        precondition(length <= count, "Cannot drop more elements than currently stored.")
        head = incrementedIndex(from: head, by: length)
        count -= length
        if count == 0 {
            head = 0
            tail = 0
        }
    }

    subscript(index: Int) -> Element {
        precondition(index >= 0 && index < count, "Index out of bounds.")
        let physical = incrementedIndex(from: head, by: index)
        return storage[physical]
    }

    func asArray() -> [Element] {
        guard count > 0 else { return [] }
        return copyRangeInternal(start: 0, length: count)
    }

    private mutating func ensureCapacity(for additionalElements: Int) {
        let required = count + additionalElements
        guard required > capacity else { return }

        var newCapacity = max(capacity > 0 ? capacity * 2 : 1, required)
        if !newCapacity.isMultiple(of: 2) {
            newCapacity = Self.nextPowerOfTwo(newCapacity)
        }

        var newStorage = Array(repeating: Element.zero, count: newCapacity)
        if count > 0 {
            if head < tail {
                let rangeCount = tail - head
                for index in 0..<rangeCount {
                    newStorage[index] = storage[head + index]
                }
            } else {
                let leadingCount = capacity - head
                for index in 0..<leadingCount {
                    newStorage[index] = storage[head + index]
                }
                for index in 0..<tail {
                    newStorage[leadingCount + index] = storage[index]
                }
            }
        }
        head = 0
        tail = count
        storage = newStorage
        capacity = newCapacity
    }

    private func copyRangeInternal(start: Int, length: Int) -> [Element] {
        guard length > 0 else { return [] }
        var result: [Element] = []
        result.reserveCapacity(length)

        var remaining = length
        var currentIndex = incrementedIndex(from: head, by: start)
        while remaining > 0 {
            let contiguousCount = min(remaining, capacity - currentIndex)
            result.append(contentsOf: storage[currentIndex..<(currentIndex + contiguousCount)])
            remaining -= contiguousCount
            currentIndex = (currentIndex + contiguousCount) % capacity
        }

        return result
    }

    private func incrementedIndex(from index: Int, by offset: Int = 1) -> Int {
        guard capacity > 0 else { return 0 }
        let value = index + offset
        if value >= capacity {
            return value % capacity
        }
        return value
    }

    private static func normalizedCapacity(_ requested: Int) -> Int {
        guard requested > 0 else { return 0 }
        if requested.isMultiple(of: 2) {
            return requested
        }
        return nextPowerOfTwo(requested)
    }

    private static func nextPowerOfTwo(_ value: Int) -> Int {
        var power = 1
        while power < value {
            power <<= 1
        }
        return power
    }
}
