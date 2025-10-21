import Foundation

/// A fixed-capacity ring buffer with O(1) append and iteration for sendable elements.
struct FixedCapacityRingBuffer<Element: Sendable>: Sequence, Sendable {

    private var storage: [Element?]
    private let capacity: Int
    private var head: Int = 0
    private var elementCount: Int = 0

    init(capacity: Int) {
        self.capacity = Swift.max(0, capacity)
        if self.capacity > 0 {
            self.storage = Array(repeating: nil, count: self.capacity)
        } else {
            self.storage = []
        }
    }

    var isEmpty: Bool { elementCount == 0 }
    var count: Int { elementCount }

    mutating func append(_ element: Element) {
        guard capacity > 0 else { return }
        let tailIndex = (head + elementCount) % capacity
        storage[tailIndex] = element
        if elementCount == capacity {
            head = (head + 1) % capacity
        } else {
            elementCount += 1
        }
    }

    mutating func removeAll() {
        if capacity > 0 {
            for index in 0..<capacity {
                storage[index] = nil
            }
        }
        head = 0
        elementCount = 0
    }

    mutating func assign<S: Sequence>(from sequence: S) where S.Element == Element {
        removeAll()
        for element in sequence {
            append(element)
        }
    }

    func makeIterator() -> Iterator {
        Iterator(buffer: self)
    }

    struct Iterator: IteratorProtocol {
        private let buffer: FixedCapacityRingBuffer
        private var offset: Int = 0

        init(buffer: FixedCapacityRingBuffer) {
            self.buffer = buffer
        }

        mutating func next() -> Element? {
            guard offset < buffer.elementCount else { return nil }
            let index = (buffer.head + offset) % buffer.capacity
            offset += 1
            return buffer.storage[index]!
        }
    }
}
