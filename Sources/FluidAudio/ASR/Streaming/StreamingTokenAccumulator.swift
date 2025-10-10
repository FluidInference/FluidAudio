struct StreamingTokenAccumulator {
    /// Base capacity for the committed token buffer to avoid immediate growth.
    private static let defaultBufferCapacity: Int = 128

    private var buffer: CircularBuffer<Int>
    private var trimmedCount: Int = 0
    private let retainedCommittedWindow: Int

    init(retainedCommittedWindow: Int = 64) {
        self.retainedCommittedWindow = max(0, retainedCommittedWindow)
        let initialCapacity = max(Self.defaultBufferCapacity, self.retainedCommittedWindow * 2)
        self.buffer = CircularBuffer(initialCapacity: initialCapacity)
    }

    mutating func reset() {
        buffer.removeAll(keepingCapacity: false)
        trimmedCount = 0
    }

    mutating func append(_ tokens: [Int]) {
        guard !tokens.isEmpty else { return }
        buffer.append(contentsOf: tokens)
    }

    var tokens: [Int] {
        buffer.asArray()
    }

    var trimmedTotalCount: Int {
        trimmedCount
    }

    var totalCount: Int {
        trimmedCount + buffer.count
    }

    @discardableResult
    mutating func dropCommittedPrefixIfNeeded(totalCommitted: Int) -> Int {
        guard buffer.count > 0 else { return 0 }
        let committedInBuffer = max(0, min(buffer.count, totalCommitted - trimmedCount))
        guard committedInBuffer > retainedCommittedWindow else { return 0 }

        // Keep only the most recent committed tokens; deduplication logic needs a short suffix.
        let dropCount = committedInBuffer - retainedCommittedWindow
        buffer.dropFirst(dropCount)
        trimmedCount += dropCount
        return dropCount
    }
}
