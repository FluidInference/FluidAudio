import Foundation

internal struct Segment: Hashable {
    let start: Double
    let end: Double
}

internal struct SlidingWindow {
    var start: Double
    var duration: Double
    var step: Double

    func time(forFrame index: Int) -> Double {
        return start + Double(index) * step
    }

    func segment(forFrame index: Int) -> Segment {
        let s = time(forFrame: index)
        return Segment(start: s, end: s + duration)
    }
}

internal struct SlidingWindowFeature {
    var data: [[[Float]]]
    var slidingWindow: SlidingWindow
}
