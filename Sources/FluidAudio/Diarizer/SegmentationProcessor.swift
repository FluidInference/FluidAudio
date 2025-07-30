import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct SegmentationProcessor {

    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "Segmentation")

    func getSegments(
        audioChunk: ArraySlice<Float>, segmentationModel: MLModel, chunkSize: Int = 160_000
    ) throws -> [[[Float]]] {

        let audioArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: chunkSize)],
            dataType: .float32
        )
        var offset = 0
        for sample in audioChunk.prefix(chunkSize) {
            audioArray[offset] = NSNumber(value: sample)
            offset &+= 1
        }

        let input = try MLDictionaryFeatureProvider(dictionary: ["audio": audioArray])

        let output = try segmentationModel.prediction(from: input)

        guard let segmentOutput = output.featureValue(for: "segments")?.multiArrayValue else {
            throw DiarizerError.processingFailed("Missing segments output from segmentation model")
        }

        let frames = segmentOutput.shape[1].intValue
        let combinations = segmentOutput.shape[2].intValue

        var segments = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: combinations), count: frames),
            count: 1)

        for f in 0..<frames {
            for c in 0..<combinations {
                let index = f * combinations + c
                segments[0][f][c] = segmentOutput[index].floatValue
            }
        }

        return powersetConversion(segments)
    }

    private func powersetConversion(_ segments: [[[Float]]]) -> [[[Float]]] {
        let powerset: [[Int]] = [
            [],
            [0],
            [1],
            [2],
            [0, 1],
            [0, 2],
            [1, 2],
        ]

        let batchSize = segments.count
        let numFrames = segments[0].count
        let numSpeakers = 3

        var binarized = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: numSpeakers),
                count: numFrames
            ),
            count: batchSize
        )

        for b in 0..<batchSize {
            for f in 0..<numFrames {
                let frame = segments[b][f]

                guard let bestIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) else {
                    continue
                }

                for speaker in powerset[bestIdx] {
                    binarized[b][f][speaker] = 1.0
                }
            }
        }

        return binarized
    }

    func createSlidingWindowFeature(
        binarizedSegments: [[[Float]]], chunkOffset: Double = 0.0
    ) -> SlidingWindowFeature {
        let slidingWindow = SlidingWindow(
            start: chunkOffset,
            duration: 0.0619375,
            step: 0.016875
        )

        return SlidingWindowFeature(
            data: binarizedSegments,
            slidingWindow: slidingWindow
        )
    }

    func getAnnotation(
        annotation: inout [Segment: Int],
        binarizedSegments: [[[Float]]],
        slidingWindow: SlidingWindow
    ) {
        let segmentation = binarizedSegments[0]
        let numFrames = segmentation.count

        var frameSpeakers: [Int] = []
        for frame in segmentation {
            if let maxIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) {
                frameSpeakers.append(maxIdx)
            } else {
                frameSpeakers.append(0)
            }
        }

        var currentSpeaker = frameSpeakers[0]
        var startFrame = 0

        for i in 1..<numFrames {
            if frameSpeakers[i] != currentSpeaker {
                let startTime = slidingWindow.time(forFrame: startFrame)
                let endTime = slidingWindow.time(forFrame: i)

                let segment = Segment(start: startTime, end: endTime)
                annotation[segment] = currentSpeaker
                currentSpeaker = frameSpeakers[i]
                startFrame = i
            }
        }

        let finalStart = slidingWindow.time(forFrame: startFrame)
        let finalEnd = slidingWindow.segment(forFrame: numFrames - 1).end
        let finalSegment = Segment(start: finalStart, end: finalEnd)
        annotation[finalSegment] = currentSpeaker
    }
}
