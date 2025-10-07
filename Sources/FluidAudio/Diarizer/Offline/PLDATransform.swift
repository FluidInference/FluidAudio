import Accelerate
import CoreML
import Foundation

@available(macOS 13.0, iOS 16.0, *)
public struct PLDATransform {
    private let pldaRhoModel: MLModel
    private let psi: [Double]
    private let memoryOptimizer = ANEMemoryOptimizer()

    private let embeddingDimension = 256
    private let rhoDimension = 128
    private let maxBatchSize = 32

    public init(pldaRhoModel: MLModel, psi: [Double]) {
        self.pldaRhoModel = pldaRhoModel
        self.psi = psi
    }

    public var phiParameters: [Double] { psi }

    /// Transform a sequence of 256-dimensional embeddings into 128-dimensional
    /// PLDA-space rho features using the Core ML model exported from Pyannote.
    public func transform(_ embeddings: [[Float]]) throws -> [[Double]] {
        guard !embeddings.isEmpty else { return [] }

        for embedding in embeddings {
            guard embedding.count == embeddingDimension else {
                throw OfflineDiarizationError.invalidConfiguration(
                    "Expected \(embeddingDimension)-dim embeddings, got \(embedding.count)"
                )
            }
        }

        var results: [[Double]] = []
        results.reserveCapacity(embeddings.count)

        var startIndex = 0
        while startIndex < embeddings.count {
            let endIndex = min(startIndex + maxBatchSize, embeddings.count)
            let batch = embeddings[startIndex..<endIndex]
            let batchResults = try transformBatch(batch)
            results.append(contentsOf: batchResults)
            startIndex = endIndex
        }

        return results
    }

    /// Convenience wrapper for single-embedding transform.
    public func transform(_ embedding: [Float]) throws -> [Double] {
        guard embedding.count == embeddingDimension else {
            throw OfflineDiarizationError.invalidConfiguration(
                "Expected \(embeddingDimension)-dim embedding, got \(embedding.count)"
            )
        }
        let transformed = try transform([embedding])
        return transformed.first ?? []
    }

    /// Cosine similarity score between two rho vectors.
    public func score(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhoDimension, rhs.count == rhoDimension else {
            return 0
        }

        var dot: Double = 0
        var normLhs: Double = 0
        var normRhs: Double = 0

        lhs.withUnsafeBufferPointer { lhsPointer in
            rhs.withUnsafeBufferPointer { rhsPointer in
                vDSP_dotprD(
                    lhsPointer.baseAddress!,
                    1,
                    rhsPointer.baseAddress!,
                    1,
                    &dot,
                    vDSP_Length(rhoDimension)
                )
                vDSP_dotprD(
                    lhsPointer.baseAddress!,
                    1,
                    lhsPointer.baseAddress!,
                    1,
                    &normLhs,
                    vDSP_Length(rhoDimension)
                )
                vDSP_dotprD(
                    rhsPointer.baseAddress!,
                    1,
                    rhsPointer.baseAddress!,
                    1,
                    &normRhs,
                    vDSP_Length(rhoDimension)
                )
            }
        }

        let magnitude = sqrt(normLhs) * sqrt(normRhs)
        if magnitude <= 0 {
            return 0
        }

        return dot / magnitude
    }

    private func transformBatch(_ embeddings: ArraySlice<[Float]>) throws -> [[Double]] {
        guard !embeddings.isEmpty else { return [] }
        guard embeddings.count <= maxBatchSize else {
            throw OfflineDiarizationError.invalidBatchSize(
                "PldaRho batch size must be <= \(maxBatchSize), got \(embeddings.count)"
            )
        }

        let shape: [NSNumber] = [NSNumber(value: embeddings.count), NSNumber(value: embeddingDimension)]
        let inputArray = try memoryOptimizer.createAlignedArray(shape: shape, dataType: .float32)
        let pointer = inputArray.dataPointer.assumingMemoryBound(to: Float.self)

        for (batchIndex, embedding) in embeddings.enumerated() {
            let base = batchIndex * embeddingDimension
            embedding.withUnsafeBufferPointer { buffer in
                vDSP_mmov(
                    buffer.baseAddress!,
                    pointer.advanced(by: base),
                    vDSP_Length(embeddingDimension),
                    1,
                    vDSP_Length(embeddingDimension),
                    1
                )
            }
        }

        let provider = ZeroCopyDiarizerFeatureProvider(
            features: ["embeddings": MLFeatureValue(multiArray: inputArray)]
        )
        let options = MLPredictionOptions()

        if #available(macOS 14.0, iOS 17.0, *) {
            inputArray.prefetchToNeuralEngine()
        }

        let output = try pldaRhoModel.prediction(from: provider, options: options)

        guard let rhoArray = output.featureValue(for: "rho")?.multiArrayValue else {
            throw OfflineDiarizationError.processingFailed("PldaRho model did not produce rho output")
        }

        let rhoPointer = rhoArray.dataPointer.assumingMemoryBound(to: Float.self)
        var results: [[Double]] = []
        results.reserveCapacity(embeddings.count)

        for batchIndex in 0..<embeddings.count {
            var rho = [Double](repeating: 0, count: rhoDimension)
            let base = batchIndex * rhoDimension
            for d in 0..<rhoDimension {
                rho[d] = Double(rhoPointer[base + d])
            }
            results.append(rho)
        }

        return results
    }
}
