import Accelerate
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
struct VBxClustering {
    private let config: OfflineDiarizerConfig
    private let pldaTransform: PLDATransform
    private let logger = AppLogger(category: "OfflineVBx")

    init(config: OfflineDiarizerConfig, pldaTransform: PLDATransform) {
        self.config = config
        self.pldaTransform = pldaTransform
    }

    func refine(
        rhoFeatures: [[Double]],
        initialClusters: [Int]
    ) -> VBxOutput {
        guard !rhoFeatures.isEmpty else {
            return VBxOutput(
                gamma: [],
                pi: [],
                hardClusters: [],
                centroids: [],
                numClusters: 0,
                elbos: []
            )
        }

        let frameCount = rhoFeatures.count
        guard let dimension = rhoFeatures.first?.count, dimension > 0 else {
            logger.error("VBx received empty feature vectors")
            return VBxOutput(
                gamma: [],
                pi: [],
                hardClusters: [],
                centroids: [],
                numClusters: 0,
                elbos: []
            )
        }

        var phi = pldaTransform.phiParameters
        if phi.count != dimension {
            logger.warning(
                "PLDA psi dimension (\(phi.count)) mismatches rho dimension (\(dimension)); falling back to identity")
            phi = Array(repeating: 1.0, count: dimension)
        }

        let speakerCount = max(1, Set(initialClusters).count)
        let histogram = initialClusters.reduce(into: [:]) { partialResult, value in
            partialResult[value, default: 0] += 1
        }
        print("[VBx] Initial clusters count: \(speakerCount) histogram: \(histogram)")

        var featureBuffer = [Double](repeating: 0, count: frameCount * dimension)
        for (index, frame) in rhoFeatures.enumerated() {
            featureBuffer.replaceSubrange(
                index * dimension..<(index + 1) * dimension,
                with: frame
            )
        }

        var initialGamma = [Double](repeating: 0, count: frameCount * speakerCount)
        if !initialClusters.isEmpty {
            for (index, cluster) in initialClusters.enumerated() {
                let speaker = max(0, min(cluster, speakerCount - 1))
                initialGamma[index * speakerCount + speaker] = 1.0
            }
        } else {
            let uniform = 1.0 / Double(speakerCount)
            for index in 0..<frameCount {
                for speaker in 0..<speakerCount {
                    initialGamma[index * speakerCount + speaker] = uniform
                }
            }
        }

        let result = runVBx(
            features: featureBuffer,
            frameCount: frameCount,
            dimension: dimension,
            phi: phi,
            initialGamma: initialGamma,
            speakerCount: speakerCount,
            maxIterations: config.vbx.maxIterations,
            epsilon: config.vbx.convergenceTolerance,
            Fa: config.clustering.warmStartFa,
            Fb: config.clustering.warmStartFb,
            initSmoothing: 7.0
        )

        let gammaMatrix = reshapeGamma(result.gamma, frameCount: frameCount, speakerCount: speakerCount)
        let hardAssignments = gammaMatrix.map { row -> Int in
            row.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }

        let elboHistory = result.elbos
        print("[VBx] piOut stats: max=\(result.pi.max() ?? 0), min=\(result.pi.min() ?? 0), count=\(result.pi.count)")

        return VBxOutput(
            gamma: gammaMatrix,
            pi: result.pi,
            hardClusters: [hardAssignments],
            centroids: [],
            numClusters: speakerCount,
            elbos: elboHistory
        )
    }

    private func runVBx(
        features: [Double],
        frameCount: Int,
        dimension: Int,
        phi: [Double],
        initialGamma: [Double],
        speakerCount: Int,
        maxIterations: Int,
        epsilon: Double,
        Fa: Double,
        Fb: Double,
        initSmoothing: Double
    ) -> (gamma: [Double], pi: [Double], elbos: [Double]) {
        var gamma = initialGamma

        if initSmoothing >= 0.0 {
            var rowBuffer = [Double](repeating: 0, count: speakerCount)
            for t in 0..<frameCount {
                let start = t * speakerCount
                var maxValue = -Double.greatestFiniteMagnitude
                for s in 0..<speakerCount {
                    let value = gamma[start + s] * initSmoothing
                    rowBuffer[s] = value
                    if value > maxValue {
                        maxValue = value
                    }
                }
                var sumExp = 0.0
                for s in 0..<speakerCount {
                    let expValue = exp(rowBuffer[s] - maxValue)
                    rowBuffer[s] = expValue
                    sumExp += expValue
                }
                if sumExp <= 0 || !sumExp.isFinite {
                    let uniform = 1.0 / Double(speakerCount)
                    for s in 0..<speakerCount {
                        gamma[start + s] = uniform
                    }
                } else {
                    let invSum = 1.0 / sumExp
                    for s in 0..<speakerCount {
                        gamma[start + s] = rowBuffer[s] * invSum
                    }
                }
            }
        }

        for t in 0..<frameCount {
            let start = t * speakerCount
            var sum = 0.0
            for s in 0..<speakerCount {
                sum += gamma[start + s]
            }
            if sum <= 0 || !sum.isFinite {
                let uniform = 1.0 / Double(speakerCount)
                for s in 0..<speakerCount {
                    gamma[start + s] = uniform
                }
            } else {
                let inv = 1.0 / sum
                for s in 0..<speakerCount {
                    gamma[start + s] *= inv
                }
            }
        }

        var pi = [Double](repeating: 1.0 / Double(speakerCount), count: speakerCount)

        let phiClamped = phi.map { max($0, 1e-12) }
        let sqrtPhi = phiClamped.map { sqrt($0) }

        var rho = [Double](repeating: 0, count: features.count)
        for t in 0..<frameCount {
            for d in 0..<dimension {
                rho[t * dimension + d] = features[t * dimension + d] * sqrtPhi[d]
            }
        }

        var G = [Double](repeating: 0, count: frameCount)
        let logConstant = Double(dimension) * log(2.0 * Double.pi)
        for t in 0..<frameCount {
            var sumSq = 0.0
            for d in 0..<dimension {
                let value = features[t * dimension + d]
                sumSq += value * value
            }
            G[t] = -0.5 * (sumSq + logConstant)
        }

        let ratio = Fa / Fb
        var invL = [Double](repeating: 0, count: speakerCount * dimension)
        var alpha = [Double](repeating: 0, count: speakerCount * dimension)
        var temp = [Double](repeating: 0, count: speakerCount * dimension)
        var phiTerms = [Double](repeating: 0, count: speakerCount)
        var gammaSum = [Double](repeating: 0, count: speakerCount)
        var logP = [Double](repeating: 0, count: frameCount * speakerCount)
        var rowBuffer = [Double](repeating: 0, count: speakerCount)
        var elbos = [Double](repeating: 0, count: max(maxIterations, 1))

        var previousElbo = -Double.greatestFiniteMagnitude
        var iterations = 0

        for iteration in 0..<maxIterations {
            iterations = iteration + 1

            gammaSum.replaceSubrange(0..<speakerCount, with: repeatElement(0, count: speakerCount))
            for t in 0..<frameCount {
                let start = t * speakerCount
                for s in 0..<speakerCount {
                    gammaSum[s] += gamma[start + s]
                }
            }

            for s in 0..<speakerCount {
                let weight = ratio * gammaSum[s]
                for d in 0..<dimension {
                    let idx = s * dimension + d
                    let denom = 1.0 + weight * phiClamped[d]
                    invL[idx] = 1.0 / max(denom, 1e-12)
                }
            }

            gamma.withUnsafeBufferPointer { gammaPtr in
                rho.withUnsafeBufferPointer { rhoPtr in
                    temp.withUnsafeMutableBufferPointer { tempPtr in
                        cblas_dgemm(
                            CblasRowMajor,
                            CblasTrans,
                            CblasNoTrans,
                            Int32(speakerCount),
                            Int32(dimension),
                            Int32(frameCount),
                            1.0,
                            gammaPtr.baseAddress!,
                            Int32(speakerCount),
                            rhoPtr.baseAddress!,
                            Int32(dimension),
                            0.0,
                            tempPtr.baseAddress!,
                            Int32(dimension)
                        )
                    }
                }
            }

            for idx in 0..<temp.count {
                alpha[idx] = ratio * invL[idx] * temp[idx]
            }

            for s in 0..<speakerCount {
                var sum = 0.0
                for d in 0..<dimension {
                    let idx = s * dimension + d
                    let inv = invL[idx]
                    let a = alpha[idx]
                    sum += (inv + a * a) * phiClamped[d]
                }
                phiTerms[s] = sum
            }

            rho.withUnsafeBufferPointer { rhoPtr in
                alpha.withUnsafeBufferPointer { alphaPtr in
                    logP.withUnsafeMutableBufferPointer { logPtr in
                        cblas_dgemm(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasTrans,
                            Int32(frameCount),
                            Int32(speakerCount),
                            Int32(dimension),
                            1.0,
                            rhoPtr.baseAddress!,
                            Int32(dimension),
                            alphaPtr.baseAddress!,
                            Int32(dimension),
                            0.0,
                            logPtr.baseAddress!,
                            Int32(speakerCount)
                        )
                    }
                }
            }

            for t in 0..<frameCount {
                let g = G[t]
                let rowStart = t * speakerCount
                for s in 0..<speakerCount {
                    logP[rowStart + s] = Fa * (logP[rowStart + s] - 0.5 * phiTerms[s] + g)
                }
            }

            let eps = 1e-8
            var logLikelihood = 0.0
            for t in 0..<frameCount {
                let rowStart = t * speakerCount
                var rowMax = -Double.greatestFiniteMagnitude
                for s in 0..<speakerCount {
                    let value = logP[rowStart + s] + log(max(pi[s], eps))
                    rowBuffer[s] = value
                    if value > rowMax {
                        rowMax = value
                    }
                }

                var sumExp = 0.0
                for s in 0..<speakerCount {
                    let expValue = exp(rowBuffer[s] - rowMax)
                    rowBuffer[s] = expValue
                    sumExp += expValue
                }

                if sumExp <= 0.0 || !sumExp.isFinite {
                    let uniform = 1.0 / Double(speakerCount)
                    for s in 0..<speakerCount {
                        gamma[rowStart + s] = uniform
                    }
                    logLikelihood += rowMax
                } else {
                    let invSum = 1.0 / sumExp
                    for s in 0..<speakerCount {
                        gamma[rowStart + s] = rowBuffer[s] * invSum
                    }
                    logLikelihood += rowMax + log(sumExp)
                }
            }

            pi = [Double](repeating: 0, count: speakerCount)
            for t in 0..<frameCount {
                let rowStart = t * speakerCount
                for s in 0..<speakerCount {
                    pi[s] += gamma[rowStart + s]
                }
            }
            let piSum = pi.reduce(0, +)
            if piSum > 0 && piSum.isFinite {
                let inv = 1.0 / piSum
                for s in 0..<speakerCount {
                    pi[s] *= inv
                }
            } else {
                let uniform = 1.0 / Double(speakerCount)
                pi = [Double](repeating: uniform, count: speakerCount)
            }

            var elbo = logLikelihood
            for idx in 0..<invL.count {
                let inv = max(invL[idx], 1e-12)
                elbo += Fb * 0.5 * (log(inv) - inv - alpha[idx] * alpha[idx] + 1.0)
            }

            if iteration < elbos.count {
                elbos[iteration] = elbo
            }

            if iteration > 0 {
                let improvement = elbo - previousElbo
                if abs(improvement) < epsilon {
                    previousElbo = elbo
                    break
                }
            }
            previousElbo = elbo
        }

        return (gamma, pi, Array(elbos.prefix(iterations)))
    }

    private func reshapeGamma(_ buffer: [Double], frameCount: Int, speakerCount: Int) -> [[Double]] {
        var result: [[Double]] = []
        result.reserveCapacity(frameCount)
        for frame in 0..<frameCount {
            let start = frame * speakerCount
            let row = Array(buffer[start..<(start + speakerCount)])
            result.append(row)
        }
        return result
    }
}
