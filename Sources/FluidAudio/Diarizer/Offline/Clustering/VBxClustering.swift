import Accelerate
import Foundation
import OSLog
import os.signpost

/// Variational Bayes clustering (VBx) for speaker diarization.
///
/// This implementation is based on the VBx algorithm from BUT Speech@FIT
/// (Brno University of Technology Speech@FIT group).
///
/// Reference:
///   - Original paper: "Improved Speaker Diarization Using a Deep Learning-based Approach"
///   - GitHub repository: https://github.com/BUTSpeechFIT/VBx
///   - License: Apache License 2.0
///   - Copyright 2021-2024 BUT Speech@FIT
///
/// The algorithm uses variational inference to cluster speaker embeddings with:
/// - Probabilistic Linear Discriminant Analysis (PLDA) transformation
/// - Expectation-Maximization (EM) iterations with convergence monitoring
/// - Evidence Lower Bound (ELBO) tracking for convergence
/// - Speaker mixture weight estimation (pi parameters)
///
/// The implementation includes warm-start initialization from initial hard cluster
/// assignments and supports PLDA whitening transformation of input features.
@available(macOS 14.0, iOS 17.0, *)
struct VBxClustering {
    private let config: OfflineDiarizerConfig
    private let pldaTransform: PLDATransform
    private let logger = AppLogger(category: "OfflineVBx")
    private let signposter = OSSignposter(
        subsystem: "com.fluidaudio.diarization",
        category: .pointsOfInterest
    )

    init(config: OfflineDiarizerConfig, pldaTransform: PLDATransform) {
        self.config = config
        self.pldaTransform = pldaTransform
    }

    // MARK: - VBx Clustering Algorithm
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

        let vbxState = signposter.beginInterval("VBx Clustering Algorithm")

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
        logger.debug("VBx warm start clusters: \(speakerCount) histogram: \(histogram)")

        var featureBuffer = [Double](repeating: 0, count: frameCount * dimension)
        featureBuffer.withUnsafeMutableBufferPointer { bufferPtr in
            guard let baseAddress = bufferPtr.baseAddress else { return }
            for (index, frame) in rhoFeatures.enumerated() {
                let destination = baseAddress.advanced(by: index * dimension)
                frame.withUnsafeBufferPointer { source in
                    guard let sourceBase = source.baseAddress else { return }
                    memcpy(
                        destination,
                        sourceBase,
                        dimension * MemoryLayout<Double>.size
                    )
                }
            }
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

        let gammaSource: [Double]
        let piSource: [Double]
        let elboHistory: [Double]

        do {
            let result = try runVBx(
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
            gammaSource = result.gamma
            piSource = result.pi
            elboHistory = result.elbos
        } catch {
            logger.error("VBx failed to prepare BLAS arguments: \(error.localizedDescription)")
            gammaSource = initialGamma
            piSource = Array(repeating: 1.0 / Double(speakerCount), count: speakerCount)
            elboHistory = []
        }

        let gammaMatrix = reshapeGamma(gammaSource, frameCount: frameCount, speakerCount: speakerCount)
        let hardAssignments = gammaMatrix.map { row -> Int in
            row.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }

        if let maxPi = piSource.max(), let minPi = piSource.min() {
            logger.debug("VBx mixture weights – min: \(minPi), max: \(maxPi), count: \(piSource.count)")
        } else {
            logger.debug("VBx mixture weights unavailable")
        }

        let output = VBxOutput(
            gamma: gammaMatrix,
            pi: piSource,
            hardClusters: [hardAssignments],
            centroids: [],
            numClusters: speakerCount,
            elbos: elboHistory
        )

        signposter.endInterval("VBx Clustering Algorithm", vbxState)
        return output
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
    ) throws -> (gamma: [Double], pi: [Double], elbos: [Double]) {
        var gamma = initialGamma
        let frameCountBlas = try makeBlasIndex(frameCount, label: "VBx frame count")
        let speakerCountBlas = try makeBlasIndex(speakerCount, label: "VBx speaker count")
        let dimensionBlas = try makeBlasIndex(dimension, label: "VBx feature dimension")

        let speakerLength = vDSP_Length(speakerCount)
        let dimensionLength = vDSP_Length(dimension)
        let onesFrame = [Double](repeating: 1.0, count: frameCount)
        var rowBuffer = [Double](repeating: 0, count: speakerCount)

        if initSmoothing >= 0.0 {
            gamma.withUnsafeMutableBufferPointer { gammaPtr in
                rowBuffer.withUnsafeMutableBufferPointer { bufferPtr in
                    guard
                        let gammaBase = gammaPtr.baseAddress,
                        let scratch = bufferPtr.baseAddress
                    else { return }
                    for t in 0..<frameCount {
                        let row = gammaBase.advanced(by: t * speakerCount)
                        var multiplier = initSmoothing
                        vDSP_vsmulD(row, 1, &multiplier, scratch, 1, speakerLength)
                        var maxValue = -Double.greatestFiniteMagnitude
                        vDSP_maxvD(scratch, 1, &maxValue, speakerLength)
                        var shift = -maxValue
                        vDSP_vsaddD(scratch, 1, &shift, scratch, 1, speakerLength)
                        var count = Int32(speakerCount)
                        vvexp(scratch, scratch, &count)
                        var sumExp = 0.0
                        vDSP_sveD(scratch, 1, &sumExp, speakerLength)
                        if sumExp <= 0.0 || !sumExp.isFinite {
                            var uniform = 1.0 / Double(speakerCount)
                            vDSP_vfillD(&uniform, row, 1, speakerLength)
                        } else {
                            var invSum = 1.0 / sumExp
                            vDSP_vsmulD(scratch, 1, &invSum, row, 1, speakerLength)
                        }
                    }
                }
            }
        }

        gamma.withUnsafeMutableBufferPointer { gammaPtr in
            guard let gammaBase = gammaPtr.baseAddress else { return }
            for t in 0..<frameCount {
                let row = gammaBase.advanced(by: t * speakerCount)
                var sum = 0.0
                vDSP_sveD(row, 1, &sum, speakerLength)
                if sum <= 0.0 || !sum.isFinite {
                    var uniform = 1.0 / Double(speakerCount)
                    vDSP_vfillD(&uniform, row, 1, speakerLength)
                } else {
                    var inv = 1.0 / sum
                    vDSP_vsmulD(row, 1, &inv, row, 1, speakerLength)
                }
            }
        }

        var pi = [Double](repeating: 1.0 / Double(speakerCount), count: speakerCount)

        let phiClamped = phi.map { max($0, 1e-12) }
        let sqrtPhi = phiClamped.map { sqrt($0) }

        var rho = [Double](repeating: 0, count: features.count)
        features.withUnsafeBufferPointer { featurePtr in
            rho.withUnsafeMutableBufferPointer { rhoPtr in
                sqrtPhi.withUnsafeBufferPointer { sqrtPtr in
                    guard
                        let featureBase = featurePtr.baseAddress,
                        let rhoBase = rhoPtr.baseAddress,
                        let sqrtBase = sqrtPtr.baseAddress
                    else { return }
                    for t in 0..<frameCount {
                        let offset = t * dimension
                        vDSP_vmulD(
                            featureBase.advanced(by: offset),
                            1,
                            sqrtBase,
                            1,
                            rhoBase.advanced(by: offset),
                            1,
                            dimensionLength
                        )
                    }
                }
            }
        }

        var G = [Double](repeating: 0, count: frameCount)
        let logConstant = Double(dimension) * log(2.0 * Double.pi)
        features.withUnsafeBufferPointer { featurePtr in
            guard let featureBase = featurePtr.baseAddress else { return }
            for t in 0..<frameCount {
                let offset = t * dimension
                var sumSq: Double = 0
                vDSP_svesqD(
                    featureBase.advanced(by: offset),
                    1,
                    &sumSq,
                    dimensionLength
                )
                G[t] = -0.5 * (sumSq + logConstant)
            }
        }

        let ratio = Fa / Fb
        var invL = [Double](repeating: 0, count: speakerCount * dimension)
        var alpha = [Double](repeating: 0, count: speakerCount * dimension)
        var temp = [Double](repeating: 0, count: speakerCount * dimension)
        var phiTerms = [Double](repeating: 0, count: speakerCount)
        var gammaSum = [Double](repeating: 0, count: speakerCount)
        var logP = [Double](repeating: 0, count: frameCount * speakerCount)
        var phiScratch = [Double](repeating: 0, count: dimension)
        var phiOffset = [Double](repeating: 0, count: speakerCount)
        var logInv = [Double](repeating: 0, count: speakerCount * dimension)
        var elbos = [Double](repeating: 0, count: max(maxIterations, 1))
        let alphaCount = alpha.count
        let invLCount = invL.count

        var previousElbo = -Double.greatestFiniteMagnitude
        var iterations = 0

        for iteration in 0..<maxIterations {
            iterations = iteration + 1

            gamma.withUnsafeBufferPointer { gammaPtr in
                onesFrame.withUnsafeBufferPointer { onesPtr in
                    gammaSum.withUnsafeMutableBufferPointer { sumPtr in
                        guard
                            let gammaBase = gammaPtr.baseAddress,
                            let onesBase = onesPtr.baseAddress,
                            let sumBase = sumPtr.baseAddress
                        else { return }
                        cblas_dgemv(
                            CblasRowMajor,
                            CblasTrans,
                            frameCountBlas,
                            speakerCountBlas,
                            1.0,
                            gammaBase,
                            speakerCountBlas,
                            onesBase,
                            1,
                            0.0,
                            sumBase,
                            1
                        )
                    }
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
                            speakerCountBlas,
                            dimensionBlas,
                            frameCountBlas,
                            1.0,
                            gammaPtr.baseAddress!,
                            speakerCountBlas,
                            rhoPtr.baseAddress!,
                            dimensionBlas,
                            0.0,
                            tempPtr.baseAddress!,
                            dimensionBlas
                        )
                    }
                }
            }

            alpha.withUnsafeMutableBufferPointer { alphaPtr in
                invL.withUnsafeBufferPointer { invPtr in
                    temp.withUnsafeBufferPointer { tempPtr in
                        guard
                            let alphaBase = alphaPtr.baseAddress,
                            let invBase = invPtr.baseAddress,
                            let tempBase = tempPtr.baseAddress
                        else { return }
                        vDSP_vmulD(
                            tempBase,
                            1,
                            invBase,
                            1,
                            alphaBase,
                            1,
                            vDSP_Length(alphaCount)
                        )
                        var ratioScalar = ratio
                        vDSP_vsmulD(
                            alphaBase,
                            1,
                            &ratioScalar,
                            alphaBase,
                            1,
                            vDSP_Length(alphaCount)
                        )
                    }
                }
            }

            alpha.withUnsafeBufferPointer { alphaPtr in
                invL.withUnsafeBufferPointer { invPtr in
                    phiClamped.withUnsafeBufferPointer { phiPtr in
                        phiScratch.withUnsafeMutableBufferPointer { scratchPtr in
                            guard
                                let alphaBase = alphaPtr.baseAddress,
                                let invBase = invPtr.baseAddress,
                                let phiBase = phiPtr.baseAddress,
                                let scratchBase = scratchPtr.baseAddress
                            else { return }
                            for s in 0..<speakerCount {
                                let offset = s * dimension
                                vDSP_vsqD(
                                    alphaBase.advanced(by: offset),
                                    1,
                                    scratchBase,
                                    1,
                                    dimensionLength
                                )
                                vDSP_vaddD(
                                    scratchBase,
                                    1,
                                    invBase.advanced(by: offset),
                                    1,
                                    scratchBase,
                                    1,
                                    dimensionLength
                                )
                                vDSP_vmulD(
                                    scratchBase,
                                    1,
                                    phiBase,
                                    1,
                                    scratchBase,
                                    1,
                                    dimensionLength
                                )
                                var sum: Double = 0
                                vDSP_sveD(scratchBase, 1, &sum, dimensionLength)
                                phiTerms[s] = sum
                            }
                        }
                    }
                }
            }

            rho.withUnsafeBufferPointer { rhoPtr in
                alpha.withUnsafeBufferPointer { alphaPtr in
                    logP.withUnsafeMutableBufferPointer { logPtr in
                        cblas_dgemm(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasTrans,
                            frameCountBlas,
                            speakerCountBlas,
                            dimensionBlas,
                            1.0,
                            rhoPtr.baseAddress!,
                            dimensionBlas,
                            alphaPtr.baseAddress!,
                            dimensionBlas,
                            0.0,
                            logPtr.baseAddress!,
                            speakerCountBlas
                        )
                    }
                }
            }

            phiTerms.withUnsafeBufferPointer { phiPtr in
                phiOffset.withUnsafeMutableBufferPointer { offsetPtr in
                    guard
                        let phiBase = phiPtr.baseAddress,
                        let offsetBase = offsetPtr.baseAddress
                    else { return }
                    var negativeHalf: Double = -0.5
                    vDSP_vsmulD(
                        phiBase,
                        1,
                        &negativeHalf,
                        offsetBase,
                        1,
                        speakerLength
                    )
                }
            }

            phiOffset.withUnsafeBufferPointer { offsetPtr in
                logP.withUnsafeMutableBufferPointer { logPtr in
                    guard
                        let offsetBase = offsetPtr.baseAddress,
                        let logBase = logPtr.baseAddress
                    else { return }
                    for t in 0..<frameCount {
                        let row = logBase.advanced(by: t * speakerCount)
                        vDSP_vaddD(row, 1, offsetBase, 1, row, 1, speakerLength)
                        var g = G[t]
                        vDSP_vsaddD(row, 1, &g, row, 1, speakerLength)
                        var faScale = Fa
                        vDSP_vsmulD(row, 1, &faScale, row, 1, speakerLength)
                    }
                }
            }

            var logLikelihood = 0.0
            var logPi = [Double](repeating: 0, count: speakerCount)
            pi.withUnsafeBufferPointer { piPtr in
                logPi.withUnsafeMutableBufferPointer { logPtr in
                    guard
                        let piBase = piPtr.baseAddress,
                        let logBase = logPtr.baseAddress
                    else { return }
                    var threshold = 1e-8
                    vDSP_vthrD(
                        piBase,
                        1,
                        &threshold,
                        logBase,
                        1,
                        speakerLength
                    )
                    var count = Int32(speakerCount)
                    vvlog(logBase, logBase, &count)
                }
            }

            rowBuffer.withUnsafeMutableBufferPointer { bufferPtr in
                gamma.withUnsafeMutableBufferPointer { gammaPtr in
                    logP.withUnsafeBufferPointer { logPtr in
                        logPi.withUnsafeBufferPointer { logPiPtr in
                            guard
                                let scratch = bufferPtr.baseAddress,
                                let gammaBase = gammaPtr.baseAddress,
                                let logBase = logPtr.baseAddress,
                                let logPiBase = logPiPtr.baseAddress
                            else { return }
                            for t in 0..<frameCount {
                                let rowOffset = t * speakerCount
                                let gammaRow = gammaBase.advanced(by: rowOffset)
                                vDSP_mmovD(
                                    logBase.advanced(by: rowOffset),
                                    scratch,
                                    speakerLength,
                                    1,
                                    speakerLength,
                                    1
                                )
                                vDSP_vaddD(
                                    scratch,
                                    1,
                                    logPiBase,
                                    1,
                                    scratch,
                                    1,
                                    speakerLength
                                )
                                var rowMax = -Double.greatestFiniteMagnitude
                                vDSP_maxvD(scratch, 1, &rowMax, speakerLength)
                                var shift = -rowMax
                                vDSP_vsaddD(scratch, 1, &shift, scratch, 1, speakerLength)
                                var count = Int32(speakerCount)
                                vvexp(scratch, scratch, &count)
                                var sumExp = 0.0
                                vDSP_sveD(scratch, 1, &sumExp, speakerLength)
                                if sumExp <= 0.0 || !sumExp.isFinite {
                                    var uniform = 1.0 / Double(speakerCount)
                                    vDSP_vfillD(&uniform, gammaRow, 1, speakerLength)
                                    logLikelihood += rowMax
                                } else {
                                    var invSum = 1.0 / sumExp
                                    vDSP_vsmulD(
                                        scratch,
                                        1,
                                        &invSum,
                                        gammaRow,
                                        1,
                                        speakerLength
                                    )
                                    logLikelihood += rowMax + log(sumExp)
                                }
                            }
                        }
                    }
                }
            }

            gamma.withUnsafeBufferPointer { gammaPtr in
                onesFrame.withUnsafeBufferPointer { onesPtr in
                    pi.withUnsafeMutableBufferPointer { piPtr in
                        guard
                            let gammaBase = gammaPtr.baseAddress,
                            let onesBase = onesPtr.baseAddress,
                            let piBase = piPtr.baseAddress
                        else { return }
                        cblas_dgemv(
                            CblasRowMajor,
                            CblasTrans,
                            frameCountBlas,
                            speakerCountBlas,
                            1.0,
                            gammaBase,
                            speakerCountBlas,
                            onesBase,
                            1,
                            0.0,
                            piBase,
                            1
                        )
                    }
                }
            }

            var piSum = 0.0
            pi.withUnsafeBufferPointer { piPtr in
                guard let piBase = piPtr.baseAddress else { return }
                vDSP_sveD(piBase, 1, &piSum, speakerLength)
            }
            if piSum > 0.0 && piSum.isFinite {
                var inv = 1.0 / piSum
                pi.withUnsafeMutableBufferPointer { piPtr in
                    guard let piBase = piPtr.baseAddress else { return }
                    vDSP_vsmulD(piBase, 1, &inv, piBase, 1, speakerLength)
                }
            } else {
                var uniform = 1.0 / Double(speakerCount)
                pi.withUnsafeMutableBufferPointer { piPtr in
                    guard let piBase = piPtr.baseAddress else { return }
                    vDSP_vfillD(&uniform, piBase, 1, speakerLength)
                }
            }

            var sumLogInv = 0.0
            var sumInv = 0.0
            var sumAlphaSq = 0.0

            invL.withUnsafeBufferPointer { invPtr in
                logInv.withUnsafeMutableBufferPointer { logPtr in
                    guard
                        let invBase = invPtr.baseAddress,
                        let logBase = logPtr.baseAddress
                    else { return }
                    logBase.update(from: invBase, count: invLCount)
                    var count = Int32(invLCount)
                    vvlog(logBase, logBase, &count)
                    vDSP_sveD(logBase, 1, &sumLogInv, vDSP_Length(invLCount))
                    vDSP_sveD(invBase, 1, &sumInv, vDSP_Length(invLCount))
                }
            }

            alpha.withUnsafeBufferPointer { alphaPtr in
                guard let alphaBase = alphaPtr.baseAddress else { return }
                vDSP_svesqD(alphaBase, 1, &sumAlphaSq, vDSP_Length(alphaCount))
            }

            var elbo = logLikelihood
            elbo += Fb * 0.5 * (sumLogInv - sumInv - sumAlphaSq + Double(invLCount))

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

    /// Refines clusters with optional speaker count constraints.
    ///
    /// - Parameters:
    ///   - rhoFeatures: PLDA-transformed features.
    ///   - trainingEmbeddings: Original embeddings for K-Means re-clustering.
    ///   - initialClusters: Initial cluster assignments from AHC.
    ///   - constraints: Optional speaker count constraints.
    /// - Returns: VBx output with potentially adjusted speaker count.
    func refineWithConstraints(
        rhoFeatures: [[Double]],
        trainingEmbeddings: [[Double]],
        initialClusters: [Int],
        constraints: SpeakerCountConstraints?
    ) -> VBxOutput {
        var output = refine(rhoFeatures: rhoFeatures, initialClusters: initialClusters)

        guard let constraints = constraints else {
            return output
        }

        let detectedCount = output.numClusters
        guard constraints.needsAdjustment(detectedCount: detectedCount) else {
            return output
        }

        let targetCount = constraints.targetCount(detectedCount: detectedCount)
        logger.info(
            "Speaker count \(detectedCount) outside bounds [\(constraints.minSpeakers), \(constraints.maxSpeakers)]; re-clustering to \(targetCount)"
        )

        let (kmeansClusters, centroids) = KMeansClustering.clusterWithCentroids(
            embeddings: trainingEmbeddings,
            numClusters: targetCount,
            maxIterations: 100
        )

        return VBxOutput(
            gamma: output.gamma,
            pi: output.pi,
            hardClusters: [kmeansClusters],
            centroids: centroids,
            numClusters: targetCount,
            elbos: output.elbos,
            wasAdjusted: true,
            originalClusterCount: detectedCount
        )
    }
}
