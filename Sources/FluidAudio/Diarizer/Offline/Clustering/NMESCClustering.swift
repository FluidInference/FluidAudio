import Accelerate
import Foundation

/// Normalized Maximum Eigengap Spectral Clustering (NME-SC).
///
/// Drop-in alternative to `AHCClustering` for the offline pipeline's initial
/// cluster assignment. Instead of cutting a linkage dendrogram at a fixed
/// similarity threshold, NME-SC:
///  1. builds a cosine affinity matrix over the embeddings,
///  2. sweeps a grid of per-row pruning levels `p` (keep only each row's
///     strongest `p·n` edges — this severs the weak "bridge" links that make
///     centroid-linkage AHC chain distinct speakers together),
///  3. for each `p`, estimates the cluster count from the eigengap of the
///     normalized graph Laplacian, scoring the candidate by the normalized
///     eigengap ratio (gap / p),
///  4. clusters the spectral embedding of the best candidate with K-Means.
///
/// The cluster count therefore adapts per recording — there is no global
/// similarity threshold to tune, which is what makes the same configuration
/// work on both clean multi-speaker audio and far-field meeting audio.
struct NMESCClustering {

    private let logger = AppLogger(category: "OfflineNMESC")

    /// Pruning grid: fraction of each row's edges to keep.
    private let pGrid: [Double] = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25]
    /// Upper bound on the estimated speaker count.
    private let maxSpeakers: Int

    init(maxSpeakers: Int = 8) {
        self.maxSpeakers = maxSpeakers
    }

    /// Returns a cluster index per embedding, like `AHCClustering.cluster`.
    func cluster(embeddingFeatures: [[Double]]) -> [Int] {
        let n = embeddingFeatures.count
        guard n > 0 else { return [] }
        guard let dim = embeddingFeatures.first?.count, dim > 0 else {
            return Array(repeating: 0, count: n)
        }
        // Too few points for spectral structure estimation.
        guard n >= 8 else { return Array(repeating: 0, count: n) }

        let similarity = cosineMatrix(embeddingFeatures, n: n, dim: dim)

        var best: (score: Double, p: Double, k: Int, affinity: [Double])?
        for p in pGrid {
            let keep = max(2, Int(Double(n) * p))
            guard keep < n else { continue }
            let affinity = prunedAffinity(similarity, n: n, keep: keep)
            guard let eigenvalues = laplacianEigenvalues(
                affinity: affinity, n: n, count: min(maxSpeakers + 1, n))
            else { continue }
            // Skip the trivial first eigenvalue; gap g_k = λ_{k+1} − λ_k
            // (indices into the sorted spectrum) estimates k clusters.
            var bestGap = -1.0
            var bestK = 1
            let usable = min(maxSpeakers, eigenvalues.count - 1)
            for k in 2...max(2, usable) where k < eigenvalues.count {
                let gap = eigenvalues[k] - eigenvalues[k - 1]
                if gap > bestGap {
                    bestGap = gap
                    bestK = k
                }
            }
            let score = bestGap / p  // normalized eigengap ratio
            if best == nil || score > best!.score {
                best = (score, p, bestK, affinity)
            }
        }

        guard let chosen = best else {
            logger.warning("NME-SC found no usable pruning level; single cluster")
            return Array(repeating: 0, count: n)
        }
        logger.debug("NME-SC selected p=\(chosen.p) k=\(chosen.k) (score \(chosen.score))")
        guard chosen.k > 1 else { return Array(repeating: 0, count: n) }

        guard let spectral = laplacianEigenvectors(
            affinity: chosen.affinity, n: n, k: chosen.k)
        else {
            logger.warning("NME-SC eigenvector computation failed; single cluster")
            return Array(repeating: 0, count: n)
        }

        // Row-normalize the spectral embedding, then K-Means (deterministic
        // n-init variant already used by the constrained-count path).
        var rows: [[Double]] = []
        rows.reserveCapacity(n)
        for i in 0..<n {
            var row = Array(spectral[(i * chosen.k)..<((i + 1) * chosen.k)])
            var norm = 0.0
            for v in row { norm += v * v }
            let scale = norm > 0 ? 1.0 / sqrt(norm) : 0
            for j in 0..<row.count { row[j] *= scale }
            rows.append(row)
        }
        let (labels, _) = KMeansClustering.clusterWithCentroidsNInit(
            embeddings: rows,
            numClusters: chosen.k,
            maxIterations: 100,
            nInit: 10,
            baseSeed: 0
        )
        return labels
    }

    // MARK: - Affinity construction

    /// Dense cosine similarity via one syrk-style GEMM over L2-normalized rows.
    private func cosineMatrix(_ features: [[Double]], n: Int, dim: Int) -> [Double] {
        var normalized = [Double](repeating: 0, count: n * dim)
        for (i, row) in features.enumerated() {
            var norm = 0.0
            row.withUnsafeBufferPointer { p in
                vDSP_svesqD(p.baseAddress!, 1, &norm, vDSP_Length(dim))
            }
            var scale = norm > 0 ? 1.0 / sqrt(norm) : 0
            row.withUnsafeBufferPointer { src in
                normalized.withUnsafeMutableBufferPointer { dst in
                    vDSP_vsmulD(src.baseAddress!, 1, &scale,
                                dst.baseAddress!.advanced(by: i * dim), 1, vDSP_Length(dim))
                }
            }
        }
        var out = [Double](repeating: 0, count: n * n)
        normalized.withUnsafeBufferPointer { a in
            out.withUnsafeMutableBufferPointer { c in
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            Int32(n), Int32(n), Int32(dim),
                            1.0, a.baseAddress!, Int32(dim),
                            a.baseAddress!, Int32(dim),
                            0.0, c.baseAddress!, Int32(n))
            }
        }
        return out
    }

    /// Keep each row's `keep` strongest edges, symmetrize with max, zero diagonal.
    private func prunedAffinity(_ S: [Double], n: Int, keep: Int) -> [Double] {
        var A = [Double](repeating: 0, count: n * n)
        var indices = Array(0..<n)
        for i in 0..<n {
            let row = Array(S[(i * n)..<((i + 1) * n)])
            indices.sort { row[$0] > row[$1] }
            var kept = 0
            for j in indices {
                if j == i { continue }
                A[i * n + j] = row[j]
                kept += 1
                if kept >= keep { break }
            }
        }
        for i in 0..<n {
            for j in (i + 1)..<n {
                let v = max(A[i * n + j], A[j * n + i])
                A[i * n + j] = v
                A[j * n + i] = v
            }
            A[i * n + i] = 0
        }
        return A
    }

    // MARK: - Laplacian spectrum (LAPACK)

    /// Symmetric normalized Laplacian L = I − D^{-1/2} A D^{-1/2}.
    private func normalizedLaplacian(affinity A: [Double], n: Int) -> [Double] {
        var dInvSqrt = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var deg = 0.0
            for j in 0..<n { deg += A[i * n + j] }
            dInvSqrt[i] = deg > 0 ? 1.0 / sqrt(deg) : 0
        }
        var L = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                let v = -dInvSqrt[i] * A[i * n + j] * dInvSqrt[j]
                L[i * n + j] = i == j ? 1.0 + v : v
            }
        }
        return L
    }

    /// Smallest `count` eigenvalues of the normalized Laplacian, ascending.
    private func laplacianEigenvalues(affinity: [Double], n: Int, count: Int) -> [Double]? {
        var matrix = normalizedLaplacian(affinity: affinity, n: n)
        guard let w = symmetricEigen(&matrix, n: n, wantVectors: false)?.values else { return nil }
        return Array(w.prefix(count))
    }

    /// Row-major n×k matrix of the eigenvectors for the k smallest eigenvalues.
    private func laplacianEigenvectors(affinity: [Double], n: Int, k: Int) -> [Double]? {
        var matrix = normalizedLaplacian(affinity: affinity, n: n)
        guard let result = symmetricEigen(&matrix, n: n, wantVectors: true) else { return nil }
        // dsyevd returns eigenvalues ascending with eigenvectors as columns of
        // the (column-major) matrix — i.e. vector j occupies elements
        // [j*n ..< (j+1)*n] of the storage we passed in.
        var out = [Double](repeating: 0, count: n * k)
        for j in 0..<k {
            for i in 0..<n {
                out[i * k + j] = result.vectors[j * n + i]
            }
        }
        return out
    }

    /// LAPACK dsyevd on a symmetric matrix. `matrix` is destroyed; when
    /// `wantVectors`, its storage holds the eigenvectors on return.
    private func symmetricEigen(
        _ matrix: inout [Double], n: Int, wantVectors: Bool
    ) -> (values: [Double], vectors: [Double])? {
        var jobz: Int8 = wantVectors ? Int8(UnicodeScalar("V").value) : Int8(UnicodeScalar("N").value)
        var uplo: Int8 = Int8(UnicodeScalar("U").value)
        var nn = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var w = [Double](repeating: 0, count: n)
        var info: __CLPK_integer = 0
        var lwork: __CLPK_integer = -1
        var liwork: __CLPK_integer = -1
        var workQuery = 0.0
        var iworkQuery: __CLPK_integer = 0
        dsyevd_(&jobz, &uplo, &nn, &matrix, &lda, &w, &workQuery, &lwork, &iworkQuery, &liwork, &info)
        guard info == 0 else { return nil }
        lwork = __CLPK_integer(workQuery)
        liwork = iworkQuery
        var work = [Double](repeating: 0, count: max(1, Int(lwork)))
        var iwork = [__CLPK_integer](repeating: 0, count: max(1, Int(liwork)))
        dsyevd_(&jobz, &uplo, &nn, &matrix, &lda, &w, &work, &lwork, &iwork, &liwork, &info)
        guard info == 0 else { return nil }
        return (values: w, vectors: matrix)
    }
}
