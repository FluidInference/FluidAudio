#include "FastClusterWrapper.h"

#include <cmath>
#include <cstddef>
#include <exception>
#include <new>
#include <vector>

#ifndef fc_isnan
#define fc_isnan(X) ((X) != (X))
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpragma-messages"
#endif

#include "fastcluster_internal.hpp"

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace {

struct CentroidDissimilarity {
    const t_float *data;
    const t_index dimension;
    const t_index count;
    std::vector<t_float> centroidStorage;
    std::vector<t_index> members;

    CentroidDissimilarity(const t_float *input, t_index sampleCount, t_index dim)
        : data(input),
          dimension(dim),
          count(sampleCount),
          centroidStorage(sampleCount > 1 ? static_cast<size_t>((sampleCount - 1) * dim) : 0u),
          members(sampleCount > 0 ? static_cast<size_t>(2 * sampleCount - 1) : 0u, 0) {
        for (t_index i = 0; i < count; ++i) {
            members[static_cast<size_t>(i)] = 1;
        }
    }

    template <bool checkNaN>
    t_float sqeuclidean(const t_index i, const t_index j) const {
        const t_float *pi = basePointer(i);
        const t_float *pj = basePointer(j);
        t_float sum = 0;
        for (t_index k = 0; k < dimension; ++k) {
            const t_float diff = pi[k] - pj[k];
            sum += diff * diff;
        }
        if constexpr (checkNaN) {
#if HAVE_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
            if (fc_isnan(sum)) {
#if HAVE_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif
                throw nan_error();
            }
        }
        return sum;
    }

    t_float sqeuclidean_extended(const t_index i, const t_index j) const {
        const t_float *pi = extendedPointer(i);
        const t_float *pj = extendedPointer(j);
        t_float sum = 0;
        for (t_index k = 0; k < dimension; ++k) {
            const t_float diff = pi[k] - pj[k];
            sum += diff * diff;
        }
#if HAVE_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        if (fc_isnan(sum)) {
            throw nan_error();
        }
#if HAVE_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif
        return sum;
    }

    void merge(const t_index i, const t_index j, const t_index newNode) {
        const t_float *pi = extendedPointer(i);
        const t_float *pj = extendedPointer(j);
        t_float *pn = centroidPointer(newNode);
        const t_float mi = static_cast<t_float>(members[static_cast<size_t>(i)]);
        const t_float mj = static_cast<t_float>(members[static_cast<size_t>(j)]);
        const t_float denom = mi + mj;
        for (t_index k = 0; k < dimension; ++k) {
            pn[k] = (pi[k] * mi + pj[k] * mj) / denom;
        }
        members[static_cast<size_t>(newNode)] = members[static_cast<size_t>(i)] + members[static_cast<size_t>(j)];
    }

    void merge_weighted(const t_index i, const t_index j, const t_index newNode) {
        const t_float *pi = extendedPointer(i);
        const t_float *pj = extendedPointer(j);
        t_float *pn = centroidPointer(newNode);
        for (t_index k = 0; k < dimension; ++k) {
            pn[k] = static_cast<t_float>(0.5) * (pi[k] + pj[k]);
        }
        members[static_cast<size_t>(newNode)] = members[static_cast<size_t>(i)] + members[static_cast<size_t>(j)];
    }

    t_float ward(const t_index i, const t_index j) const {
        return sqeuclidean<true>(i, j);
    }

    t_float ward_initial(const t_index i, const t_index j) const {
        return sqeuclidean<true>(i, j);
    }

    static t_float ward_initial_conversion(const t_float value) {
        return value * static_cast<t_float>(0.5);
    }

    t_float ward_extended(const t_index i, const t_index j) const {
        return sqeuclidean_extended(i, j);
    }

    void postprocess(cluster_result &result) const {
        result.sqrt();
    }

private:
    const t_float *basePointer(const t_index index) const {
        return data + static_cast<size_t>(index) * static_cast<size_t>(dimension);
    }

    const t_float *extendedPointer(const t_index index) const {
        if (index < count) {
            return basePointer(index);
        }
        return centroidStorage.data() + static_cast<size_t>(index - count) * static_cast<size_t>(dimension);
    }

    t_float *centroidPointer(const t_index index) {
        return centroidStorage.data() + static_cast<size_t>(index - count) * static_cast<size_t>(dimension);
    }
};

class LinkageOutput {
public:
    explicit LinkageOutput(t_float *buffer) : cursor(buffer) {}

    void append(t_index node1, t_index node2, t_float distance, t_float size) {
        if (node1 < node2) {
            *(cursor++) = static_cast<t_float>(node1);
            *(cursor++) = static_cast<t_float>(node2);
        } else {
            *(cursor++) = static_cast<t_float>(node2);
            *(cursor++) = static_cast<t_float>(node1);
        }
        *(cursor++) = distance;
        *(cursor++) = size;
    }

private:
    t_float *cursor;
};

template <bool sorted>
void generateSciPyDendrogram(t_float *Z, cluster_result &Z2, const t_index N) {
    union_find nodes(sorted ? 0 : N);
    if (!sorted) {
        std::stable_sort(Z2[0], Z2[N - 1]);
    }

    LinkageOutput output(Z);
    t_index node1;
    t_index node2;

    for (node const *entry = Z2[0]; entry != Z2[N - 1]; ++entry) {
        if (sorted) {
            node1 = entry->node1;
            node2 = entry->node2;
        } else {
            node1 = nodes.Find(entry->node1);
            node2 = nodes.Find(entry->node2);
            nodes.Union(node1, node2);
        }
        output.append(node1, node2, entry->dist,
                      ((node1 < N) ? 1 : Z_(node1 - N, 3)) + ((node2 < N) ? 1 : Z_(node2 - N, 3)));
    }
}

} // namespace

fastcluster_wrapper_status fastcluster_compute_centroid_linkage(
    const double *data,
    size_t pointCount,
    size_t dimension,
    double *dendrogramOut,
    size_t dendrogramLength
) {
    if (data == nullptr || dendrogramOut == nullptr) {
        return FASTCLUSTER_WRAPPER_INVALID_ARGUMENT;
    }
    if (pointCount == 0) {
        return FASTCLUSTER_WRAPPER_SUCCESS;
    }
    if (dimension == 0) {
        return FASTCLUSTER_WRAPPER_INVALID_ARGUMENT;
    }
    if (pointCount > static_cast<size_t>(MAX_INDEX) || dimension > static_cast<size_t>(MAX_INDEX)) {
        return FASTCLUSTER_WRAPPER_INDEX_OVERFLOW;
    }

    const size_t requiredLength = (pointCount > 1) ? (pointCount - 1) * 4 : 0;
    if (dendrogramLength < requiredLength) {
        return FASTCLUSTER_WRAPPER_OUTPUT_TOO_SMALL;
    }

    if (pointCount == 1) {
        return FASTCLUSTER_WRAPPER_SUCCESS;
    }

    try {
        const t_index N = static_cast<t_index>(pointCount);
        const t_index dim = static_cast<t_index>(dimension);

        CentroidDissimilarity dist(data, N, dim);
        cluster_result result(N - 1);
        generic_linkage_vector_alternative<METHOD_VECTOR_CENTROID>(N, dist, result);
        dist.postprocess(result);
        generateSciPyDendrogram<true>(dendrogramOut, result, N);
        return FASTCLUSTER_WRAPPER_SUCCESS;
    } catch (const std::bad_alloc &) {
        return FASTCLUSTER_WRAPPER_ALLOCATION_FAILURE;
    } catch (const nan_error &) {
        return FASTCLUSTER_WRAPPER_RUNTIME_ERROR;
    } catch (const std::exception &) {
        return FASTCLUSTER_WRAPPER_RUNTIME_ERROR;
    } catch (...) {
        return FASTCLUSTER_WRAPPER_UNKNOWN_ERROR;
    }
}
