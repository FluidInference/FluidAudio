#include "LinkagePolicy.hpp"
#include "EmbeddingDistanceMatrix.hpp"

const UPGMALinkage LinkagePolicy::upgmaLinkage{};
const WardLinkage LinkagePolicy::wardLinkage{};
const CompleteLinkage LinkagePolicy::completeLinkage{};
const SingleLinkage LinkagePolicy::singleLinkage{};

const LinkagePolicy* LinkagePolicy::getPolicy(LinkagePolicyType forType) {
    switch (forType) {
        case LinkagePolicyType::wardLinkage:
            return &LinkagePolicy::wardLinkage;
        case LinkagePolicyType::upgma:
            return &LinkagePolicy::upgmaLinkage;
        case LinkagePolicyType::singleLinkage:
            return &LinkagePolicy::singleLinkage;
        case LinkagePolicyType::completeLinkage:
            return &LinkagePolicy::completeLinkage;
        default:
            return nullptr;
    }
}

template<LinkagePolicy::NormalizeBy Normalization>
void LinkagePolicy::computeCentroidHelper(const SpeakerEmbeddingWrapper *embeddings, const Cluster &cluster, SpeakerEmbeddingWrapper& result) {
    if (cluster.count() == 0) return;
    if (cluster.count() == 1) {
        result.setFrom(embeddings[cluster[0]]);
        return;
    }

    auto iter = cluster.begin();
    const auto &e0 = embeddings[*(iter++)];
    const auto &e1 = embeddings[*(iter++)];
    
    auto buffer = result.vector();
    auto& isOutlierOut = result.isOutlier();
    
    isOutlierOut = e0.isOutlier() && e1.isOutlier();
    result.weight() = cluster.weight();

    float normalizer, w0, w1, weight;
    if constexpr (Normalization == NormalizeBy::weight) {
        normalizer = 1.f / cluster.weight();
        w0 = e0.weight() * normalizer;
        w1 = e1.weight() * normalizer;
    } else {
        w0 = e0.weight();
        w1 = e1.weight();
    }

    // Centroid = emb0 * w0 + emb1 * w1
    vDSP_vsmsma(e0.vector(), 1, &w0,
                e1.vector(), 1, &w1,
                buffer, 1,
                SpeakerEmbeddingWrapper::dims);

    // Track segment IDs
    long segmentCount = 0;
    for (auto index: cluster.indices()) {
        segmentCount += embeddings[index].segmentCount();
    }
    
    auto& segments = result.segments();
    segments.reserve(segmentCount);
    segments.insert(segments.end(), e0.segments().begin(), e0.segments().end());
    segments.insert(segments.end(), e1.segments().begin(), e1.segments().end());

    for (; iter != cluster.end(); ++iter) {
        const auto &embedding = embeddings[*iter];
        if constexpr (Normalization == NormalizeBy::weight)
            weight = embedding.weight() * normalizer;
        else
            weight = embedding.weight();

        // Centroid <- emb * w + centroid
        vDSP_vsma(embedding.vector(), 1, &weight,
                  buffer, 1,
                  buffer, 1,
                  SpeakerEmbeddingWrapper::dims);

        isOutlierOut &= embedding.isOutlier();

        // Update segment IDs
        auto newSegments = embedding.segments();
        segments.insert(segments.end(), newSegments.begin(), newSegments.end());
    }
    
    // Remove duplicate segments
    std::sort(segments.begin(), segments.end());
    segments.erase(std::unique(segments.begin(), segments.end()), segments.end());

    if constexpr (Normalization == NormalizeBy::l2Norm) {
        result.normalizedInPlace();
    }
}
