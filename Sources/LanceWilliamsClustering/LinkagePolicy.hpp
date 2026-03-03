#pragma once
#include "SpeakerEmbeddingWrapper.hpp"
#include "Cluster.hpp"
#include <Accelerate/Accelerate.h>
#include <vector>

class WardLinkage;
class UPGMALinkage;
class CompleteLinkage;
class SingleLinkage;

enum class LinkagePolicyType {
    wardLinkage,
    completeLinkage,
    singleLinkage,
    upgma,
};

class EmbeddingDistanceMatrix;

class LinkagePolicy {
public:
    static const WardLinkage wardLinkage;
    static const UPGMALinkage upgmaLinkage;
    static const CompleteLinkage completeLinkage;
    static const SingleLinkage singleLinkage;
    
    [[nodiscard]] virtual consteval LinkagePolicyType getPolicyType() const = 0;
    
    [[nodiscard]] virtual float distance(float distAC, float wA,
                                         float distBC, float wB,
                                         float distAB, float wC) const = 0;
    
    [[nodiscard]] virtual float distance(const SpeakerEmbeddingWrapper &a,
                                         const SpeakerEmbeddingWrapper &b) const = 0;
    
    virtual void computeCentroid(const SpeakerEmbeddingWrapper *embeddings,
                                 const Cluster &cluster,
                                 SpeakerEmbeddingWrapper& result) const = 0;
    
    [[nodiscard]] static const LinkagePolicy *getPolicy(LinkagePolicyType forType);
    
    [[nodiscard]] static inline float distance(const LinkagePolicy* policy,
                                               const SpeakerEmbeddingWrapper &a,
                                               const SpeakerEmbeddingWrapper &b) {
        return policy->distance(a, b);
    }
    
protected:
    enum class NormalizeBy {
        weight,
        l2Norm
    };
    
    template<NormalizeBy Normalization>
    static void computeCentroidHelper(const SpeakerEmbeddingWrapper *embeddings,
                                      const Cluster &cluster,
                                      SpeakerEmbeddingWrapper& result);
};

class WardLinkage: public LinkagePolicy {
public:
    [[nodiscard]] consteval LinkagePolicyType getPolicyType() const final {
        return LinkagePolicyType::wardLinkage;
    }
    
    [[nodiscard]] inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) const final {
        return ((wA + wC) * distAC + (wB + wC) * distBC - wC * distAB) / (wA + wB + wC);
    }

    [[nodiscard]] inline float distance(const SpeakerEmbeddingWrapper &a, const SpeakerEmbeddingWrapper &b) const final {
        return a.wardDistanceTo(b);
    }

    inline void
    computeCentroid(const SpeakerEmbeddingWrapper *embeddings,
                    const Cluster &cluster,
                    SpeakerEmbeddingWrapper& result) const final {
        computeCentroidHelper<NormalizeBy::l2Norm>(embeddings, cluster, result);
    }
};

class UPGMALinkage: public LinkagePolicy {
public:
    [[nodiscard]] consteval LinkagePolicyType getPolicyType() const final {
        return LinkagePolicyType::upgma;
    }
    
    [[nodiscard]] inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) const final {
        return (wA * distAC + wB * distBC) / (wA + wB);
    }

    [[nodiscard]] inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) const final {
        return a.unitCosineDistanceTo(b);
    }

    inline void computeCentroid(const SpeakerEmbeddingWrapper* embeddings,
                                const Cluster& cluster,
                                SpeakerEmbeddingWrapper& result) const final {
        computeCentroidHelper<NormalizeBy::weight>(embeddings, cluster, result);
    }
};

class CompleteLinkage: public LinkagePolicy {
public:
    [[nodiscard]] consteval LinkagePolicyType getPolicyType() const final {
        return LinkagePolicyType::completeLinkage;
    }
    
    [[nodiscard]] inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) const final {
        return std::max(distAC, distBC);
    }

    [[nodiscard]] inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) const final {
        return a.unitCosineDistanceTo(b);
    }

    inline void computeCentroid(const SpeakerEmbeddingWrapper* embeddings,
                                const Cluster& cluster,
                                SpeakerEmbeddingWrapper& result) const final {
        computeCentroidHelper<NormalizeBy::l2Norm>(embeddings, cluster, result);
    }
};

class SingleLinkage: public LinkagePolicy {
public:
    [[nodiscard]] consteval LinkagePolicyType getPolicyType() const final {
        return LinkagePolicyType::singleLinkage;
    }
    
    [[nodiscard]] inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) const final {
        return std::min(distAC, distBC);
    }

    [[nodiscard]] inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) const final {
        return a.unitCosineDistanceTo(b);
    }

    inline void computeCentroid(const SpeakerEmbeddingWrapper* embeddings,
                                const Cluster& cluster,
                                SpeakerEmbeddingWrapper& result) const final {
        computeCentroidHelper<NormalizeBy::l2Norm>(embeddings, cluster, result);
    }
};

