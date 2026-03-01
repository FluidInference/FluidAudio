#pragma once

#include <vector>
#include <span>
#include <ranges>
#include "SpeakerEmbeddingWrapper.hpp"
#include "LinkagePolicy.hpp"

class Dendrogram;

class EmbeddingDistanceMatrix {
private:
    float* _matrixStart;
    float* _matrixEnd;
    SpeakerEmbeddingWrapper* _embeddings;
    long _size{0};
    long _capacity{0};
    const LinkagePolicy* _linkagePolicy{nullptr};

    friend class Dendrogram;
    
public:
    EmbeddingDistanceMatrix() = default;
    explicit EmbeddingDistanceMatrix(const LinkagePolicy* linkagePolicy);
    inline explicit EmbeddingDistanceMatrix(LinkagePolicyType linkagePolicy):
            EmbeddingDistanceMatrix(LinkagePolicy::getPolicy(linkagePolicy)) {}
    EmbeddingDistanceMatrix(EmbeddingDistanceMatrix&& other) noexcept;
    EmbeddingDistanceMatrix(const EmbeddingDistanceMatrix&) = delete;
    ~EmbeddingDistanceMatrix();
    
    [[nodiscard]] inline long size() const { return _size; }
    
    [[nodiscard]] inline long capacity() const { return _capacity; }
    
    [[nodiscard]] inline const LinkagePolicy* linkagePolicy() { return _linkagePolicy; }
    
    [[nodiscard]] inline const SpeakerEmbeddingWrapper* embeddings() const {
        return _embeddings;
    }
    
    // Free the matrix
    inline void free() {
        delete[] _matrixStart;
        delete[] _embeddings;
        _matrixStart = nullptr;
        _matrixEnd = nullptr;
        _embeddings = nullptr;
        _size = 0;
        _capacity = 0;
    }
    
    /**
     * @brief Compute the centroid of a cluster
     * @param cluster The cluster from the dendrogram
     * @param result The `SpeakerEmbeddingWrapper` object in which the result should be stored
     * @returns True if successful, false if not.
     */
    inline bool computeClusterCentroid(const Cluster& cluster,
                                       SpeakerEmbeddingWrapper& result) const {
        if (_embeddings == nullptr) return false;
        _linkagePolicy->computeCentroid(_embeddings, cluster, result);
        return true;
    }
    
    /**
     * @brief Reserve memory for more embeddings if necessary.
     * @param newCapacity The new maximum number of embeddings that can be stored.
     */
    void reserve(long newCapacity);

    /**
     * @brief Get the spread between the embedding at index row and the embedding at index col 
     * @param row Index of embedding 1
     * @param col Index of embedding 2
     * @return The _spread between the two embeddings
     */
    [[nodiscard]] float distance(long row, long col) const;
    
    // Get the embedding at the row
    [[nodiscard]] inline SpeakerEmbeddingWrapper embedding(long index) const {
        return _embeddings[index];
    }
    
    /**
     * Insert a new embedding to the matrix at the next free slot and records the slot in the embedding
     * @param embedding The embedding to add
     */
    void append(SpeakerEmbeddingWrapper const& embedding);

    /**
     * Replace the embedding at a given index with a new one and update the distances
     * @param index The index of the embedding to replace
     * @param embedding The embedding to add
     */
    void replace(long index, SpeakerEmbeddingWrapper& embedding);
    
    /**
     * @brief Build dendrogram from the embeddings
     * @return Dendrogram object
     */
    [[nodiscard]] Dendrogram dendrogram() const;
    
    EmbeddingDistanceMatrix& operator=(const EmbeddingDistanceMatrix&) = delete;
    EmbeddingDistanceMatrix& operator=(EmbeddingDistanceMatrix&&) noexcept;
};
