//
// Created by Benjamin Lee on 1/29/26.
//

#pragma once

#include "EmbeddingDistanceMatrix.hpp"
#include "Cluster.hpp"
#include "LinkagePolicy.hpp"
#include <unordered_set>

class Dendrogram {
public:
    static constexpr DendrogramNode nullNode = {-1, -1, -1, -1, -1.f, 0};
    
    Dendrogram() = default;
    Dendrogram(Dendrogram const& clustering) = default;
    Dendrogram(Dendrogram&& clustering) noexcept;
    explicit Dendrogram(EmbeddingDistanceMatrix const& embeddingMatrix);
    
    // Get number of nodes
    [[nodiscard]] inline long nodeCount() const { return count; }

    // Get pointer to nodes
    [[nodiscard]] inline std::shared_ptr<DendrogramNode[]> nodes() const { return _nodes; }
    
    // Get root node
    [[nodiscard]] inline DendrogramNode root() const { return _nodes ? _nodes[_rootId] : nullNode; }

    // Get node by index (returns nullNode if out of range)
    [[nodiscard]] inline DendrogramNode node(long index) const {
        if (!_nodes || index < 0 || index >= count) {
            return nullNode;
        }
        return _nodes[index];
    }
    
    // Get root ID
    [[nodiscard]] inline long rootId() const { return _rootId; }
    
    // Get elbow linkage lower bound
    [[nodiscard]] inline float lowerElbowLinkage() const { return _lowerElbowLinkage; }
    
    // Get elbow linkage upper bound
    [[nodiscard]] inline float upperElbowLinkage() const { return _upperElbowLinkage; }
    
    /**
     * @brief Extract sub-clusters from a parent node
     * @param root Parent node ID
     * @param linkageThreshold Maximum linkage distance to split. Leave as -1 for the max gap linkage threshold.
     * @param maxClusters Maximum number of clusters to extract. Leave as -1 for no limit.
     * @return Array of Cluster objects
     */
    [[nodiscard]] std::vector<Cluster> extractSubClusters(long root, float linkageThreshold = -1.f, long maxClusters = -1) const;

    /**
     * @brief Extract sub-cluster root IDs from a parent node
     * @param root Parent node ID
     * @param linkageThreshold Maximum linkage distance to split. Leave as -1 for the max gap linkage threshold.
     * @param maxClusters Maximum number of clusters to extract. Leave as -1 for no limit.
     * @return Array of each cluster's root ID
     */
    [[nodiscard]] std::vector<long> extractSubClusterRoots(long root, float linkageThreshold = -1.f, long maxClusters = -1) const;

    /**
     * @brief Extract clusters
     * @param linkageThreshold Maximum linkage distance to split. Leave as -1 for the max gap linkage threshold.
     * @param maxClusters Maximum number of clusters to extract. Leave as -1 for no limit.
     * @return Array of Cluster objects
     */
    [[nodiscard]] inline std::vector<Cluster> extractClusters(float linkageThreshold = -1.f, long maxClusters = -1) const {
        return extractSubClusters(_rootId, linkageThreshold, maxClusters);
    }

    /**
     * @brief Extract cluster root IDs
     * @param linkageThreshold Maximum linkage distance to split. Leave as -1 for the max gap linkage threshold.
     * @param maxClusters Maximum number of clusters to extract. Leave as -1 for no limit.
     * @return Array of each cluster's root ID
     */
    [[nodiscard]] inline std::vector<long> extractClusterRoots(float linkageThreshold = -1.f, long maxClusters = -1) const {
        return extractSubClusterRoots(_rootId, linkageThreshold, maxClusters);
    }

    /**
     * @brief Collect the cluster leaves. These can be used to extract the original embeddings from the distance matrix.
     * @param root Cluster root ID
     * @return Cluster object containing cluster leaf IDs
     */
    [[nodiscard]] Cluster collectClusterLeaves(long root) const;
    
    Dendrogram& operator=(const Dendrogram&) = default;
    Dendrogram& operator=(Dendrogram&&) noexcept;
    
private:
    struct Aux {
        const LinkagePolicy* linkagePolicy;
        std::unique_ptr<float[]> matrix;
        std::unique_ptr<bool[]> activeFlags;
        std::unique_ptr<long[]> matrixToNode;
        long size;
        long numClustersRemaining;
        long firstActiveMatrixIndex;
        long numMerged;
        float maxGap;
    };
    
    std::shared_ptr<DendrogramNode[]> _nodes{nullptr};
    long count{0};
    long _rootId{-1};
    float _lowerElbowLinkage{0};
    float _upperElbowLinkage{0};
    
    void buildDendrogram(Aux& aux);
    
    // Merge two rows in the matrix to make a new cluster and return the row of the merged buildDendrogram
    std::ptrdiff_t merge(std::ptrdiff_t leftIndex, std::ptrdiff_t rightIndex, Aux& aux);
    
    // Get the index of the nearest neighbor to a matrix row
    [[nodiscard]] static std::ptrdiff_t nearestNeighbor(std::ptrdiff_t index, Aux& aux) ;

    // Get the _spread between the cluster at index row and the buildDendrogram at index col where row != col
    [[nodiscard]] static inline float distance(std::ptrdiff_t row, std::ptrdiff_t col, Aux& aux) {
        if (col > row) std::swap(row, col);
        return aux.matrix[row * (row - 1) / 2 + col];
    }
};
