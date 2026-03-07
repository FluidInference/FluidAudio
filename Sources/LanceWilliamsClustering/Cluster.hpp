#pragma once

#include "SpeakerEmbeddingWrapper.hpp"
#include <memory>
#include <span>

struct DendrogramNode {
    long matrixIndex{0};
    long leftChild{-1};
    long rightChild{-1};
    long count{1};
    float weight{1};
    float mergeDistance{0};
};

class Cluster {
private:
    std::shared_ptr<long[]> _indices{nullptr};
    long _count;
    float _weight;
    float _spread;

public:
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type        = long;
        using difference_type   = std::ptrdiff_t;
        using pointer           = long*;
        using reference         = long&;
        
        pointer ptr;
        inline reference operator*() const { return *ptr; }
        inline pointer operator->() const { return ptr; }
        inline Iterator& operator++() { ++ptr; return *this; }
        inline Iterator operator++(int) { auto tmp = *this; ++ptr; return tmp; }
        friend bool operator== (const Iterator& a, const Iterator& b) = default;
        friend bool operator!= (const Iterator& a, const Iterator& b) = default;
    };

    inline Cluster(Cluster&& other) noexcept: 
            _indices(std::move(other._indices)),
            _count(other._count),
            _weight(other._weight), 
            _spread(other._spread) {}
            
    inline Cluster(const Cluster& other) = default;
    explicit inline Cluster(const DendrogramNode& node):
            _indices(std::make_shared<long[]>(node.count)),
            _count(node.count),
            _weight(node.weight),
            _spread(node.mergeDistance) {}

    inline long& operator[](long i) { return _indices[i]; }
    inline long operator[](long i) const { return _indices[i]; }

    [[nodiscard]] inline long count() const { return _count; }
    [[nodiscard]] inline long segmentCount() const { return _count; }
    [[nodiscard]] inline float weight() const { return _weight; }
    [[nodiscard]] inline float spread() const { return _spread; }

    [[nodiscard]] inline long front() const { return _indices[0]; }
    [[nodiscard]] inline long back() const { return _indices[_count - 1]; }
    
    [[nodiscard]] inline Iterator begin() const { return Iterator(_indices.get()); }
    [[nodiscard]] inline Iterator end() const { return Iterator(_indices.get() + _count); }
    
    [[nodiscard]] inline std::span<long> indices() const {
        return std::span<long>(_indices.get(), _indices.get() + _count);
    }

    inline Cluster& operator=(const Cluster& cluster) = default;
    
    inline Cluster& operator=(Cluster&& cluster) noexcept {
        this->_indices = std::move(cluster._indices);
        this->_count = cluster._count;
        this->_spread = cluster._spread;
        this->_weight = cluster._weight;
        
        cluster._count = 0;
        return *this;
    }
};
