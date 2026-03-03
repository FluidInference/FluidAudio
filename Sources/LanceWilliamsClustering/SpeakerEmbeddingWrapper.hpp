//
// Created by Benjamin Lee on 1/26/26.
//
#pragma once

#include <cstdint>
#include <vector>
#include <span>


class SpeakerEmbeddingWrapper {
public:
    using Segment = uint64_t;
    static constexpr std::size_t dims = 192;
    
    SpeakerEmbeddingWrapper() = default;
    SpeakerEmbeddingWrapper(const SpeakerEmbeddingWrapper& other);
    SpeakerEmbeddingWrapper(SpeakerEmbeddingWrapper&& other) noexcept;

    /**
     * @brief Construct a new SpeakerEmbedding
     * @param buffer Pointer to the embedding vector's buffer
     * @param weight Speaker embedding weight. Used as a metric of embedding quality
     * @param isOutlier Whether this embedding represents an outlier
     * @param segments Pointer to the start of the segments array
     * @param segmentCount Number of segments that this embedding owns
     */
    explicit SpeakerEmbeddingWrapper(float* buffer,
                                     float weight,
                                     bool isOutlier,
                                     const Segment* segments,
                                     long segmentCount);
    
    /**
     * @brief Construct a new SpeakerEmbedding
     * @param buffer Pointer to the embedding vector's buffer
     */
    explicit SpeakerEmbeddingWrapper(float* buffer);
    
    /** 
     * @brief Get the squared L2 distance to another embedding
     * @param other Another embedding 
     * @return |a - b|^2
     */
    [[nodiscard]] float squaredDistanceTo(const SpeakerEmbeddingWrapper& other) const;

    /** 
     * @brief Get the weighted squared L2 distance to another embedding
     * @param other Another embedding 
     * @return |a - b|^2
     */
    [[nodiscard]] float wardDistanceTo(const SpeakerEmbeddingWrapper& other) const;

    /** 
     * @brief Get the cosine distance to another embedding
     * @param other Another embedding 
     * @return max(0, 1 - (a • b) / (|a||b|))
     */
    [[nodiscard]] float cosineDistanceTo(const SpeakerEmbeddingWrapper& other) const;

    /** 
     * @brief Get the cosine dissimilarity to another embedding, assuming both are unit vectors
     * @param other Another unit embedding 
     * @return max(0, 1 - a • b)
     */
    [[nodiscard]] float unitCosineDistanceTo(const SpeakerEmbeddingWrapper& other) const;

    /** 
     * @brief Get the dot product with another embedding
     * @param other Another embedding 
     * @return Dot product with `other` if both are active
     */
    [[nodiscard]] float dot(const SpeakerEmbeddingWrapper& other) const;
    
    // Normalize this embedding vector in place
    inline SpeakerEmbeddingWrapper& normalizedInPlace() { return rescaledInPlaceToLength(1.f); }
    
    /** 
     * @brief Rescale the embedding _vector to a new length
     * @param newLength New embedding vector length
     * @returns: This embedding
     */
    SpeakerEmbeddingWrapper& rescaledInPlaceToLength(float newLength);
    
    // Get weight
    [[nodiscard]] inline float weight() const { return _weight; }
    
    // Get weight reference
    [[nodiscard]] inline float& weight() { return _weight; }
    
    // Whether this represents an outlier embedding
    [[nodiscard]] inline bool isOutlier() const { return _isOutlier; }
    
    // Get reference to outlier flag
    [[nodiscard]] inline bool& isOutlier() { return _isOutlier; }
    
    // Get raw pointer to the embedding vector
    [[nodiscard]] inline float* vector() const { return _vector; }
    
    // Get segment count
    [[nodiscard]] inline long segmentCount() const {
        return static_cast<long>(_segments.size());
    }
    
    // Get segments
    [[nodiscard]] inline std::span<const Segment> segments() const {
        return { _segments.begin(), _segments.end() };
    }
    
    // Get reference to segments
    [[nodiscard]] inline std::vector<Segment>& segments() { return _segments; }
    
    // Embedding vector norm
    [[nodiscard]] float norm() const;
    
    // Embedding vector norm squared
    [[nodiscard]] float normSquared() const;
    
    // Check if this embedding holds a null buffer
    [[nodiscard]] inline bool isNull() const { return _vector == nullptr; }
    
    // Check if this embedding holds a valid buffer
    [[nodiscard]] inline bool isValid() const { return _vector != nullptr; }
    
    // Copy the contents of the other vector, but preserve the pointer
    void setFrom(const SpeakerEmbeddingWrapper& other);
    
    // Copy the contents of the other vector, but preserve the pointer
    void setFrom(SpeakerEmbeddingWrapper&& other);
    
    bool operator==(const SpeakerEmbeddingWrapper& other) const;
    
    SpeakerEmbeddingWrapper& operator=(const SpeakerEmbeddingWrapper& other);
    SpeakerEmbeddingWrapper& operator=(SpeakerEmbeddingWrapper&& other) noexcept;
    SpeakerEmbeddingWrapper& operator*=(float scalar);
    SpeakerEmbeddingWrapper& operator/=(float scalar);

protected:
    std::vector<Segment> _segments{};
    float* _vector = nullptr;
    float _weight = 0.f;
    bool _isOutlier = false;
};
