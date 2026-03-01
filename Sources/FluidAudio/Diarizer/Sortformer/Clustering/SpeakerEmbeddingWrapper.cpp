//
// Created by Benjamin Lee on 1/26/26.
//

#include <utility>
#include "SpeakerEmbeddingWrapper.hpp"
#include <Accelerate/Accelerate.h>

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(float* buffer,
                                                 float weight,
                                                 bool isOutlier,
                                                 const Segment* segments,
                                                 long segmentCount):
        _vector(buffer),
        _weight(weight),
        _isOutlier(isOutlier),
        _segments(segments, segments + segmentCount) {}

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(float* buffer):
        _vector(buffer) {}

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(const SpeakerEmbeddingWrapper& other):
        _vector(other._vector),
        _weight(other._weight),
        _isOutlier(other._isOutlier),
        _segments(other._segments)
{}

SpeakerEmbeddingWrapper::SpeakerEmbeddingWrapper(SpeakerEmbeddingWrapper&& other) noexcept:
        _vector(other._vector), 
        _weight(other._weight),
        _isOutlier(other._isOutlier),
        _segments(std::move(other._segments))
{}
        
float SpeakerEmbeddingWrapper::squaredDistanceTo(const SpeakerEmbeddingWrapper& other) const {
    float dist;
    vDSP_distancesq(
            this->_vector, 1,
            other._vector, 1,
            &dist,
            SpeakerEmbeddingWrapper::dims
    );
    return dist; 
}

float SpeakerEmbeddingWrapper::wardDistanceTo(const SpeakerEmbeddingWrapper& other) const {
    float dist;
    vDSP_distancesq(
            this->_vector, 1,
            other._vector, 1,
            &dist,
            SpeakerEmbeddingWrapper::dims
    );
    return (this->_weight * other._weight) / (this->_weight + other._weight) * dist;
}

float SpeakerEmbeddingWrapper::cosineDistanceTo(const SpeakerEmbeddingWrapper& other) const {
    float dot;
    vDSP_dotpr(
            this->_vector, 1,
            other._vector, 1,
            &dot,
            SpeakerEmbeddingWrapper::dims
    );
    
    float normSqA, normSqB;
    vDSP_svesq(this->_vector, 1, &normSqA, SpeakerEmbeddingWrapper::dims);
    vDSP_svesq(other._vector, 1, &normSqB, SpeakerEmbeddingWrapper::dims);
    return std::max(0.f, 1.f - dot / std::sqrt(normSqA * normSqB));
}

float SpeakerEmbeddingWrapper::unitCosineDistanceTo(const SpeakerEmbeddingWrapper& other) const {
    float dot;
    vDSP_dotpr(
            this->_vector, 1,
            other._vector, 1,
            &dot,
            SpeakerEmbeddingWrapper::dims
    );

    return std::max(0.f, 1.f - dot);
}

float SpeakerEmbeddingWrapper::dot(const SpeakerEmbeddingWrapper& other) const {
    float dot;
    vDSP_dotpr(
            this->_vector, 1,
            other._vector, 1,
            &dot,
            SpeakerEmbeddingWrapper::dims
    );
    return dot;
}

float SpeakerEmbeddingWrapper::norm() const {
    float normSq;
    vDSP_svesq(this->_vector, 1, &normSq, SpeakerEmbeddingWrapper::dims);
    return std::sqrt(normSq);
}

float SpeakerEmbeddingWrapper::normSquared() const {
    float normSq;
    vDSP_svesq(this->_vector, 1, &normSq, SpeakerEmbeddingWrapper::dims);
    return normSq;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::rescaledInPlaceToLength(float newLength) {
    auto length = norm();
    if (length < 1e-6) return *this;
    return *this *= newLength / length;
}

void SpeakerEmbeddingWrapper::setFrom(const SpeakerEmbeddingWrapper& other) {
    if (this == &other) return;
    this->_weight = other._weight;
    this->_isOutlier = other._isOutlier;
    this->_segments = other._segments;
    if (this->_vector == other._vector)
        return;
    std::copy(other._vector, other._vector + dims, this->_vector);
}

void SpeakerEmbeddingWrapper::setFrom(SpeakerEmbeddingWrapper&& other) {
    this->_weight = other._weight;
    this->_isOutlier = other._isOutlier;
    this->_segments = std::move(other._segments);
    if (this->_vector == other._vector)
        return;
    std::copy(other._vector, other._vector + dims, this->_vector);
}

bool SpeakerEmbeddingWrapper::operator==(const SpeakerEmbeddingWrapper& other) const {
    return this->_vector == other._vector;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::operator=(SpeakerEmbeddingWrapper&& other) noexcept {
    this->_vector = other._vector;
    this->_weight = other._weight;
    this->_isOutlier = other._isOutlier;
    this->_segments = std::move(other._segments);
    return *this;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::operator=(const SpeakerEmbeddingWrapper& other) {
    if (this == &other) return *this;
    this->_vector = other._vector;
    this->_weight = other._weight;
    this->_isOutlier = other._isOutlier;
    this->_segments = other._segments;
    return *this;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::operator*=(float scalar) {
    vDSP_vsmul(
            _vector, 1,
            &scalar,
            _vector, 1,
            SpeakerEmbeddingWrapper::dims
    );
    return *this;
}

SpeakerEmbeddingWrapper& SpeakerEmbeddingWrapper::operator/=(float scalar) {
    vDSP_vsdiv(
            _vector, 1,
            &scalar,
            _vector, 1,
            SpeakerEmbeddingWrapper::dims
    );
    return *this;
}
