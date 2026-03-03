//
// Created by Benjamin Lee on 1/29/26.
//
#include "EmbeddingDistanceMatrix.hpp"
#include "Dendrogram.hpp"


EmbeddingDistanceMatrix::EmbeddingDistanceMatrix(const LinkagePolicy* linkagePolicy): 
        _matrixStart(nullptr),
        _embeddings(nullptr),
        _linkagePolicy(linkagePolicy) {}

EmbeddingDistanceMatrix::EmbeddingDistanceMatrix(EmbeddingDistanceMatrix&& other) noexcept :
        _matrixStart(other._matrixStart),
        _embeddings(other._embeddings),
        _size(other._size),
        _capacity(other._capacity),
        _matrixEnd(other._matrixEnd),
        _linkagePolicy(other._linkagePolicy)
{
    other._matrixStart = nullptr;
    other._embeddings = nullptr;
    other._matrixEnd = nullptr;
    other._size = 0;
    other._capacity = 0;
}

EmbeddingDistanceMatrix::~EmbeddingDistanceMatrix() {
    delete[] _matrixStart;
    delete[] _embeddings;
}

void EmbeddingDistanceMatrix::reserve(long newCapacity) {
    if (newCapacity <= _capacity)
        return;
    
    // Copy embeddings
    auto newEmbeddings = new SpeakerEmbeddingWrapper[newCapacity]{};
    if (_embeddings != nullptr) {
        for (auto i = 0; i < _size; ++i) {
            newEmbeddings[i] = std::move(_embeddings[i]);
        }
        delete[] _embeddings;
    }
    _embeddings = newEmbeddings;

    // Copy matrix
    auto newMatrixCapacity = newCapacity * (newCapacity - 1) / 2;
    auto newMatrix = new float[newMatrixCapacity];
    if (_matrixStart != nullptr) {
        std::copy(_matrixStart, _matrixEnd, newMatrix);
        delete[] _matrixStart;
    }
    _matrixEnd = newMatrix + (_matrixEnd - _matrixStart);
    _matrixStart = newMatrix;
    
    _capacity = newCapacity;
}


float EmbeddingDistanceMatrix::distance(long row, long col) const {
    // Diagonal elements are always 0 because cosineDistance(E, E) = 0
    if (row == col)
        return 0.f;

    // Data is stored as a lower triangle matrix without the diagonal, so we need col ≤ row.
    if (row < col)
        std::swap(row, col);

    // n(n+1) / 2 - n = n(n-1) / 2
    return _matrixStart[row * (row - 1) / 2 + col];
}


void EmbeddingDistanceMatrix::append(SpeakerEmbeddingWrapper const& embedding) {
    long index, matrixIndex;
    
    index = _size;
    // Ensure there is enough capacity if appending to the end
    if (_size + 1 > _capacity)
        reserve(std::max(_capacity * 2, _size + 1));
    
    _matrixEnd += _size++;
    
    // Update each (index, col)
    matrixIndex = index * (index - 1) / 2;
    for (auto col = 0; col < index; ++col) {
        _matrixStart[matrixIndex++] = _linkagePolicy->distance(embedding, _embeddings[col]);
    }

    _embeddings[index] = embedding;
}


void EmbeddingDistanceMatrix::replace(long index, SpeakerEmbeddingWrapper& embedding) {
    if (_embeddings[index] == embedding) return;
    
    // Update each (index, col)
    auto matrixIndex = index * (index - 1) / 2;

    for (auto i = 0; i < index; ++i) {
        _matrixStart[matrixIndex++] = _linkagePolicy->distance(embedding, _embeddings[i]);
    }

    if (index < _size) {
        // Update each (r, index) squaredDistanceTo for r > index
        // Start at (r, c) = (index + 1, 0)
        matrixIndex = index * (index + 1) / 2;
        for (auto i = index + 1; i < _size; ++i) {
            // This makes column = index in first iteration and increments the index in subsequent ones
            matrixIndex += i - 1;
            _matrixStart[matrixIndex] = _linkagePolicy->distance(embedding, _embeddings[i]);
        }
    }

    _embeddings[index] = embedding;
}

EmbeddingDistanceMatrix& EmbeddingDistanceMatrix::operator=(EmbeddingDistanceMatrix&& other) noexcept {
    delete[] _matrixStart;
    delete[] _embeddings;
    _matrixStart = other._matrixStart;
    _embeddings = other._embeddings;
    _matrixEnd = other._matrixEnd;
    _size = other._size;
    _capacity = other._capacity;
    
    other._matrixStart = nullptr;
    other._embeddings = nullptr;
    other._matrixEnd = nullptr;
    other._size = 0;
    other._capacity = 0;
    return *this;
}

Dendrogram EmbeddingDistanceMatrix::dendrogram() const {
    return Dendrogram(*this);
}
