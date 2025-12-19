//
//  SentencePieceBridge.cpp
//  C++ implementation exposing a C interface for Swift
//

#include "SentencePieceBridge.h"

#ifdef __has_include
#if __has_include(<SentencePiece/sentencepiece_processor.h>)
#include <SentencePiece/sentencepiece_processor.h>
#elif __has_include("sentencepiece_processor.h")
#include "sentencepiece_processor.h"
#else
#include <sentencepiece_processor.h>
#endif
#else
#include <sentencepiece_processor.h>
#endif

#include <cstring>
#include <string>
#include <vector>

extern "C" {

SentencePieceProcessor sentencepiece_create(const char* model_path) {
    auto* processor = new sentencepiece::SentencePieceProcessor();
    const auto status = processor->Load(model_path);
    if (!status.ok()) {
        delete processor;
        return nullptr;
    }
    return processor;
}

int sentencepiece_encode_as_ids(SentencePieceProcessor processor,
                               const char* text,
                               int** ids) {
    if (!processor || !text || !ids) return 0;

    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    std::vector<int> ids_vec;

    const auto status = sp->Encode(text, &ids_vec);
    if (!status.ok()) return 0;

    // Allocate array of ints
    *ids = static_cast<int*>(malloc(ids_vec.size() * sizeof(int)));
    if (*ids == nullptr) return 0;

    std::memcpy(*ids, ids_vec.data(), ids_vec.size() * sizeof(int));
    return static_cast<int>(ids_vec.size());
}

char* sentencepiece_decode_ids(SentencePieceProcessor processor, const int* ids, int num_ids) {
    if (!processor || !ids || num_ids <= 0) return nullptr;

    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    std::vector<int> ids_vec(ids, ids + num_ids);
    std::string decoded;

    const auto status = sp->Decode(ids_vec, &decoded);
    if (!status.ok()) return nullptr;

    return strdup(decoded.c_str());
}

int sentencepiece_get_piece_size(SentencePieceProcessor processor) {
    if (!processor) return 0;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    return sp->GetPieceSize();
}

int sentencepiece_piece_to_id(SentencePieceProcessor processor, const char* piece) {
    if (!processor || !piece) return -1;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    return sp->PieceToId(piece);
}

const char* sentencepiece_id_to_piece(SentencePieceProcessor processor, int id) {
    if (!processor) return nullptr;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    static thread_local std::string piece;
    piece = sp->IdToPiece(id);
    return piece.c_str();
}

float sentencepiece_get_score(SentencePieceProcessor processor, int id) {
    if (!processor) return 0.0f;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    return sp->GetScore(id);
}

void sentencepiece_free_ids(int* ids) {
    if (ids) {
        free(ids);
    }
}

void sentencepiece_destroy(SentencePieceProcessor processor) {
    if (processor) {
        delete static_cast<sentencepiece::SentencePieceProcessor*>(processor);
    }
}

} // extern "C"
