//
//  SentencePieceBridge.h
//  C bridge for the SentencePiece C++ library
//

#ifndef SentencePieceBridge_h
#define SentencePieceBridge_h

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to SentencePiece processor
typedef void* SentencePieceProcessor;

// Create and load a SentencePiece model
SentencePieceProcessor sentencepiece_create(const char* model_path);

// Encode text to IDs. Returns the number of IDs written to `ids`.
int sentencepiece_encode_as_ids(SentencePieceProcessor processor,
                               const char* text,
                               int** ids);

// Decode IDs back to text. Caller owns the returned buffer.
char* sentencepiece_decode_ids(SentencePieceProcessor processor, const int* ids, int num_ids);

// Vocabulary helpers
int sentencepiece_get_piece_size(SentencePieceProcessor processor);
int sentencepiece_piece_to_id(SentencePieceProcessor processor, const char* piece);
const char* sentencepiece_id_to_piece(SentencePieceProcessor processor, int id);
float sentencepiece_get_score(SentencePieceProcessor processor, int id);

// Free allocated memory and destroy processor
void sentencepiece_free_ids(int* ids);
void sentencepiece_destroy(SentencePieceProcessor processor);

#ifdef __cplusplus
}
#endif

#endif /* SentencePieceBridge_h */
