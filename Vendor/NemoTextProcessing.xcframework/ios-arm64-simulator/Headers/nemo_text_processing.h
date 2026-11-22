#ifndef NEMO_TEXT_PROCESSING_H
#define NEMO_TEXT_PROCESSING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Normalize spoken-form text to written form.
 *
 * @param input Null-terminated UTF-8 string of spoken text
 * @return Newly allocated string with written form, or NULL on error.
 *         Must be freed with nemo_free_string().
 *
 * Example:
 *   char* result = nemo_normalize("two hundred");
 *   // result is "200"
 *   nemo_free_string(result);
 */
char* nemo_normalize(const char* input);

/**
 * Normalize a full sentence, replacing spoken-form spans with written form.
 *
 * Unlike nemo_normalize which expects the entire input to be a single expression,
 * this scans for normalizable spans within a larger sentence.
 *
 * @param input Null-terminated UTF-8 string
 * @return Newly allocated string, must be freed with nemo_free_string().
 */
char* nemo_normalize_sentence(const char* input);

/**
 * Normalize a single spoken-form expression with caller-specified options.
 *
 * @param input Null-terminated UTF-8 string
 * @param concat_compound_numbers When non-zero, consecutive 0-99 number
 *        words concatenate (aviation flight-number style — e.g.
 *        "thirty five sixty two" -> "3562", "seven eighty eight" -> "788")
 *        instead of adding.
 * @param disable_bare_second When non-zero, the bare word "second" is NOT
 *        rewritten to "2nd" (issue #22), so phrases like
 *        "give me a second" stay literal. Compound ordinals
 *        ("twenty second" -> "22nd") still convert.
 * @return Newly allocated string, must be freed with nemo_free_string().
 */
char* nemo_normalize_with_options(
    const char* input,
    uint32_t concat_compound_numbers,
    uint32_t disable_bare_second
);

/**
 * Normalize a full sentence with caller-specified options.
 *
 * @param input Null-terminated UTF-8 string
 * @param concat_compound_numbers See nemo_normalize_with_options.
 * @param max_span_tokens Maximum consecutive tokens per span. Pass 0 to
 *        use the library default (16).
 * @param disable_bare_second See nemo_normalize_with_options.
 * @return Newly allocated string, must be freed with nemo_free_string().
 */
char* nemo_normalize_sentence_with_options(
    const char* input,
    uint32_t concat_compound_numbers,
    uint32_t max_span_tokens,
    uint32_t disable_bare_second
);

/**
 * Add a custom spoken-to-written normalization rule.
 * Custom rules have the highest priority, checked before all built-in taggers.
 * If a rule with the same spoken form exists, it is replaced.
 *
 * @param spoken Null-terminated UTF-8 spoken form (case-insensitive)
 * @param written Null-terminated UTF-8 written form
 */
void nemo_add_rule(const char* spoken, const char* written);

/**
 * Remove a custom normalization rule.
 *
 * @param spoken Null-terminated UTF-8 spoken form to remove
 * @return 1 if the rule was found and removed, 0 otherwise
 */
int32_t nemo_remove_rule(const char* spoken);

/**
 * Clear all custom normalization rules.
 */
void nemo_clear_rules(void);

/**
 * Get the number of custom rules currently registered.
 */
uint32_t nemo_rule_count(void);

/**
 * Text Normalization: convert written-form text to spoken form (TTS preprocessing).
 *
 * @param input Null-terminated UTF-8 string of written text
 * @return Newly allocated string with spoken form, or NULL on error.
 *         Must be freed with nemo_free_string().
 *
 * Example:
 *   char* result = nemo_tn_normalize("$5.50");
 *   // result is "five dollars and fifty cents"
 *   nemo_free_string(result);
 */
char* nemo_tn_normalize(const char* input);

/**
 * Text Normalization: normalize a full sentence, replacing written-form spans with spoken form.
 *
 * @param input Null-terminated UTF-8 string
 * @return Newly allocated string, must be freed with nemo_free_string().
 */
char* nemo_tn_normalize_sentence(const char* input);

/**
 * Text Normalization: normalize a full sentence with configurable max span size.
 *
 * @param input Null-terminated UTF-8 string
 * @param max_span_tokens Maximum number of consecutive tokens per span (default 16)
 * @return Newly allocated string, must be freed with nemo_free_string().
 */
char* nemo_tn_normalize_sentence_with_max_span(const char* input, uint32_t max_span_tokens);

/**
 * Text Normalization: convert written-form text to spoken form for a specific language.
 *
 * Supported language codes: "en", "fr", "es", "de", "zh", "hi", "ja".
 * Falls back to English for unrecognized codes.
 *
 * @param input Null-terminated UTF-8 string of written text
 * @param lang Null-terminated language code (e.g. "fr", "de")
 * @return Newly allocated string with spoken form, or NULL on error.
 *         Must be freed with nemo_free_string().
 */
char* nemo_tn_normalize_lang(const char* input, const char* lang);

/**
 * Text Normalization: normalize a full sentence for a specific language.
 *
 * @param input Null-terminated UTF-8 string
 * @param lang Null-terminated language code (e.g. "fr", "de")
 * @return Newly allocated string, must be freed with nemo_free_string().
 */
char* nemo_tn_normalize_sentence_lang(const char* input, const char* lang);

/**
 * Text Normalization: normalize a full sentence for a specific language
 * with configurable max span size.
 *
 * @param input Null-terminated UTF-8 string
 * @param lang Null-terminated language code (e.g. "fr", "de")
 * @param max_span_tokens Maximum number of consecutive tokens per span (default 16)
 * @return Newly allocated string, must be freed with nemo_free_string().
 */
char* nemo_tn_normalize_sentence_with_max_span_lang(const char* input, const char* lang, uint32_t max_span_tokens);

/**
 * Text Normalization via the compiled-FST engine — byte-exact NeMo parity.
 *
 * Supported langs: "en", "zh", "ja", "fr", "es", "de", "hi".
 * Returns NULL if the library was built without the fst-engine feature or the
 * language is unsupported, so callers can fall back to nemo_tn_normalize_lang.
 *
 * @param input Null-terminated UTF-8 string
 * @param lang Null-terminated language code
 * @return Newly allocated string (free with nemo_free_string), or NULL.
 */
char* nemo_tn_fst(const char* input, const char* lang);

/**
 * Free a string allocated by nemo_normalize or nemo_normalize_sentence.
 *
 * @param s Pointer returned by nemo_normalize, or NULL (no-op)
 */
void nemo_free_string(char* s);

/**
 * Get the library version.
 *
 * @return Static version string, do not free.
 */
const char* nemo_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NEMO_TEXT_PROCESSING_H */
