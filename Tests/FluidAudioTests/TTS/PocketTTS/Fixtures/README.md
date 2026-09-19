# PocketTTS test resource

`pocket_tts_english_tokenizer.model` is the English SentencePiece tokenizer
distributed with the `v2.1/english` FluidInference PocketTTS Core ML pack:

https://huggingface.co/FluidInference/pocket-tts-coreml/blob/main/v2.1/english/constants_bin/tokenizer.model

SHA-256: `d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6`

The fixture is copied unchanged and is used only to verify production
tokenization and chunking behavior without network access during tests.
