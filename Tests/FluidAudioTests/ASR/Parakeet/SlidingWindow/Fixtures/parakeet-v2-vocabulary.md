# Parakeet v2 vocabulary fixture

`parakeet-v2-vocabulary.json` is an unmodified copy of the vocabulary published by
FluidInference for NVIDIA's Parakeet TDT 0.6B v2, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Source: https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml/blob/ee09c569f73759e6d44c9bd16766f477b2b36d39/parakeet_vocab.json

SHA-256: `57019fe3c745772ca83a1b048a4bb951cd51329504ea33d4d83316b96e279a97`

The 1,031-entry fixture covers every token below v2's blank ID of 1,024 and
includes additional entries. Local-loading regressions use it without model
downloads; the missing-token test removes one entry in a temporary copy.
