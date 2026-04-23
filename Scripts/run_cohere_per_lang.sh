#!/usr/bin/env bash
# Run the Cohere hybrid (INT8 encoder + FP16 decoder) FLEURS benchmark one
# language at a time. Each language writes its own JSON on completion, which
# gives incremental results and resumability (existing JSONs are skipped).
#
# Environment overrides:
#   BIN        path to fluidaudiocli binary  (default: ./.build/release/fluidaudiocli)
#   MODEL_DIR  Cohere model directory         (required, or set via caller)
#   MAX_FILES  cap per language               (default: unset = full FLEURS split)
#   OUT_TAG   subdir under benchmark_results/ (default: cohere_per_lang)
#   LANGS     space-separated lang codes      (default: all 14 Cohere languages)
#
# Example:
#   MODEL_DIR=/tmp/cohere-mixed MAX_FILES=100 OUT_TAG=cohere_max100 \
#     Scripts/run_cohere_per_lang.sh
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-${REPO_ROOT}/.build/release/fluidaudiocli}"
MODEL_DIR="${MODEL_DIR:-}"
MAX_FILES="${MAX_FILES:-}"
OUT_TAG="${OUT_TAG:-cohere_per_lang}"
OUT_DIR="${REPO_ROOT}/benchmark_results/${OUT_TAG}"
MASTER_LOG="${OUT_DIR}/_master.log"

if [[ -z "${MODEL_DIR}" ]]; then
    echo "ERROR: MODEL_DIR is not set. Point it at a directory containing" >&2
    echo "       cohere_encoder.mlmodelc, cohere_decoder_cache_external.mlmodelc," >&2
    echo "       and vocab.json." >&2
    exit 2
fi
if [[ ! -x "${BIN}" ]]; then
    echo "ERROR: fluidaudiocli not found at ${BIN}." >&2
    echo "       Build it first: swift build -c release --product fluidaudiocli" >&2
    exit 2
fi

# Default = 14 Cohere-supported languages (order matches Figure 4 grouping:
# English first, then Europe, then Asia).
DEFAULT_LANGS="en_us fr_fr de_de es_419 it_it pt_br nl_nl pl_pl el_gr ar_eg ja_jp cmn_hans_cn ko_kr vi_vn"
read -r -a LANGS <<< "${LANGS:-${DEFAULT_LANGS}}"

mkdir -p "${OUT_DIR}"

echo "=== Cohere hybrid per-language benchmark ===" | tee "${MASTER_LOG}"
echo "Started:  $(date)" | tee -a "${MASTER_LOG}"
echo "Binary:   ${BIN}" | tee -a "${MASTER_LOG}"
echo "ModelDir: ${MODEL_DIR}" | tee -a "${MASTER_LOG}"
echo "OutDir:   ${OUT_DIR}" | tee -a "${MASTER_LOG}"
echo "MaxFiles: ${MAX_FILES:-all}" | tee -a "${MASTER_LOG}"
echo "Langs:    ${LANGS[*]}" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"

for lang in "${LANGS[@]}"; do
    OUT="${OUT_DIR}/${lang}.json"
    LOG="${OUT_DIR}/${lang}.log"

    if [[ -f "${OUT}" ]]; then
        echo "[$(date '+%H:%M:%S')] SKIP ${lang} (already have ${OUT})" | tee -a "${MASTER_LOG}"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] START ${lang}" | tee -a "${MASTER_LOG}"
    START_TS=$(date +%s)

    MAX_ARGS=()
    if [[ -n "${MAX_FILES}" ]]; then
        MAX_ARGS=(--max-files "${MAX_FILES}")
    fi

    "${BIN}" cohere-benchmark \
        --model-dir "${MODEL_DIR}" \
        --languages "${lang}" \
        --auto-download \
        ${MAX_ARGS[@]+"${MAX_ARGS[@]}"} \
        --output "${OUT}" \
        > "${LOG}" 2>&1
    RC=$?

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    # JSON presence is the truth (rc can be 139 from a post-write CoreML
    # teardown segfault; results are still valid in that case).
    if [[ -f "${OUT}" ]]; then
        echo "[$(date '+%H:%M:%S')] DONE  ${lang} in ${ELAPSED}s (rc=${RC})" | tee -a "${MASTER_LOG}"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  ${lang} in ${ELAPSED}s (rc=${RC}, no JSON)" | tee -a "${MASTER_LOG}"
    fi
done

echo "" | tee -a "${MASTER_LOG}"
echo "Finished: $(date)" | tee -a "${MASTER_LOG}"
