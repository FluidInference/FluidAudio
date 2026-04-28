#!/bin/bash
# Run the FLEURS benchmark across all 24 Parakeet TDT v3 languages for
# local testing / regression checks. 100 samples/language by default.
#
# Usage:
#   ./Scripts/fleurs_parakeet_sub_benchmark.sh           # 100 samples/lang
#   SAMPLES=10 ./Scripts/fleurs_parakeet_sub_benchmark.sh # quick smoke test
#
# FLEURS data and models download automatically if missing. Results land in
# benchmark_results/ as per-language JSON plus a combined summary CSV.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$RESULTS_DIR/fleurs_${TIMESTAMP}.log"
SUMMARY_CSV="$RESULTS_DIR/fleurs_${TIMESTAMP}_summary.csv"
SAMPLES="${SAMPLES:-100}"

LANGUAGES=(
    en_us es_419 it_it fr_fr de_de
    ru_ru nl_nl pl_pl uk_ua sk_sk
    cs_cz bg_bg hr_hr ro_ro fi_fi
    hu_hu sv_se et_ee da_dk lt_lt
    el_gr mt_mt lv_lv sl_si
)

mkdir -p "$RESULTS_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

command -v python3 >/dev/null \
    || { log "ERROR: python3 required for summary extraction"; exit 1; }

cd "$PROJECT_DIR"

log "Building release binary..."
if ! swift build -c release 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: swift build failed"
    exit 1
fi
CLI="$PROJECT_DIR/.build/release/fluidaudiocli"

log "=== FLEURS: $SAMPLES samples x ${#LANGUAGES[@]} languages = $(( SAMPLES * ${#LANGUAGES[@]} )) total ==="

SUITE_START=$(date +%s)

for i in "${!LANGUAGES[@]}"; do
    lang="${LANGUAGES[$i]}"
    output_file="$RESULTS_DIR/fleurs_${lang}_${TIMESTAMP}.json"

    log "[$((i+1))/${#LANGUAGES[@]}] $lang: starting ($SAMPLES samples)"
    start_time=$(date +%s)

    if ! "$CLI" fleurs-benchmark \
        --languages "$lang" \
        --samples "$SAMPLES" \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"; then
        log "WARN: $lang failed — continuing"
    fi

    log "[$((i+1))/${#LANGUAGES[@]}] $lang: done in $(( $(date +%s) - start_time ))s"
done

SUITE_ELAPSED=$(( $(date +%s) - SUITE_START ))
log "=== Suite complete in $((SUITE_ELAPSED / 60))m $((SUITE_ELAPSED % 60))s ==="

log ""
log "=== Summary ($SAMPLES samples per language) ==="
printf 'lang,wer_pct,cer_pct,rtfx\n' > "$SUMMARY_CSV"
printf '%-10s %10s %10s %10s\n' "Language" "WER%" "CER%" "RTFx" | tee -a "$LOG_FILE"

for lang in "${LANGUAGES[@]}"; do
    json_file="$RESULTS_DIR/fleurs_${lang}_${TIMESTAMP}.json"
    row=$(python3 - "$json_file" <<'PY' 2>/dev/null || printf 'N/A,N/A,N/A'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    s = d["summary"]
    print(f"{s['averageWER']*100:.2f},{s['averageCER']*100:.2f},{s['averageRTFx']:.1f}")
except Exception:
    print("N/A,N/A,N/A")
PY
)
    printf '%s,%s\n' "$lang" "$row" >> "$SUMMARY_CSV"

    IFS=',' read -r wer cer rtfx <<< "$row"
    printf '%-10s %9s%% %9s%% %9sx\n' "$lang" "$wer" "$cer" "$rtfx" | tee -a "$LOG_FILE"
done

log ""
log "Results: $RESULTS_DIR"
log "Summary CSV: $SUMMARY_CSV"
