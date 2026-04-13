#!/bin/bash
# Run all Parakeet model benchmarks (100 files each) with sleep prevention.
#
# Benchmarks:
#   1. ASR v3            — parakeet-tdt-0.6b-v3 on LibriSpeech test-clean
#   2. ASR v2            — parakeet-tdt-0.6b-v2 on LibriSpeech test-clean
#   3. ASR tdt-ctc-110m  — parakeet-tdt-ctc-110m on LibriSpeech test-clean
#   4. CTC custom vocab  — ctc-earnings-benchmark (v2 TDT + CTC 110m keyword spotting)
#   5. EOU streaming     — parakeet-eou 320ms on LibriSpeech test-clean
#   6. Nemotron streaming — nemotron 1120ms on LibriSpeech test-clean
#   7. TDT Japanese      — parakeet-tdt-ja on JSUT dataset
#   8. CTC Chinese       — parakeet-ctc-zh-cn on THCHS-30 dataset
#
# Usage:
#   ./Scripts/parakeet_subset_benchmark.sh              # verify + run
#   ./Scripts/parakeet_subset_benchmark.sh --download    # download missing assets, then exit
#
# The script verifies all models and dataset files exist locally before running.
# If anything is missing it will tell you exactly what and exit (unless --download).
# Uses caffeinate to prevent sleep so you can close the lid.
# Results are saved to benchmark_results/ with timestamps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.log"
MAX_FILES=100
SUBSET="test-clean"

MODELS_DIR="$HOME/Library/Application Support/FluidAudio/Models"
DATASETS_DIR="$HOME/Library/Application Support/FluidAudio/Datasets"
EARNINGS_DIR="$HOME/Library/Application Support/FluidAudio/earnings22-kws/test-dataset"

mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Verify local assets
# ---------------------------------------------------------------------------
verify_assets() {
    local missing=0

    # --- Parakeet v3 ---
    local v3_dir="$MODELS_DIR/parakeet-tdt-0.6b-v3"
    for f in Preprocessor.mlmodelc Encoder.mlmodelc Decoder.mlmodelc JointDecision.mlmodelc parakeet_vocab.json; do
        if [[ ! -e "$v3_dir/$f" ]]; then
            log "MISSING  v3: $v3_dir/$f"
            missing=1
        fi
    done

    # --- Parakeet v2 (folder may have -coreml suffix) ---
    local v2_dir=""
    if [[ -d "$MODELS_DIR/parakeet-tdt-0.6b-v2-coreml" ]]; then
        v2_dir="$MODELS_DIR/parakeet-tdt-0.6b-v2-coreml"
    elif [[ -d "$MODELS_DIR/parakeet-tdt-0.6b-v2" ]]; then
        v2_dir="$MODELS_DIR/parakeet-tdt-0.6b-v2"
    fi
    if [[ -z "$v2_dir" ]]; then
        log "MISSING  v2: no parakeet-tdt-0.6b-v2* directory found"
        missing=1
    else
        for f in Preprocessor.mlmodelc Encoder.mlmodelc Decoder.mlmodelc JointDecision.mlmodelc parakeet_vocab.json; do
            if [[ ! -e "$v2_dir/$f" ]]; then
                log "MISSING  v2: $v2_dir/$f"
                missing=1
            fi
        done
    fi

    # --- TDT-CTC-110M (fused: no separate Encoder) ---
    local tdt_ctc_dir="$MODELS_DIR/parakeet-tdt-ctc-110m"
    for f in Preprocessor.mlmodelc Decoder.mlmodelc JointDecision.mlmodelc parakeet_vocab.json; do
        if [[ ! -e "$tdt_ctc_dir/$f" ]]; then
            log "MISSING  tdt-ctc-110m: $tdt_ctc_dir/$f"
            missing=1
        fi
    done

    # --- CTC 110M model (for custom vocabulary / keyword spotting) ---
    local ctc_dir="$MODELS_DIR/parakeet-ctc-110m-coreml"
    for f in MelSpectrogram.mlmodelc AudioEncoder.mlmodelc vocab.json; do
        if [[ ! -e "$ctc_dir/$f" ]]; then
            log "MISSING  ctc-110m: $ctc_dir/$f"
            missing=1
        fi
    done

    # --- EOU streaming models (320ms chunks) ---
    local eou_dir="$MODELS_DIR/parakeet-eou-streaming/320ms"
    if [[ ! -d "$eou_dir" ]]; then
        log "MISSING  eou-320ms: $eou_dir"
        missing=1
    fi

    # --- Nemotron models (uses v3 encoder + nemotron-specific models) ---
    # Nemotron reuses the v3 models directory; no separate check needed beyond v3 above.

    # --- Japanese TDT (hybrid: CTC preprocessor/encoder + TDT decoder/joint) ---
    local ja_dir="$MODELS_DIR/parakeet-tdt-ja"
    for f in Preprocessor.mlmodelc Encoder.mlmodelc Decoderv2.mlmodelc Jointerv2.mlmodelc vocab.json; do
        if [[ ! -e "$ja_dir/$f" ]]; then
            log "MISSING  tdt-ja: $ja_dir/$f"
            missing=1
        fi
    done

    # --- Chinese CTC ---
    local zh_dir="$MODELS_DIR/parakeet-ctc-zh-cn"
    for f in Preprocessor.mlmodelc Decoder.mlmodelc vocab.json; do
        if [[ ! -e "$zh_dir/$f" ]]; then
            log "MISSING  ctc-zh-cn: $zh_dir/$f"
            missing=1
        fi
    done
    # Check that at least one encoder variant exists (int8 or fp32)
    if [[ ! -e "$zh_dir/Encoder-v2-int8.mlmodelc" ]] && [[ ! -e "$zh_dir/Encoder-v1-fp32.mlmodelc" ]]; then
        log "MISSING  ctc-zh-cn: $zh_dir/Encoder-v2-int8.mlmodelc or Encoder-v1-fp32.mlmodelc"
        missing=1
    fi

    # --- LibriSpeech test-clean ---
    local ls_dir="$DATASETS_DIR/LibriSpeech/$SUBSET"
    local trans_count
    trans_count=$(find "$ls_dir" -name "*.trans.txt" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$trans_count" -lt 5 ]]; then
        log "MISSING  LibriSpeech $SUBSET: found $trans_count transcript files (need >= 5)"
        missing=1
    fi

    # --- Earnings22 KWS dataset ---
    local earnings_wav_count
    earnings_wav_count=$(find "$EARNINGS_DIR" -maxdepth 1 -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$earnings_wav_count" -lt 10 ]]; then
        log "MISSING  Earnings22 KWS: found $earnings_wav_count wav files (need >= 10)"
        missing=1
    fi

    # --- JSUT Japanese dataset ---
    local jsut_dir="$DATASETS_DIR/JSUT-basic5000"
    if [[ ! -f "$jsut_dir/metadata.jsonl" ]]; then
        log "MISSING  JSUT: $jsut_dir/metadata.jsonl"
        missing=1
    fi

    # --- THCHS-30 Chinese dataset ---
    local thchs_dir="$DATASETS_DIR/THCHS-30"
    if [[ ! -f "$thchs_dir/metadata.jsonl" ]]; then
        log "MISSING  THCHS-30: $thchs_dir/metadata.jsonl"
        missing=1
    fi

    return $missing
}

# ---------------------------------------------------------------------------
# Phase 1: --download  (verify first, download only what's missing)
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--download" ]]; then
    log "=== Checking local assets ==="

    if verify_assets; then
        log "All models and datasets already present locally. Nothing to download."
        exit 0
    fi

    log "Some assets are missing — downloading..."

    log "Building release binary..."
    cd "$PROJECT_DIR" && swift build -c release 2>&1 | tail -1 | tee -a "$LOG_FILE"
    CLI="$PROJECT_DIR/.build/release/fluidaudiocli"

    log "Downloading LibriSpeech $SUBSET dataset..."
    "$CLI" download --dataset "librispeech-$SUBSET" 2>&1 | tee -a "$LOG_FILE"

    log "Downloading Earnings22 KWS dataset..."
    "$CLI" download --dataset earnings22-kws 2>&1 | tee -a "$LOG_FILE"

    log "Pre-loading Parakeet v3 models (triggers download if missing)..."
    "$CLI" asr-benchmark --model-version v3 --subset "$SUBSET" --max-files 1 \
        --output "$RESULTS_DIR/warmup_v3.json" 2>&1 | tee -a "$LOG_FILE"

    log "Pre-loading Parakeet v2 models..."
    "$CLI" asr-benchmark --model-version v2 --subset "$SUBSET" --max-files 1 \
        --output "$RESULTS_DIR/warmup_v2.json" 2>&1 | tee -a "$LOG_FILE"

    log "Pre-loading CTC earnings models..."
    "$CLI" ctc-earnings-benchmark --max-files 1 --auto-download \
        --output "$RESULTS_DIR/warmup_ctc.json" 2>&1 | tee -a "$LOG_FILE"

    log "Pre-loading EOU streaming models..."
    "$CLI" parakeet-eou --benchmark --chunk-size 320 --max-files 1 \
        --output "$RESULTS_DIR/warmup_eou.json" 2>&1 | tee -a "$LOG_FILE"

    log "Pre-loading Nemotron streaming models..."
    "$CLI" nemotron-benchmark --max-files 1 2>&1 | tee -a "$LOG_FILE"

    log "Pre-loading Japanese TDT models and JSUT dataset..."
    "$CLI" ja-benchmark --decoder tdt --dataset jsut --samples 1 --auto-download \
        --output "$RESULTS_DIR/warmup_ja.json" 2>&1 | tee -a "$LOG_FILE"

    log "Pre-loading Chinese CTC models and THCHS-30 dataset..."
    "$CLI" ctc-zh-cn-benchmark --samples 1 --auto-download \
        --output "$RESULTS_DIR/warmup_zh.json" 2>&1 | tee -a "$LOG_FILE"

    rm -f "$RESULTS_DIR"/warmup_*.json /tmp/nemotron_*_benchmark.json
    log "=== Downloads complete ==="
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: Run benchmarks (offline-safe, sleep-prevented)
# ---------------------------------------------------------------------------
log "=== Verifying local assets before offline run ==="
if ! verify_assets; then
    log ""
    log "ERROR: Missing assets — cannot run offline."
    log "Run with --download first while connected to the internet:"
    log "  ./Scripts/parakeet_subset_benchmark.sh --download"
    exit 1
fi
log "All assets verified locally."

log "=== Parakeet benchmark suite: $MAX_FILES files x 8 benchmarks ==="
log "Results directory: $RESULTS_DIR"

cd "$PROJECT_DIR"

# Build release if not already built
if [[ ! -x ".build/release/fluidaudiocli" ]]; then
    log "Building release binary..."
    swift build -c release 2>&1 | tail -1 | tee -a "$LOG_FILE"
fi
CLI="$PROJECT_DIR/.build/release/fluidaudiocli"

# caffeinate -s: prevent sleep even on AC power / lid closed
# caffeinate -i: prevent idle sleep
# We wrap the entire benchmark suite so caffeinate dies when the script ends.
caffeinate -si -w $$ &
CAFFEINATE_PID=$!
log "caffeinate started (PID $CAFFEINATE_PID) — safe to close the lid"

run_asr_benchmark() {
    local model_version="$1"
    local label="$2"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"

    log "--- $label: starting ($MAX_FILES files, $SUBSET) ---"
    local start_time=$(date +%s)

    "$CLI" asr-benchmark \
        --model-version "$model_version" \
        --subset "$SUBSET" \
        --max-files "$MAX_FILES" \
        --no-auto-download \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

run_ctc_earnings_benchmark() {
    local label="ctc_earnings_vocab"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"

    log "--- $label: starting ($MAX_FILES files, v2 TDT + CTC keyword spotting) ---"
    local start_time=$(date +%s)

    # TDT v2 is used for transcription to match benchmarks100.md baseline
    "$CLI" ctc-earnings-benchmark \
        --ctc-variant 110m \
        --max-files "$MAX_FILES" \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

run_eou_benchmark() {
    local label="eou_320ms"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"

    log "--- $label: starting ($MAX_FILES files, $SUBSET, 320ms chunks) ---"
    local start_time=$(date +%s)

    "$CLI" parakeet-eou \
        --benchmark \
        --chunk-size 320 \
        --max-files "$MAX_FILES" \
        --use-cache \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

run_nemotron_benchmark() {
    local label="nemotron_1120ms"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"

    log "--- $label: starting ($MAX_FILES files, $SUBSET, 1120ms chunks) ---"
    local start_time=$(date +%s)

    "$CLI" nemotron-benchmark \
        --max-files "$MAX_FILES" \
        2>&1 | tee -a "$LOG_FILE"

    # Nemotron writes to /tmp; copy to our results dir
    local tmp_file="/tmp/nemotron_1120ms_benchmark.json"
    if [[ -f "$tmp_file" ]]; then
        cp "$tmp_file" "$output_file"
    fi

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

run_ja_benchmark() {
    local label="parakeet_tdt_ja"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"

    log "--- $label: starting ($MAX_FILES files, JSUT dataset, TDT decoder) ---"
    local start_time=$(date +%s)

    "$CLI" ja-benchmark \
        --decoder tdt \
        --dataset jsut \
        --samples "$MAX_FILES" \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

run_zh_benchmark() {
    local label="parakeet_ctc_zh_cn"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"

    log "--- $label: starting ($MAX_FILES files, THCHS-30 dataset) ---"
    local start_time=$(date +%s)

    "$CLI" ctc-zh-cn-benchmark \
        --samples "$MAX_FILES" \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

SUITE_START=$(date +%s)

run_asr_benchmark "v3"            "parakeet_v3"
run_asr_benchmark "v2"            "parakeet_v2"
run_asr_benchmark "tdt-ctc-110m"  "parakeet_tdt_ctc_110m"
run_ctc_earnings_benchmark
run_eou_benchmark
run_nemotron_benchmark
run_ja_benchmark
run_zh_benchmark

SUITE_END=$(date +%s)
SUITE_ELAPSED=$(( SUITE_END - SUITE_START ))

log "=== All benchmarks complete in ${SUITE_ELAPSED}s ==="
log "Results:"
ls -lh "$RESULTS_DIR"/*_${TIMESTAMP}.json 2>/dev/null | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Compare WER against benchmarks100.md baselines
# ---------------------------------------------------------------------------
# Baselines from Documentation/ASR/benchmarks100.md (main column)
BASELINE_V3_WER="2.6"
BASELINE_V2_WER="3.8"
BASELINE_TDT_CTC_WER="3.6"
BASELINE_EARNINGS_WER="16.54"
BASELINE_EOU_WER="7.11"
BASELINE_NEMOTRON_WER="1.99"
BASELINE_JA_CER="7.77"
BASELINE_ZH_CER="8.37"

extract_wer() {
    local json_file="$1"
    local field="$2"
    if [[ -f "$json_file" ]]; then
        python3 -c "import json,sys; d=json.load(open('$json_file')); print(round(d['summary']['$field']*100, 2))" 2>/dev/null || echo "N/A"
    else
        echo "N/A"
    fi
}

# For JSON fields that already store WER as a percentage (not decimal)
extract_wer_pct() {
    local json_file="$1"
    local section="$2"
    local field="$3"
    if [[ -f "$json_file" ]]; then
        if [[ -n "$section" ]]; then
            python3 -c "import json; d=json.load(open('$json_file')); print(round(d['$section']['$field'], 2))" 2>/dev/null || echo "N/A"
        else
            python3 -c "import json; d=json.load(open('$json_file')); print(round(d['$field'], 2))" 2>/dev/null || echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

V3_FILE="$RESULTS_DIR/parakeet_v3_${TIMESTAMP}.json"
V2_FILE="$RESULTS_DIR/parakeet_v2_${TIMESTAMP}.json"
TDT_CTC_FILE="$RESULTS_DIR/parakeet_tdt_ctc_110m_${TIMESTAMP}.json"
EARNINGS_FILE="$RESULTS_DIR/ctc_earnings_vocab_${TIMESTAMP}.json"
EOU_FILE="$RESULTS_DIR/eou_320ms_${TIMESTAMP}.json"
NEMOTRON_FILE="$RESULTS_DIR/nemotron_1120ms_${TIMESTAMP}.json"
JA_FILE="$RESULTS_DIR/parakeet_tdt_ja_${TIMESTAMP}.json"
ZH_FILE="$RESULTS_DIR/parakeet_ctc_zh_cn_${TIMESTAMP}.json"

V3_WER=$(extract_wer "$V3_FILE" "averageWER")
V2_WER=$(extract_wer "$V2_FILE" "averageWER")
TDT_CTC_WER=$(extract_wer "$TDT_CTC_FILE" "averageWER")
EARNINGS_WER=$(extract_wer_pct "$EARNINGS_FILE" "summary" "avgWer")
EOU_WER=$(extract_wer "$EOU_FILE" "averageWER")
NEMOTRON_WER=$(extract_wer_pct "$NEMOTRON_FILE" "" "wer")
JA_CER=$(extract_wer "$JA_FILE" "mean_cer")
ZH_CER=$(extract_wer "$ZH_FILE" "mean_cer")

log ""
log "=== WER Comparison vs benchmarks100.md baselines ==="
log ""
printf "%-25s %10s %10s %10s\n" "Model" "Baseline" "Current" "Delta" | tee -a "$LOG_FILE"
printf "%-25s %10s %10s %10s\n" "-------------------------" "----------" "----------" "----------" | tee -a "$LOG_FILE"

compare_wer() {
    local label="$1" baseline="$2" current="$3"
    if [[ "$current" == "N/A" ]]; then
        printf "%-25s %9s%% %10s %10s\n" "$label" "$baseline" "N/A" "—" | tee -a "$LOG_FILE"
        return
    fi
    local delta
    delta=$(python3 -c "print(f'{$current - $baseline:+.2f}')" 2>/dev/null || echo "?")
    local marker=""
    local regression
    regression=$(python3 -c "print('YES' if $current > $baseline + 0.3 else 'NO')" 2>/dev/null || echo "NO")
    if [[ "$regression" == "YES" ]]; then
        marker=" ← REGRESSION"
    fi
    printf "%-25s %9s%% %9s%% %9s%%%s\n" "$label" "$baseline" "$current" "$delta" "$marker" | tee -a "$LOG_FILE"
}

compare_wer "Parakeet TDT v3 (0.6B)" "$BASELINE_V3_WER" "$V3_WER"
compare_wer "Parakeet TDT v2 (0.6B)" "$BASELINE_V2_WER" "$V2_WER"
compare_wer "CTC-TDT 110M"           "$BASELINE_TDT_CTC_WER" "$TDT_CTC_WER"
compare_wer "CTC Earnings"            "$BASELINE_EARNINGS_WER" "$EARNINGS_WER"
compare_wer "EOU 320ms (120M)"        "$BASELINE_EOU_WER" "$EOU_WER"
compare_wer "Nemotron 1120ms (0.6B)"  "$BASELINE_NEMOTRON_WER" "$NEMOTRON_WER"

log ""
log "=== CER Comparison (Character Error Rate for non-English) ==="
log ""
printf "%-25s %10s %10s %10s\n" "Model" "Baseline" "Current" "Delta" | tee -a "$LOG_FILE"
printf "%-25s %10s %10s %10s\n" "-------------------------" "----------" "----------" "----------" | tee -a "$LOG_FILE"

compare_wer "TDT Japanese (0.6B)" "$BASELINE_JA_CER" "$JA_CER"
if [[ "$BASELINE_ZH_CER" != "TBD" ]]; then
    compare_wer "CTC Chinese (0.6B)" "$BASELINE_ZH_CER" "$ZH_CER"
else
    printf "%-25s %10s %10s %10s\n" "CTC Chinese (0.6B)" "TBD" "$ZH_CER" "—" | tee -a "$LOG_FILE"
fi

log ""

# Check for any regressions (>0.3% WER/CER increase)
ANY_REGRESSION=$(python3 -c "
baselines = [($BASELINE_V3_WER, '$V3_WER'), ($BASELINE_V2_WER, '$V2_WER'), ($BASELINE_TDT_CTC_WER, '$TDT_CTC_WER'), ($BASELINE_EARNINGS_WER, '$EARNINGS_WER'), ($BASELINE_EOU_WER, '$EOU_WER'), ($BASELINE_NEMOTRON_WER, '$NEMOTRON_WER'), ($BASELINE_JA_CER, '$JA_CER')]
if '$BASELINE_ZH_CER' != 'TBD':
    baselines.append(($BASELINE_ZH_CER, '$ZH_CER'))
for b, c in baselines:
    if c != 'N/A' and float(c) > b + 0.3:
        print('YES'); exit()
print('NO')
" 2>/dev/null || echo "NO")

if [[ "$ANY_REGRESSION" == "YES" ]]; then
    log "⚠ WER/CER REGRESSION DETECTED — investigate before merging"
else
    log "✓ No WER/CER regressions (all within 0.3% of baseline)"
fi

# caffeinate will exit automatically since the parent process ($$) exits
