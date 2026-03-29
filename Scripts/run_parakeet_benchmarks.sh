#!/bin/bash
# Run all Parakeet model benchmarks (100 files each) with sleep prevention.
#
# Benchmarks:
#   1. ASR v3            — parakeet-tdt-0.6b-v3 on LibriSpeech test-clean
#   2. ASR v2            — parakeet-tdt-0.6b-v2 on LibriSpeech test-clean
#   3. ASR tdt-ctc-110m  — parakeet-tdt-ctc-110m on LibriSpeech test-clean
#   4. CTC custom vocab  — ctc-earnings-benchmark (tdt-ctc-110m + CTC 110m keyword spotting)
#
# Usage:
#   ./Scripts/run_parakeet_benchmarks.sh              # verify + run
#   ./Scripts/run_parakeet_benchmarks.sh --download    # download missing assets, then exit
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

    log "Pre-loading TDT-CTC-110M + CTC models..."
    "$CLI" ctc-earnings-benchmark --tdt-version tdt-ctc-110m --max-files 1 --auto-download \
        --output "$RESULTS_DIR/warmup_ctc.json" 2>&1 | tee -a "$LOG_FILE"

    rm -f "$RESULTS_DIR"/warmup_*.json
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
    log "  ./Scripts/run_parakeet_benchmarks.sh --download"
    exit 1
fi
log "All assets verified locally."

log "=== Parakeet benchmark suite: $MAX_FILES files x 4 benchmarks ==="
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

    log "--- $label: starting ($MAX_FILES files, tdt-ctc-110m + CTC keyword spotting) ---"
    local start_time=$(date +%s)

    "$CLI" ctc-earnings-benchmark \
        --tdt-version tdt-ctc-110m \
        --ctc-variant 110m \
        --max-files "$MAX_FILES" \
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

SUITE_END=$(date +%s)
SUITE_ELAPSED=$(( SUITE_END - SUITE_START ))

log "=== All benchmarks complete in ${SUITE_ELAPSED}s ==="
log "Results:"
ls -lh "$RESULTS_DIR"/*_${TIMESTAMP}.json 2>/dev/null | tee -a "$LOG_FILE"

# caffeinate will exit automatically since the parent process ($$) exits
