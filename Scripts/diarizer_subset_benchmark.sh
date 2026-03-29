#!/bin/bash
# Run all diarizer model benchmarks on AMI SDM with sleep prevention.
#
# Benchmarks:
#   1. Offline (VBx)       — OfflineDiarizerManager, step=0.2, min-seg=1.0
#   2. Streaming (5s)      — DiarizerManager, 5s chunks, 0s overlap, threshold=0.8
#   3. Sortformer          — SortformerDiarizer, NVIDIA high-latency config
#   4. LS-EEND             — LSEENDDiarizer, AMI variant
#
# Usage:
#   ./Scripts/diarizer_subset_benchmark.sh                    # quick run (4 meetings)
#   ./Scripts/diarizer_subset_benchmark.sh --all              # full run (all 16 meetings)
#   ./Scripts/diarizer_subset_benchmark.sh --max-files 8      # custom subset
#   ./Scripts/diarizer_subset_benchmark.sh --download         # download missing assets, then exit
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
LOG_FILE="$RESULTS_DIR/diarizer_benchmark_${TIMESTAMP}.log"

MODELS_DIR="$HOME/Library/Application Support/FluidAudio/Models"
DATASETS_DIR="$HOME/FluidAudioDatasets"
AMI_SDM_DIR="$DATASETS_DIR/ami_official/sdm"
AMI_RTTM_DIR="$DATASETS_DIR/ami_official/rttm"
MAX_FILES=4  # default: quick 4-meeting subset

# AMI SDM has 16 meetings — this is the standard diarization test set.
# Ordered so the first N picks one from each speaker group for maximum diversity.
# Groups: EN2002 (4 speakers), ES2004 (4), IS1009 (4), TS3003 (4)
ALL_AMI_MEETINGS=(
    EN2002a ES2004a IS1009a TS3003a
    EN2002b ES2004b IS1009b TS3003b
    EN2002c ES2004c IS1009c TS3003c
    EN2002d ES2004d IS1009d TS3003d
)

# Parse --all / --max-files <N> from arguments
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
        --all)        MAX_FILES=${#ALL_AMI_MEETINGS[@]} ;;
        --max-files)  MAX_FILES="${args[$((i+1))]}" ; i=$((i+1)) ;;
    esac
done

# Select the subset of meetings to run
AMI_MEETINGS=("${ALL_AMI_MEETINGS[@]:0:$MAX_FILES}")

mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Verify local assets
# ---------------------------------------------------------------------------
verify_assets() {
    local missing=0

    # --- AMI SDM audio files ---
    local wav_count=0
    for meeting in "${AMI_MEETINGS[@]}"; do
        if [[ -f "$AMI_SDM_DIR/${meeting}.Mix-Headset.wav" ]]; then
            wav_count=$((wav_count + 1))
        else
            log "MISSING  AMI SDM: $AMI_SDM_DIR/${meeting}.Mix-Headset.wav"
            missing=1
        fi
    done
    if [[ "$wav_count" -eq 0 ]]; then
        log "MISSING  AMI SDM: no wav files found in $AMI_SDM_DIR"
        missing=1
    fi

    # --- AMI RTTM annotations (downloaded automatically by --auto-download) ---
    local rttm_count=0
    for meeting in "${ALL_AMI_MEETINGS[@]}"; do
        if [[ -f "$AMI_RTTM_DIR/${meeting}.rttm" ]]; then
            rttm_count=$((rttm_count + 1))
        fi
    done
    if [[ "$rttm_count" -eq 0 ]]; then
        log "NOTE     AMI RTTM annotations not found — will be auto-downloaded by CLI"
    fi

    # --- Offline diarizer models (pyannote segmentation + wespeaker embedding) ---
    local diar_dir="$MODELS_DIR/speaker-diarization-coreml"
    if [[ ! -d "$diar_dir" ]]; then
        log "MISSING  Diarizer models: $diar_dir"
        missing=1
    fi

    # --- Sortformer models (folder may or may not have -coreml suffix) ---
    if [[ ! -d "$MODELS_DIR/diar-streaming-sortformer-coreml" ]] && [[ ! -d "$MODELS_DIR/diar-streaming-sortformer" ]]; then
        log "MISSING  Sortformer models: $MODELS_DIR/diar-streaming-sortformer{,-coreml}"
        missing=1
    fi

    # --- LS-EEND models (folder may or may not have -coreml suffix) ---
    if [[ ! -d "$MODELS_DIR/ls-eend-coreml" ]] && [[ ! -d "$MODELS_DIR/ls-eend" ]]; then
        log "MISSING  LS-EEND models: $MODELS_DIR/ls-eend{,-coreml}"
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

    log "Downloading AMI SDM dataset + annotations..."
    "$CLI" diarization-benchmark --mode offline --auto-download --max-files 1 \
        --output "$RESULTS_DIR/warmup_offline.json" 2>&1 | tee -a "$LOG_FILE"

    log "Pre-loading Sortformer models..."
    "$CLI" sortformer-benchmark --nvidia-high-latency --hf --auto-download --max-files 1 \
        --output "$RESULTS_DIR/warmup_sortformer.json" 2>&1 | tee -a "$LOG_FILE"

    log "Pre-loading LS-EEND models..."
    "$CLI" lseend-benchmark --variant ami --auto-download --max-files 1 \
        --output "$RESULTS_DIR/warmup_lseend.json" 2>&1 | tee -a "$LOG_FILE"

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
    log "  ./Scripts/diarizer_subset_benchmark.sh --download"
    exit 1
fi
log "All assets verified locally."

log "=== Diarizer benchmark suite: ${#AMI_MEETINGS[@]}/${#ALL_AMI_MEETINGS[@]} meetings x 4 systems ==="
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
caffeinate -si -w $$ &
CAFFEINATE_PID=$!
log "caffeinate started (PID $CAFFEINATE_PID) — safe to close the lid"

# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

# Run a benchmark for each meeting via --single-file, then merge JSON results.
# This ensures we control exactly which meetings run (not the CLI's internal order).
merge_json_results() {
    local output_file="$1"
    shift
    local tmp_files=("$@")
    python3 -c "
import json, sys
results = []
for f in sys.argv[2:]:
    try:
        with open(f) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    except: pass
with open(sys.argv[1], 'w') as out:
    json.dump(results, out, indent=2)
" "$output_file" "${tmp_files[@]}" 2>/dev/null
    rm -f "${tmp_files[@]}"
}

run_offline_benchmark() {
    local label="offline_vbx"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"
    local tmp_files=()

    log "--- $label: starting (${#AMI_MEETINGS[@]} meetings, AMI SDM, offline VBx) ---"
    local start_time=$(date +%s)

    for meeting in "${AMI_MEETINGS[@]}"; do
        local tmp="$RESULTS_DIR/${label}_tmp_${meeting}.json"
        tmp_files+=("$tmp")
        log "  [$label] $meeting"
        "$CLI" diarization-benchmark \
            --mode offline \
            --dataset ami-sdm \
            --single-file "$meeting" \
            --auto-download \
            --output "$tmp" \
            2>&1 | tee -a "$LOG_FILE"
    done

    merge_json_results "$output_file" "${tmp_files[@]}"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

run_streaming_benchmark() {
    local label="streaming_5s"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"
    local tmp_files=()

    log "--- $label: starting (${#AMI_MEETINGS[@]} meetings, AMI SDM, 5s chunks, threshold=0.8) ---"
    local start_time=$(date +%s)

    for meeting in "${AMI_MEETINGS[@]}"; do
        local tmp="$RESULTS_DIR/${label}_tmp_${meeting}.json"
        tmp_files+=("$tmp")
        log "  [$label] $meeting"
        "$CLI" diarization-benchmark \
            --mode streaming \
            --dataset ami-sdm \
            --single-file "$meeting" \
            --chunk-seconds 5.0 \
            --overlap-seconds 0.0 \
            --threshold 0.8 \
            --auto-download \
            --output "$tmp" \
            2>&1 | tee -a "$LOG_FILE"
    done

    merge_json_results "$output_file" "${tmp_files[@]}"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

run_sortformer_benchmark() {
    local label="sortformer"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"
    local tmp_files=()

    log "--- $label: starting (${#AMI_MEETINGS[@]} meetings, AMI SDM, NVIDIA high-latency) ---"
    local start_time=$(date +%s)

    for meeting in "${AMI_MEETINGS[@]}"; do
        local tmp="$RESULTS_DIR/${label}_tmp_${meeting}.json"
        tmp_files+=("$tmp")
        log "  [$label] $meeting"
        "$CLI" sortformer-benchmark \
            --nvidia-high-latency \
            --hf \
            --dataset ami \
            --single-file "$meeting" \
            --auto-download \
            --output "$tmp" \
            2>&1 | tee -a "$LOG_FILE"
    done

    merge_json_results "$output_file" "${tmp_files[@]}"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

run_lseend_benchmark() {
    local label="lseend_ami"
    local output_file="$RESULTS_DIR/${label}_${TIMESTAMP}.json"
    local tmp_files=()

    log "--- $label: starting (${#AMI_MEETINGS[@]} meetings, AMI SDM, AMI variant) ---"
    local start_time=$(date +%s)

    for meeting in "${AMI_MEETINGS[@]}"; do
        local tmp="$RESULTS_DIR/${label}_tmp_${meeting}.json"
        tmp_files+=("$tmp")
        log "  [$label] $meeting"
        "$CLI" lseend-benchmark \
            --variant ami \
            --dataset ami \
            --single-file "$meeting" \
            --auto-download \
            --output "$tmp" \
            2>&1 | tee -a "$LOG_FILE"
    done

    merge_json_results "$output_file" "${tmp_files[@]}"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    log "--- $label: finished in ${elapsed}s — $output_file ---"
}

# ---------------------------------------------------------------------------
# Run all 4 benchmarks
# ---------------------------------------------------------------------------
SUITE_START=$(date +%s)

run_offline_benchmark
run_streaming_benchmark
run_sortformer_benchmark
run_lseend_benchmark

SUITE_END=$(date +%s)
SUITE_ELAPSED=$(( SUITE_END - SUITE_START ))

log "=== All benchmarks complete in ${SUITE_ELAPSED}s ==="
log "Results:"
ls -lh "$RESULTS_DIR"/*_${TIMESTAMP}.json 2>/dev/null | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Extract DER and RTFx from JSON results
# ---------------------------------------------------------------------------

# Streaming diarization benchmark: JSON is array of per-meeting results with "der" and "rtfx"
extract_streaming_metrics() {
    local json_file="$1"
    if [[ -f "$json_file" ]]; then
        python3 -c "
import json, sys
with open('$json_file') as f:
    results = json.load(f)
if not results:
    print('N/A N/A')
    sys.exit()
avg_der = sum(r['der'] for r in results) / len(results)
avg_rtfx = sum(r['rtfx'] for r in results) / len(results)
print(f'{avg_der:.1f} {avg_rtfx:.1f}')
" 2>/dev/null || echo "N/A N/A"
    else
        echo "N/A N/A"
    fi
}

# Sortformer/LS-EEND: same JSON array format via DiarizationBenchmarkUtils
extract_shared_metrics() {
    local json_file="$1"
    if [[ -f "$json_file" ]]; then
        python3 -c "
import json, sys
with open('$json_file') as f:
    results = json.load(f)
if not results:
    print('N/A N/A')
    sys.exit()
avg_der = sum(r['der'] for r in results) / len(results)
avg_rtfx = sum(r['rtfx'] for r in results) / len(results)
print(f'{avg_der:.1f} {avg_rtfx:.1f}')
" 2>/dev/null || echo "N/A N/A"
    else
        echo "N/A N/A"
    fi
}

# ---------------------------------------------------------------------------
# Compare DER & RTFx against Benchmarks.md baselines
# ---------------------------------------------------------------------------

# Baselines from Documentation/Benchmarks.md (AMI SDM, all 16 meetings)
# Note: when running a subset (--max-files <16), DER will differ from these baselines
# due to per-meeting variance. Baselines are for full 16-meeting runs only.
# Offline: no AMI SDM baseline yet — first --all run establishes it.
# Streaming: 5s/0s/0.8 on AMI SDM (7 meetings) = 26.2% DER, 223.1x RTFx
# Sortformer: NVIDIA high-latency on AMI SDM (16 meetings) = 31.7% DER, 126.7x RTFx
# LS-EEND: AMI variant on AMI SDM (16 meetings) = 20.7% DER, 74.5x RTFx
BASELINE_STREAMING_DER="26.2"
BASELINE_STREAMING_RTFX="223.1"
BASELINE_SORTFORMER_DER="31.7"
BASELINE_SORTFORMER_RTFX="126.7"
BASELINE_LSEEND_DER="20.7"
BASELINE_LSEEND_RTFX="74.5"

OFFLINE_FILE="$RESULTS_DIR/offline_vbx_${TIMESTAMP}.json"
STREAMING_FILE="$RESULTS_DIR/streaming_5s_${TIMESTAMP}.json"
SORTFORMER_FILE="$RESULTS_DIR/sortformer_${TIMESTAMP}.json"
LSEEND_FILE="$RESULTS_DIR/lseend_ami_${TIMESTAMP}.json"

read OFFLINE_DER OFFLINE_RTFX <<< $(extract_streaming_metrics "$OFFLINE_FILE")
read STREAMING_DER STREAMING_RTFX <<< $(extract_streaming_metrics "$STREAMING_FILE")
read SORTFORMER_DER SORTFORMER_RTFX <<< $(extract_shared_metrics "$SORTFORMER_FILE")
read LSEEND_DER LSEEND_RTFX <<< $(extract_shared_metrics "$LSEEND_FILE")

log ""
log "=== DER & RTFx Comparison vs Benchmarks.md baselines (AMI SDM, ${#AMI_MEETINGS[@]} meetings) ==="
log ""
printf "%-25s %12s %12s %12s %12s %12s\n" \
    "System" "Base DER" "DER" "Delta" "Base RTFx" "RTFx" | tee -a "$LOG_FILE"
printf "%-25s %12s %12s %12s %12s %12s\n" \
    "-------------------------" "------------" "------------" "------------" "------------" "------------" | tee -a "$LOG_FILE"

compare_der_rtfx() {
    local label="$1" base_der="$2" current_der="$3" base_rtfx="$4" current_rtfx="$5"

    if [[ "$current_der" == "N/A" ]]; then
        printf "%-25s %11s%% %12s %12s %11sx %12s\n" \
            "$label" "$base_der" "N/A" "—" "$base_rtfx" "N/A" | tee -a "$LOG_FILE"
        return
    fi

    local delta marker=""
    delta=$(python3 -c "print(f'{$current_der - $base_der:+.1f}')" 2>/dev/null || echo "?")
    local regression
    regression=$(python3 -c "print('YES' if $current_der > $base_der + 2.0 else 'NO')" 2>/dev/null || echo "NO")
    if [[ "$regression" == "YES" ]]; then
        marker=" <- REGRESSION"
    fi

    printf "%-25s %11s%% %11s%% %11s%% %11sx %11sx%s\n" \
        "$label" "$base_der" "$current_der" "$delta" "$base_rtfx" "$current_rtfx" "$marker" | tee -a "$LOG_FILE"
}

# Offline has no AMI SDM baseline yet — show as "new"
if [[ "$OFFLINE_DER" != "N/A" ]]; then
    printf "%-25s %12s %11s%% %12s %12s %11sx\n" \
        "Offline (VBx)" "—" "$OFFLINE_DER" "(new)" "—" "$OFFLINE_RTFX" | tee -a "$LOG_FILE"
else
    printf "%-25s %12s %12s %12s %12s %12s\n" \
        "Offline (VBx)" "—" "N/A" "—" "—" "N/A" | tee -a "$LOG_FILE"
fi

compare_der_rtfx "Streaming (5s/0.8)" "$BASELINE_STREAMING_DER" "$STREAMING_DER" "$BASELINE_STREAMING_RTFX" "$STREAMING_RTFX"
compare_der_rtfx "Sortformer (high-lat)" "$BASELINE_SORTFORMER_DER" "$SORTFORMER_DER" "$BASELINE_SORTFORMER_RTFX" "$SORTFORMER_RTFX"
compare_der_rtfx "LS-EEND (AMI)" "$BASELINE_LSEEND_DER" "$LSEEND_DER" "$BASELINE_LSEEND_RTFX" "$LSEEND_RTFX"

log ""

# Check for any DER regressions (>2.0% increase — diarization is noisier than ASR)
ANY_REGRESSION=$(python3 -c "
baselines = [
    ($BASELINE_STREAMING_DER, '$STREAMING_DER'),
    ($BASELINE_SORTFORMER_DER, '$SORTFORMER_DER'),
    ($BASELINE_LSEEND_DER, '$LSEEND_DER'),
]
for b, c in baselines:
    if c != 'N/A' and float(c) > b + 2.0:
        print('YES'); exit()
print('NO')
" 2>/dev/null || echo "NO")

if [[ "$ANY_REGRESSION" == "YES" ]]; then
    log "WARNING: DER REGRESSION DETECTED (>2.0% above baseline) — investigate before merging"
else
    log "No DER regressions (all within 2.0% of baseline)"
fi

# caffeinate will exit automatically since the parent process ($$) exits
