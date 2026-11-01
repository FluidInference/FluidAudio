#!/bin/zsh
# Bark typewriter showcase: one Ghostty window (asitop, or macmon when sudo would prompt; model log below) + the app.
# The app waits for the Transcribe button; set BARK_DEMO_AUTORUN=1 to start transcribing on launch.
# Extra arguments go to the app, e.g. ./demo.sh ~/audio.wav --model clean
set -e
HERE=${0:A:h}
FILE=${1:-"$HOME/Library/Application Support/FluidAudio/earnings22-1h/earnings22_top4_1h.wav"}
[[ $# -gt 0 ]] && shift
LOG=${BARK_DEMO_LOG:-/tmp/bark-demo.log}
SESSION=barkdemo

cd "$HERE"
# Build with the same toolchain as the rest of the repo; the Command Line Tools one forces a full rebuild.
XCODE=/Applications/Xcode-27.0.0.app/Contents/Developer
[[ -z "$DEVELOPER_DIR" && -d $XCODE ]] && export DEVELOPER_DIR=$XCODE
echo "building BarkDemo (release)…"
swift build -c release --product BarkDemo 2>&1 | grep -E "error|Compiling|Build complete" | tail -3
APP="$HERE/.build/release/BarkDemo"

: > "$LOG"
# One Ghostty window, two rows: the load monitor on top, the app log below. The session is created detached and
# Ghostty attaches to it, so the window always shows this session.
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -n demo -x 200 -y 50 \
  "printf '\\e[1mNeural Engine / CPU / GPU load\\e[0m\\n'; if sudo -n true 2>/dev/null; then sudo asitop || macmon; else macmon; fi"
tmux set-option -t $SESSION remain-on-exit on
tmux split-window -v -l 45% -t $SESSION "printf '\\e[1mBark demo log\\e[0m  $LOG\\n'; tail -n +1 -f $LOG"
tmux select-pane -t $SESSION:0.0
open -na /Applications/Ghostty.app --args -e "$(command -v tmux)" attach -t $SESSION
sleep 2

echo "launching the app on: $FILE"
"$APP" --file "$FILE" ${BARK_DEMO_AUTORUN:+--autorun} "$@" 2>&1 | tee -a "$LOG"
