#!/bin/zsh
# Phonon-2 vs Parakeet TDT v3 showcase: one Ghostty window (asitop top, model log bottom) + the demo app.
# The app waits for the Transcribe button; set PHONON2_DEMO_AUTORUN=1 to start transcribing on launch.
set -e
HERE=${0:A:h}
FILE=${1:-"$HOME/Library/Application Support/FluidAudio/earnings22-1h/earnings22_top4_1h.wav"}
LOG=${PHONON2_DEMO_LOG:-/tmp/phonon2-demo.log}
SESSION=phonon2demo

cd "$HERE"
echo "building Phonon2Demo (release)…"
swift build -c release --product Phonon2Demo 2>&1 | grep -E "error|Compiling|Build complete" | tail -3
APP="$HERE/.build/release/Phonon2Demo"

: > "$LOG"
cat > /tmp/phonon2-demo-tmux.sh <<TMUX
#!/bin/zsh
tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -s $SESSION -n demo \\
  "printf '\\e[1mNeural Engine / CPU / GPU load\\e[0m  (asitop needs sudo; macmon fallback)\\n'; sudo asitop || macmon" \; \\
  set-option remain-on-exit on \; \\
  split-window -v -l 45% "printf '\\e[1mPhonon-2 demo log\\e[0m  $LOG\\n'; tail -n +1 -f $LOG" \; \\
  select-pane -t 0
TMUX
chmod +x /tmp/phonon2-demo-tmux.sh
open -na /Applications/Ghostty.app --args -e /bin/zsh /tmp/phonon2-demo-tmux.sh
sleep 2

echo "launching the app on: $FILE"
"$APP" --file "$FILE" ${PHONON2_DEMO_AUTORUN:+--autorun} 2>&1 | tee -a "$LOG"
