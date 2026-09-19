#!/bin/bash
# Build a native app bundle with SwiftPM; a full Xcode project is not required.
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
swift build --package-path "$DEMO_DIR" -j 4
BIN_DIR="$(swift build --package-path "$DEMO_DIR" --show-bin-path)"
APP_DIR="$DEMO_DIR/.build/CUA Forms.app"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "$BIN_DIR/CuaS1FormsDemo" "$APP_DIR/Contents/MacOS/CuaS1FormsDemo.new"
mv -f "$APP_DIR/Contents/MacOS/CuaS1FormsDemo.new" "$APP_DIR/Contents/MacOS/CuaS1FormsDemo"
# Store resource bundles in the standard signed-app location.
for BUNDLE_PATH in "$BIN_DIR"/*.bundle; do
    [ -d "$BUNDLE_PATH" ] || continue
    ditto "$BUNDLE_PATH" "$APP_DIR/Contents/Resources/$(basename "$BUNDLE_PATH")"
done
cat > "$APP_DIR/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>CuaS1FormsDemo</string>
<key>CFBundleIdentifier</key><string>com.fluidinference.cua-forms-demo</string>
<key>CFBundleName</key><string>CUA Forms</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$APP_DIR"
open -n "$APP_DIR" --args "$@"
echo "Opened $APP_DIR"
