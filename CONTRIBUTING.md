# Contributing

This project uses `swift-format` to maintain consistent code style. All pull requests are automatically checked for formatting compliance.

## Local Development

```bash
# Format all code (requires Swift 6+ for contributors only)
# Users of the library don't need Swift 6
swift format --in-place --recursive --configuration .swift-format Sources/ Tests/

# Check formatting without modifying
swift format lint --recursive --configuration .swift-format Sources/ Tests/

# For Swift <6, install swift-format separately:
# git clone https://github.com/apple/swift-format
# cd swift-format && swift build -c release
# cp .build/release/swift-format /usr/local/bin/
```

## Automatic Checks

- PRs will fail if code is not properly formatted
- GitHub Actions runs formatting checks on all Swift file changes
- See `.swift-format` for style configuration
