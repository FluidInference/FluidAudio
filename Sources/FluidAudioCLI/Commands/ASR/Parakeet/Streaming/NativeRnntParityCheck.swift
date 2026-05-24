#if os(macOS)
import FluidAudio
import Foundation

/// CLI tool to verify the Swift `NativeRnntInner` reproduces the PyTorch
/// reference output saved by `extract_decoder_joint_weights.py`.
///
/// Usage:
///     fluidaudiocli native-rnnt-parity --weights-dir <path-to-native_weights>
///
/// Reads:
///   weights.bin
///   weights_index.json
///   parity_full.json (dec_out, h0/c0 per layer, logits)
///   test_enc_step.bin (float32 raw, 1024 elements)
///
/// Runs one decode step starting from h=c=0, currentToken=blank_idx,
/// encStep=fixed-seed PyTorch randn(1024). Prints:
///   - First-5 dec_out cosine sim + max-abs-diff vs reference
///   - argmax vs reference
///   - Sampled logit values
public enum NativeRnntParityCheck {
    public static func run(arguments: [String]) async {
        var weightsDir: URL?

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--weights-dir":
                if i + 1 < arguments.count {
                    weightsDir = URL(fileURLWithPath: arguments[i + 1])
                    i += 1
                }
            case "--help", "-h":
                printUsage()
                return
            default:
                print("Unknown arg: \(arguments[i])")
            }
            i += 1
        }

        guard let dir = weightsDir else {
            print("Missing --weights-dir")
            printUsage()
            return
        }

        guard let inner = NativeRnntInner(directory: dir) else {
            print("Failed to load NativeRnntInner from \(dir.path)")
            return
        }

        // Load fixed test inputs
        let encStepURL = dir.appendingPathComponent("test_enc_step.bin")
        guard let encStepData = try? Data(contentsOf: encStepURL) else {
            print("Missing test_enc_step.bin")
            return
        }
        let encStepCount = encStepData.count / MemoryLayout<Float>.stride
        guard encStepCount == inner.encoderDim else {
            print("Expected \(inner.encoderDim) Float values in test_enc_step.bin, got \(encStepCount)")
            return
        }

        // Load reference outputs
        let parityURL = dir.appendingPathComponent("parity_full.json")
        guard let parityData = try? Data(contentsOf: parityURL),
            let parityJSON = try? JSONSerialization.jsonObject(with: parityData) as? [String: Any]
        else {
            print("Missing or malformed parity_full.json")
            return
        }

        let refDecOut = (parityJSON["dec_out"] as? [Double]) ?? []
        let refH0L0 = (parityJSON["h0_layer0"] as? [Double]) ?? []
        let refH0L1 = (parityJSON["h0_layer1"] as? [Double]) ?? []
        let refC0L0 = (parityJSON["c0_layer0"] as? [Double]) ?? []
        let refC0L1 = (parityJSON["c0_layer1"] as? [Double]) ?? []
        let refLogits = (parityJSON["logits"] as? [Double]) ?? []

        // Find expected argmax + max from refLogits
        var refMaxVal = -Double.greatestFiniteMagnitude
        var refArgmax = 0
        for (idx, val) in refLogits.enumerated() {
            if val > refMaxVal {
                refMaxVal = val
                refArgmax = idx
            }
        }

        // Token = blank_idx = vocab - 1
        let blankIdx = Int32(inner.vocab - 1)
        inner.resetState()

        // Run forward
        let predToken: Int = encStepData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int in
            let encPtr = raw.baseAddress!.assumingMemoryBound(to: Float.self)
            return inner.step(currentToken: blankIdx, encStep: encPtr)
        }

        // Compare h0 (decoder output = h1_new). We need to compare against
        // h_n[1] (h after LSTM layer 1). The Swift NativeRnntInner internal
        // state h0/h1 IS the post-step state — see `step()` swap. So after
        // step(), inner.h1 == h_n[1] (PyTorch reference h0_layer1).
        let swiftH = inner.hAsArray  // [h0_layer0, h0_layer1] concatenated
        let swiftC = inner.cAsArray

        func maxAbsDiff(_ a: [Float], _ ref: [Double]) -> Float {
            guard a.count == ref.count else { return .greatestFiniteMagnitude }
            var d: Float = 0
            for i in 0..<a.count {
                d = max(d, abs(a[i] - Float(ref[i])))
            }
            return d
        }

        func cosSim(_ a: [Float], _ ref: [Double]) -> Float {
            guard a.count == ref.count, a.count > 0 else { return 0 }
            var dot: Float = 0
            var na: Float = 0
            var nb: Float = 0
            for i in 0..<a.count {
                let bv = Float(ref[i])
                dot += a[i] * bv
                na += a[i] * a[i]
                nb += bv * bv
            }
            return dot / (sqrt(na) * sqrt(nb) + 1e-12)
        }

        let h0L0 = Array(swiftH[0..<inner.hidden])
        let h0L1 = Array(swiftH[inner.hidden..<2 * inner.hidden])
        let c0L0 = Array(swiftC[0..<inner.hidden])
        let c0L1 = Array(swiftC[inner.hidden..<2 * inner.hidden])

        print("=== Native RNN-T Parity Check ===")
        print("Vocab:        \(inner.vocab)")
        print("Blank idx:    \(blankIdx)")
        print("Hidden:       \(inner.hidden)")
        print("Encoder dim:  \(inner.encoderDim)")
        print("")
        print("LSTM layer 0 h_new:  cos=\(cosSim(h0L0, refH0L0))  maxAbs=\(maxAbsDiff(h0L0, refH0L0))")
        print("LSTM layer 0 c_new:  cos=\(cosSim(c0L0, refC0L0))  maxAbs=\(maxAbsDiff(c0L0, refC0L0))")
        print("LSTM layer 1 h_new:  cos=\(cosSim(h0L1, refH0L1))  maxAbs=\(maxAbsDiff(h0L1, refH0L1))")
        print("LSTM layer 1 c_new:  cos=\(cosSim(c0L1, refC0L1))  maxAbs=\(maxAbsDiff(c0L1, refC0L1))")
        print("Decoder out (= h_layer1):  cos=\(cosSim(h0L1, refDecOut))  maxAbs=\(maxAbsDiff(h0L1, refDecOut))")
        print("")
        print("Argmax:")
        print("  Swift: \(predToken)")
        print("  Ref:   \(refArgmax)")
        print("  Match: \(predToken == refArgmax ? "✓" : "✗")")
        print("Logit (max) Ref: \(refMaxVal)")
    }

    private static func printUsage() {
        print(
            """
            native-rnnt-parity — verify NativeRnntInner matches the PyTorch reference

            Usage:
                fluidaudio native-rnnt-parity --weights-dir <path>

            The weights directory must contain:
                weights.bin
                weights_index.json
                parity_full.json
                test_enc_step.bin

            Produce these with:
                .venv/bin/python extract_decoder_joint_weights.py \\
                    --nemo-path ... --prune-corpus-jsonl ... --output-dir ...
            """
        )
    }
}
#endif
