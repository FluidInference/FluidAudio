import Foundation

/// Corpus BLEU-4 with a sacreBLEU-13a-style tokenizer (case-sensitive,
/// punctuation split into standalone tokens). Close to `sacrebleu` defaults
/// for sanity checks in-process; cite numbers from sacrebleu itself (the
/// benchmark writes a hypotheses JSON for that).
enum BLEUCalculator {

    /// 13a-style tokenization: split out punctuation/symbols, collapse whitespace.
    /// Like mteval-13a, `.` and `,` stay attached when BOTH neighbors are ASCII
    /// digits ("3,5" and "2.5" are single tokens).
    static func tokenize(_ text: String) -> [String] {
        let scalars = Array(text.unicodeScalars)
        var out: [String] = []
        var current = ""

        func isAsciiDigit(_ i: Int) -> Bool {
            guard i >= 0, i < scalars.count else { return false }
            return scalars[i] >= "0" && scalars[i] <= "9"
        }

        for (i, scalar) in scalars.enumerated() {
            let c = Character(scalar)
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                if !current.isEmpty {
                    out.append(current)
                    current = ""
                }
            } else if CharacterSet.punctuationCharacters.contains(scalar)
                || CharacterSet.symbols.contains(scalar)
            {
                if scalar == "." || scalar == ",", isAsciiDigit(i - 1), isAsciiDigit(i + 1) {
                    current.append(c)
                    continue
                }
                if !current.isEmpty {
                    out.append(current)
                    current = ""
                }
                out.append(String(c))
            } else {
                current.append(c)
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    /// Corpus BLEU over parallel hypothesis/reference lists, as a percentage.
    static func corpusBLEU(hypotheses: [String], references: [String]) -> Double {
        precondition(hypotheses.count == references.count)
        let maxOrder = 4
        var matches = [Int](repeating: 0, count: maxOrder)
        var totals = [Int](repeating: 0, count: maxOrder)
        var hypLength = 0
        var refLength = 0

        for (hyp, ref) in zip(hypotheses, references) {
            let h = tokenize(hyp)
            let r = tokenize(ref)
            hypLength += h.count
            refLength += r.count

            for order in 1...maxOrder {
                guard h.count >= order else { continue }
                totals[order - 1] += h.count - order + 1

                var refCounts: [ArraySlice<String>: Int] = [:]
                if r.count >= order {
                    for i in 0...(r.count - order) {
                        refCounts[r[i..<(i + order)], default: 0] += 1
                    }
                }
                for i in 0...(h.count - order) {
                    let gram = h[i..<(i + order)]
                    if let c = refCounts[gram], c > 0 {
                        matches[order - 1] += 1
                        refCounts[gram] = c - 1
                    }
                }
            }
        }

        // Geometric mean of n-gram precisions (0 if any order has no match).
        var logSum = 0.0
        for i in 0..<maxOrder {
            guard totals[i] > 0, matches[i] > 0 else { return 0 }
            logSum += log(Double(matches[i]) / Double(totals[i]))
        }
        let geoMean = exp(logSum / Double(maxOrder))
        let bp = hypLength >= refLength ? 1.0 : exp(1.0 - Double(refLength) / Double(max(hypLength, 1)))
        return geoMean * bp * 100.0
    }
}
