import Foundation
import RegexBuilder

/// Official Text Normalizer ported from HuggingFace OpenASR Leaderboard
/// Source: https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/normalizer.py
public struct TextNormalizerOfficial {
    
    // MARK: - Constants & Mappings
    
    private static let additionalDiacritics: [Character: String] = [
        "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
        "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
    ]
    
    // MARK: - Basic Normalization
    
    private static func removeSymbolsAndDiacritics(_ text: String, keep: String = "") -> String {
        // "Replace any other markers, symbols, and punctuations with a space, and drop any diacritics"
        // Python: unicodedata.normalize("NFKD", s)
        let nfkd = text.precomposedStringWithCompatibilityMapping // Swift's closest equivalent to NFKD/NFKC usually
        // Actually Swift String is already canonical. But we can iterate unicode scalars.
        
        var result = ""
        for char in nfkd {
            if keep.contains(char) {
                result.append(char)
                continue
            }
            
            if let mapped = additionalDiacritics[char] {
                result.append(mapped)
                continue
            }
            
            let scalars = char.unicodeScalars
            if let first = scalars.first {
                if first.properties.generalCategory == .nonspacingMark { // Mn
                    continue
                }
                // MSP: Modifier Symbol (Sk), Symbol (S*), Punctuation (P*)
                // Python: unicodedata.category(char)[0] in "MSP"
                let cat = first.properties.generalCategory
                let catStr = String(describing: cat)
                // Swift categories: uppercaseLetter, lowercaseLetter, etc.
                // We need to map Swift categories to "M", "S", "P"
                // Punctuation: .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation, .initialQuotePunctuation, .finalQuotePunctuation, .otherPunctuation
                // Symbol: .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol
                // Mark: .nonspacingMark, .spacingMark, .enclosingMark
                
                let isPunctuation = cat == .connectorPunctuation || cat == .dashPunctuation || cat == .openPunctuation || cat == .closePunctuation || cat == .initialPunctuation || cat == .finalPunctuation || cat == .otherPunctuation
                let isSymbol = cat == .mathSymbol || cat == .currencySymbol || cat == .modifierSymbol || cat == .otherSymbol
                let isMark = cat == .nonspacingMark || cat == .spacingMark || cat == .enclosingMark
                
                if isPunctuation || isSymbol || isMark {
                     // Python logic: if category[0] in "MSP" -> " "
                     // But wait, Mn is dropped earlier.
                     // Here we replace MSP with space.
                     result.append(" ")
                } else {
                    result.append(char)
                }
            } else {
                result.append(char)
            }
        }
        return result
    }
    
    // MARK: - English Number Normalizer
    
    private struct EnglishNumberNormalizer {
        let zeros: Set<String> = ["o", "oh", "zero"]
        let ones: [String: Int] = [
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19
        ]
        let onesPlural: [String: (Int, String)]
        let onesOrdinal: [String: (Int, String)]
        let onesSuffixed: [String: (Int, String)]
        
        let tens: [String: Int] = [
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
        ]
        let tensPlural: [String: (Int, String)]
        let tensOrdinal: [String: (Int, String)]
        let tensSuffixed: [String: (Int, String)]
        
        let multipliers: [String: Int] = [
            "hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000,
            "trillion": 1000000000000, "quadrillion": 1000000000000000
        ]
        let multipliersPlural: [String: (Int, String)]
        let multipliersOrdinal: [String: (Int, String)]
        let multipliersSuffixed: [String: (Int, String)]
        
        let decimals: Set<String>
        
        let precedingPrefixers = ["minus": "-", "negative": "-", "plus": "+", "positive": "+"]
        let followingPrefixers = ["pound": "£", "pounds": "£", "euro": "€", "euros": "€", "dollar": "$", "dollars": "$", "cent": "¢", "cents": "¢"]
        let prefixes: Set<String>
        let suffixers = ["per": ["cent": "%"], "percent": "%"] as [String : Any]
        let specials: Set<String> = ["and", "double", "triple", "point"]
        
        let words: Set<String>
        
        init() {
            // Initialize derived maps
            var op: [String: (Int, String)] = [:]
            for (k, v) in ones { op[k == "six" ? "sixes" : k + "s"] = (v, "s") }
            self.onesPlural = op
            
            var oo: [String: (Int, String)] = [
                "zeroth": (0, "th"), "first": (1, "st"), "second": (2, "nd"), "third": (3, "rd"),
                "fifth": (5, "th"), "twelfth": (12, "th")
            ]
            for (k, v) in ones {
                if v > 3 && v != 5 && v != 12 {
                    let suffix = k.hasSuffix("t") ? "h" : "th"
                    oo[k + suffix] = (v, "th")
                }
            }
            self.onesOrdinal = oo
            self.onesSuffixed = op.merging(oo) { (_, new) in new }
            
            var tp: [String: (Int, String)] = [:]
            for (k, v) in tens { tp[k.replacingOccurrences(of: "y", with: "ies")] = (v, "s") }
            self.tensPlural = tp
            
            var to: [String: (Int, String)] = [:]
            for (k, v) in tens { to[k.replacingOccurrences(of: "y", with: "ieth")] = (v, "th") }
            self.tensOrdinal = to
            self.tensSuffixed = tp.merging(to) { (_, new) in new }
            
            var mp: [String: (Int, String)] = [:]
            for (k, v) in multipliers { mp[k + "s"] = (v, "s") }
            self.multipliersPlural = mp
            
            var mo: [String: (Int, String)] = [:]
            for (k, v) in multipliers { mo[k + "th"] = (v, "th") }
            self.multipliersOrdinal = mo
            self.multipliersSuffixed = mp.merging(mo) { (_, new) in new }
            
            self.decimals = Set(ones.keys).union(tens.keys).union(zeros)
            
            self.prefixes = Set(precedingPrefixers.values).union(followingPrefixers.values)
            
            var w = zeros
            w.formUnion(ones.keys)
            w.formUnion(onesSuffixed.keys)
            w.formUnion(tens.keys)
            w.formUnion(tensSuffixed.keys)
            w.formUnion(multipliers.keys)
            w.formUnion(multipliersSuffixed.keys)
            w.formUnion(precedingPrefixers.keys)
            w.formUnion(followingPrefixers.keys)
            w.formUnion(suffixers.keys)
            w.formUnion(specials)
            self.words = w
        }
        
        func callAsFunction(_ s: String) -> String {
            let preprocessed = preprocess(s)
            let words = preprocessed.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            let processed = processWords(words)
            return postprocess(processed.joined(separator: " "))
        }
        
        private func preprocess(_ s: String) -> String {
            // replace "<number> and a half" with "<number> point five"
            // Simplified regex approach
            var res = s
            // TODO: Implement "and a half" logic if critical. Skipping for brevity for now as it's complex regex.
            
            // put a space at number/letter boundary
            res = res.replacingOccurrences(of: "([a-z])([0-9])", with: "$1 $2", options: .regularExpression)
            res = res.replacingOccurrences(of: "([0-9])([a-z])", with: "$1 $2", options: .regularExpression)
            
            // remove spaces which could be a suffix
            res = res.replacingOccurrences(of: "([0-9])\\s+(st|nd|rd|th|s)\\b", with: "$1$2", options: .regularExpression)
            
            return res
        }
        
        private enum NumberValue {
            case int(Int)
            case string(String)
            
            var asString: String {
                switch self {
                case .int(let i): return String(i)
                case .string(let s): return s
                }
            }
        }

        private func processWords(_ words: [String]) -> [String] {
            var result: [String] = []
            var prefix: String? = nil
            var value: NumberValue? = nil
            var skip = false
            
            func output(_ val: NumberValue) -> String {
                var res = val.asString
                if let p = prefix {
                    res = p + res
                }
                value = nil
                prefix = nil
                return res
            }
            
            for (i, current) in words.enumerated() {
                if skip { skip = false; continue }
                
                let prev = i > 0 ? words[i-1] : nil
                // let next = i < words.count - 1 ? words[i+1] : nil
                
                let hasPrefix = prefixes.contains(String(current.prefix(1)))
                let currentWithoutPrefix = hasPrefix ? String(current.dropFirst()) : current
                
                if let _ = Double(currentWithoutPrefix) {
                    // Arabic numbers
                    if let v = value {
                        if case .string(let vStr) = v, vStr.hasSuffix(".") {
                            value = .string(vStr + current)
                            continue
                        } else {
                            result.append(output(v))
                        }
                    }
                    
                    if hasPrefix { prefix = String(current.prefix(1)) }
                    value = .string(currentWithoutPrefix)
                } else if !self.words.contains(current) {
                    if let v = value { result.append(output(v)) }
                    result.append(current) // No prefix logic for non-number words? Python code just appends current.
                } else if zeros.contains(current) {
                    if let v = value {
                        value = .string(v.asString + "0")
                    } else {
                        value = .string("0")
                    }
                } else if let val = ones[current] {
                    if value == nil {
                        value = .int(val)
                    } else {
                        switch value! {
                        case .string(let s):
                            if let p = prev, ones.keys.contains(p) && val < 10 {
                                value = .string(s + String(val))
                            } else {
                                result.append(output(value!))
                                value = .int(val)
                            }
                        case .int(let v):
                            if let p = prev, ones.keys.contains(p) {
                                // "one two" -> "12"
                                value = .string(String(v) + String(val))
                            } else if val < 10 {
                                if v % 10 == 0 {
                                    value = .int(v + val)
                                } else {
                                    value = .string(String(v) + String(val))
                                }
                            } else { // eleven to nineteen
                                if v % 100 == 0 {
                                    value = .int(v + val)
                                } else {
                                    value = .string(String(v) + String(val))
                                }
                            }
                        }
                    }
                } else if let val = tens[current] {
                    if value == nil {
                        value = .int(val)
                    } else {
                        switch value! {
                        case .string(_):
                            result.append(output(value!))
                            value = .int(val)
                        case .int(let v):
                            if v % 100 == 0 {
                                value = .int(v + val)
                            } else {
                                value = .string(String(v) + String(val))
                            }
                        }
                    }
                } else if let val = multipliers[current] {
                    if value == nil {
                        value = .int(val)
                    } else {
                        switch value! {
                        case .string(_):
                            result.append(output(value!))
                            value = .int(val)
                        case .int(let v):
                            value = .int(v * val)
                        }
                    }
                } else {
                    // Fallback for other words in self.words (suffixes, etc.) not yet implemented
                    if let v = value { result.append(output(v)) }
                    result.append(current)
                }
            }
            
            if let v = value { result.append(output(v)) }
            return result
        }
        
        private func postprocess(_ s: String) -> String {
            var res = s
            // currency postprocessing
            res = res.replacingOccurrences(of: "([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\\b", with: "$1$2.$3", options: .regularExpression)
            res = res.replacingOccurrences(of: "[€£$]0.([0-9]{1,2})\\b", with: "¢$1", options: .regularExpression)
            res = res.replacingOccurrences(of: "\\b1(s?)\\b", with: "one$1", options: .regularExpression)
            return res
        }
    }

    // MARK: - Main Call
    
    public static func normalize(_ text: String) -> String {
        var s = text.lowercased()
        
        // Remove brackets/parens
        s = s.replacingOccurrences(of: "[<\\[][^>\\]]*[>\\]]", with: "", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\([^)]+?\\)", with: "", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\b(hmm|mm|mhm|mmm|uh|um)\\b", with: "", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\s+'", with: "'", options: .regularExpression)
        
        // Replacers (Contractions)
        let replacers = [
            "\\bwon't\\b": "will not", "\\bcan't\\b": "can not", "\\blet's\\b": "let us", "\\bain't\\b": "aint",
            "\\by'all\\b": "you all", "\\bwanna\\b": "want to", "\\bgotta\\b": "got to", "\\bgonna\\b": "going to",
            "\\bi'ma\\b": "i am going to", "\\bimma\\b": "i am going to", "\\bwoulda\\b": "would have",
            "\\bcoulda\\b": "could have", "\\bshoulda\\b": "should have", "\\bma'am\\b": "madam",
            "\\bmr\\b": "mister ", "\\bmrs\\b": "missus ", "\\bst\\b": "saint ", "\\bdr\\b": "doctor ",
            "\\bprof\\b": "professor ", "\\bcapt\\b": "captain ", "\\bgov\\b": "governor ", "\\bald\\b": "alderman ",
            "\\bgen\\b": "general ", "\\bsen\\b": "senator ", "\\brep\\b": "representative ", "\\bpres\\b": "president ",
            "\\brev\\b": "reverend ", "\\bhon\\b": "honorable ", "\\basst\\b": "assistant ", "\\bassoc\\b": "associate ",
            "\\blt\\b": "lieutenant ", "\\bcol\\b": "colonel ", "\\bjr\\b": "junior ", "\\bsr\\b": "senior ",
            "\\besq\\b": "esquire ",
            "'d been\\b": " had been", "'s been\\b": " has been", "'d gone\\b": " had gone", "'s gone\\b": " has gone",
            "'d done\\b": " had done", "'s got\\b": " has got",
            "n't\\b": " not", "'re\\b": " are", "'s\\b": " is", "'d\\b": " would", "'ll\\b": " will",
            "'t\\b": " not", "'ve\\b": " have", "'m\\b": " am"
        ]
        
        for (pattern, replacement) in replacers {
            s = s.replacingOccurrences(of: pattern, with: replacement, options: .regularExpression)
        }
        
        s = s.replacingOccurrences(of: "(\\d),(\\d)", with: "$1$2", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\.([^0-9]|$)", with: " $1", options: .regularExpression)
        
        s = removeSymbolsAndDiacritics(s, keep: ".%$¢€£")
        
        // Number Normalization
        let numberNormalizer = EnglishNumberNormalizer()
        s = numberNormalizer(s)
        
        // Spelling Normalization
        s = s.components(separatedBy: .whitespaces).map { englishSpellingNormalizer[$0] ?? $0 }.joined(separator: " ")
        
        s = s.replacingOccurrences(of: "[.$¢€£]([^0-9])", with: " $1", options: .regularExpression)
        s = s.replacingOccurrences(of: "([^0-9])%", with: "$1 ", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        
        // EOU Stripping
        if s.hasSuffix("eou") {
            s = String(s.dropLast(3))
        }
        
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
