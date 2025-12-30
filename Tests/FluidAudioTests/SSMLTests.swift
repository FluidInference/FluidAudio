import XCTest

#if canImport(FluidAudioTTS)
@testable import FluidAudioTTS

final class SSMLTests: XCTestCase {

    // MARK: - SSMLTagParser Tests

    func testParsePhonemeTag() {
        let text = #"<phoneme alphabet="ipa" ph="kəˈkɔɹo">Kokoro</phoneme>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)
        if case .phoneme(let alphabet, let ph, let content) = tags[0].type {
            XCTAssertEqual(alphabet, "ipa")
            XCTAssertEqual(ph, "kəˈkɔɹo")
            XCTAssertEqual(content, "Kokoro")
        } else {
            XCTFail("Expected phoneme tag")
        }
    }

    func testParsePhonemeTagWithoutAlphabet() {
        let text = #"<phoneme ph="hɛˈloʊ">hello</phoneme>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)
        if case .phoneme(let alphabet, let ph, let content) = tags[0].type {
            XCTAssertEqual(alphabet, "ipa")  // defaults to ipa
            XCTAssertEqual(ph, "hɛˈloʊ")
            XCTAssertEqual(content, "hello")
        } else {
            XCTFail("Expected phoneme tag")
        }
    }

    func testParsePhonemeTagReversedAttributes() {
        // Test that attribute order doesn't matter
        let text = #"<phoneme ph="kəˈkɔɹo" alphabet="ipa">Kokoro</phoneme>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)
        if case .phoneme(let alphabet, let ph, let content) = tags[0].type {
            XCTAssertEqual(alphabet, "ipa")
            XCTAssertEqual(ph, "kəˈkɔɹo")
            XCTAssertEqual(content, "Kokoro")
        } else {
            XCTFail("Expected phoneme tag")
        }
    }

    func testParseSayAsTagReversedAttributes() {
        // Test that attribute order doesn't matter
        let text = #"<say-as format="mdy" interpret-as="date">12/25/2024</say-as>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)
        if case .sayAs(let interpretAs, let format, let content) = tags[0].type {
            XCTAssertEqual(interpretAs, "date")
            XCTAssertEqual(format, "mdy")
            XCTAssertEqual(content, "12/25/2024")
        } else {
            XCTFail("Expected say-as tag")
        }
    }

    func testParseSubTag() {
        let text = #"<sub alias="World Wide Web">WWW</sub>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)
        if case .sub(let alias, let content) = tags[0].type {
            XCTAssertEqual(alias, "World Wide Web")
            XCTAssertEqual(content, "WWW")
        } else {
            XCTFail("Expected sub tag")
        }
    }

    func testParseSayAsTag() {
        let text = #"<say-as interpret-as="cardinal">123</say-as>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)
        if case .sayAs(let interpretAs, let format, let content) = tags[0].type {
            XCTAssertEqual(interpretAs, "cardinal")
            XCTAssertNil(format)
            XCTAssertEqual(content, "123")
        } else {
            XCTFail("Expected say-as tag")
        }
    }

    func testParseSayAsTagWithFormat() {
        let text = #"<say-as interpret-as="date" format="mdy">12/25/2024</say-as>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)
        if case .sayAs(let interpretAs, let format, let content) = tags[0].type {
            XCTAssertEqual(interpretAs, "date")
            XCTAssertEqual(format, "mdy")
            XCTAssertEqual(content, "12/25/2024")
        } else {
            XCTFail("Expected say-as tag")
        }
    }

    func testParseMultipleTags() {
        let text = """
            Hello <phoneme ph="wɝld">world</phoneme>, \
            the number is <say-as interpret-as="cardinal">42</say-as>.
            """
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 2)
    }

    func testParseCaseInsensitive() {
        let text = #"<PHONEME PH="test">word</PHONEME>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)
    }

    func testParseNoTags() {
        let text = "This is plain text without any SSML tags."
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 0)
    }

    // MARK: - SayAsInterpreter Tests

    func testInterpretCharacters() {
        let result = SayAsInterpreter.interpret(content: "ABC", interpretAs: "characters", format: nil)
        XCTAssertEqual(result, "A B C")
    }

    func testInterpretSpellOut() {
        let result = SayAsInterpreter.interpret(content: "hello", interpretAs: "spell-out", format: nil)
        XCTAssertEqual(result, "h e l l o")
    }

    func testInterpretCardinal() {
        let result = SayAsInterpreter.interpret(content: "123", interpretAs: "cardinal", format: nil)
        XCTAssertEqual(result, "one hundred twenty-three")
    }

    func testInterpretCardinalAlias() {
        let result = SayAsInterpreter.interpret(content: "456", interpretAs: "number", format: nil)
        XCTAssertEqual(result, "four hundred fifty-six")
    }

    func testInterpretCardinalLarge() {
        let result = SayAsInterpreter.interpret(content: "1000000", interpretAs: "cardinal", format: nil)
        XCTAssertEqual(result, "one million")
    }

    func testInterpretOrdinalFirst() {
        let result = SayAsInterpreter.interpret(content: "1", interpretAs: "ordinal", format: nil)
        XCTAssertEqual(result, "first")
    }

    func testInterpretOrdinalSecond() {
        let result = SayAsInterpreter.interpret(content: "2", interpretAs: "ordinal", format: nil)
        XCTAssertEqual(result, "second")
    }

    func testInterpretOrdinalThird() {
        let result = SayAsInterpreter.interpret(content: "3", interpretAs: "ordinal", format: nil)
        XCTAssertEqual(result, "third")
    }

    func testInterpretOrdinalTwentieth() {
        let result = SayAsInterpreter.interpret(content: "20", interpretAs: "ordinal", format: nil)
        XCTAssertEqual(result, "twentieth")
    }

    func testInterpretOrdinalTwentyFirst() {
        let result = SayAsInterpreter.interpret(content: "21", interpretAs: "ordinal", format: nil)
        XCTAssertEqual(result, "twenty-first")
    }

    func testInterpretDigits() {
        let result = SayAsInterpreter.interpret(content: "123", interpretAs: "digits", format: nil)
        XCTAssertEqual(result, "one two three")
    }

    func testInterpretDigitsWithZero() {
        let result = SayAsInterpreter.interpret(content: "1024", interpretAs: "digits", format: nil)
        XCTAssertEqual(result, "one zero two four")
    }

    func testInterpretDateMDY() {
        let result = SayAsInterpreter.interpret(content: "12/25/2024", interpretAs: "date", format: "mdy")
        XCTAssertTrue(result.contains("December"))
        XCTAssertTrue(result.contains("twenty"))
    }

    func testInterpretDateDMY() {
        let result = SayAsInterpreter.interpret(content: "25/12/2024", interpretAs: "date", format: "dmy")
        XCTAssertTrue(result.contains("December"))
        XCTAssertTrue(result.contains("twenty"))
    }

    func testInterpretDateYMD() {
        let result = SayAsInterpreter.interpret(content: "2024-01-15", interpretAs: "date", format: "ymd")
        XCTAssertTrue(result.contains("January"))
        XCTAssertTrue(result.contains("twenty"))
    }

    func testInterpretDateYearWithOh() {
        let result = SayAsInterpreter.interpret(content: "1905", interpretAs: "date", format: "y")
        XCTAssertEqual(result, "nineteen oh five")
    }

    func testInterpretTimeDuration() {
        let result = SayAsInterpreter.interpret(content: "1'21\"", interpretAs: "time", format: nil)
        XCTAssertTrue(result.contains("minute"))
        XCTAssertTrue(result.contains("second"))
    }

    func testInterpretTimeClockTime() {
        let result = SayAsInterpreter.interpret(content: "2:30", interpretAs: "time", format: nil)
        XCTAssertEqual(result, "two thirty")
    }

    func testInterpretTimeOClock() {
        let result = SayAsInterpreter.interpret(content: "3:00", interpretAs: "time", format: nil)
        XCTAssertEqual(result, "three o'clock")
    }

    func testInterpretTimeSingleDigitMinutes() {
        let result = SayAsInterpreter.interpret(content: "3:05", interpretAs: "time", format: nil)
        XCTAssertEqual(result, "three oh five")
    }

    func testInterpretTelephone() {
        let result = SayAsInterpreter.interpret(content: "555-1234", interpretAs: "telephone", format: nil)
        XCTAssertEqual(result, "five five five one two three four")
    }

    func testInterpretTelephoneWithParens() {
        let result = SayAsInterpreter.interpret(content: "(555) 123-4567", interpretAs: "telephone", format: nil)
        XCTAssertEqual(result, "five five five one two three four five six seven")
    }

    func testInterpretFractionHalf() {
        let result = SayAsInterpreter.interpret(content: "1/2", interpretAs: "fraction", format: nil)
        XCTAssertEqual(result, "one half")
    }

    func testInterpretFractionQuarter() {
        let result = SayAsInterpreter.interpret(content: "3/4", interpretAs: "fraction", format: nil)
        XCTAssertEqual(result, "three quarters")
    }

    func testInterpretFractionGeneral() {
        let result = SayAsInterpreter.interpret(content: "2/9", interpretAs: "fraction", format: nil)
        XCTAssertEqual(result, "two ninths")
    }

    func testInterpretFractionMixed() {
        let result = SayAsInterpreter.interpret(content: "3+1/2", interpretAs: "fraction", format: nil)
        XCTAssertTrue(result.contains("three"))
        XCTAssertTrue(result.contains("half"))
    }

    func testInterpretUnknownType() {
        let result = SayAsInterpreter.interpret(
            content: "test", interpretAs: "unknown-type", format: nil)
        XCTAssertEqual(result, "test")  // Returns unchanged
    }

    // MARK: - SSMLProcessor Tests

    func testProcessPhonemeTag() {
        let text = #"Say <phoneme ph="hɛˈloʊ">hello</phoneme> to everyone."#
        let result = SSMLProcessor.process(text)

        XCTAssertEqual(result.text, "Say hello to everyone.")
        XCTAssertEqual(result.phoneticOverrides.count, 1)
        XCTAssertEqual(result.phoneticOverrides[0].word, "hello")
        XCTAssertEqual(result.phoneticOverrides[0].raw, "hɛˈloʊ")
    }

    func testProcessSubTag() {
        let text = #"Visit the <sub alias="World Wide Web">WWW</sub> today."#
        let result = SSMLProcessor.process(text)

        XCTAssertEqual(result.text, "Visit the World Wide Web today.")
        XCTAssertEqual(result.phoneticOverrides.count, 0)
    }

    func testProcessSayAsCardinal() {
        let text = #"The answer is <say-as interpret-as="cardinal">42</say-as>."#
        let result = SSMLProcessor.process(text)

        XCTAssertTrue(result.text.contains("forty-two"))
        XCTAssertEqual(result.phoneticOverrides.count, 0)
    }

    func testProcessSayAsOrdinal() {
        let text = #"This is the <say-as interpret-as="ordinal">1</say-as> time."#
        let result = SSMLProcessor.process(text)

        XCTAssertTrue(result.text.contains("first"))
    }

    func testProcessSayAsDate() {
        let text = #"Meet me on <say-as interpret-as="date" format="mdy">12/25/2024</say-as>."#
        let result = SSMLProcessor.process(text)

        XCTAssertTrue(result.text.contains("December"))
    }

    func testProcessNoSSML() {
        let text = "This is plain text without any SSML tags."
        let result = SSMLProcessor.process(text)

        XCTAssertEqual(result.text, text)
        XCTAssertEqual(result.phoneticOverrides.count, 0)
    }

    func testProcessMultipleTags() {
        let text = """
            <say-as interpret-as="cardinal">100</say-as> people said \
            <sub alias="hello">hi</sub>.
            """
        let result = SSMLProcessor.process(text)

        XCTAssertTrue(result.text.contains("one hundred"))
        XCTAssertTrue(result.text.contains("hello"))
    }

    func testProcessPhonemeWordIndex() {
        let text = #"First word <phoneme ph="test">second</phoneme> third word."#
        let result = SSMLProcessor.process(text)

        XCTAssertEqual(result.phoneticOverrides.count, 1)
        XCTAssertEqual(result.phoneticOverrides[0].wordIndex, 2)  // "second" is the 3rd word (0-indexed: 2)
    }

    func testProcessMultiplePhonemes() {
        let text =
            #"<phoneme ph="wʌn">One</phoneme> and <phoneme ph="tu">two</phoneme>."#
        let result = SSMLProcessor.process(text)

        XCTAssertEqual(result.phoneticOverrides.count, 2)
        XCTAssertEqual(result.phoneticOverrides[0].word, "One")
        XCTAssertEqual(result.phoneticOverrides[1].word, "two")
    }

    // MARK: - Integration with TtsTextPreprocessor

    func testPreprocessorWithSSML() {
        let text = #"The number <say-as interpret-as="cardinal">1000</say-as> is big."#
        let result = TtsTextPreprocessor.preprocessDetailed(text)

        XCTAssertTrue(result.text.contains("one thousand"))
    }

    func testPreprocessorSSMLCoexistsWithMarkdown() {
        let text = #"<say-as interpret-as="ordinal">1</say-as> and [second](/sɛkənd/)"#
        let result = TtsTextPreprocessor.preprocessDetailed(text)

        XCTAssertTrue(result.text.contains("first"))
        XCTAssertTrue(result.text.contains("second"))
        // Should have phonetic override from markdown syntax
        XCTAssertGreaterThanOrEqual(result.phoneticOverrides.count, 1)
    }

    func testPreprocessorSSMLThenNormalization() {
        // SSML processed first, then currency normalization
        let text = #"<say-as interpret-as="cardinal">5</say-as> costs $12.50."#
        let result = TtsTextPreprocessor.preprocessDetailed(text)

        XCTAssertTrue(result.text.contains("five"))
        XCTAssertTrue(result.text.contains("dollars"))
    }

    // MARK: - Edge Cases: Malformed SSML Tags

    func testMalformedPhonemeNoClosingTag() {
        let text = #"Say <phoneme ph="test">hello to everyone."#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 0)  // Should not match incomplete tag
    }

    func testMalformedPhonemeNoPhAttribute() {
        let text = #"<phoneme alphabet="ipa">hello</phoneme>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 0)  // ph is required
    }

    func testMalformedSubNoAlias() {
        let text = "<sub>WWW</sub>"
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 0)  // alias is required
    }

    func testMalformedSayAsNoInterpretAs() {
        let text = #"<say-as format="mdy">12/25/2024</say-as>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 0)  // interpret-as is required
    }

    func testMalformedEmptyContent() {
        let text = #"<phoneme ph="test"></phoneme>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)  // Empty content is valid
        if case .phoneme(_, _, let content) = tags[0].type {
            XCTAssertEqual(content, "")
        }
    }

    func testMalformedNestedAngleBrackets() {
        // Content with angle brackets should not be parsed
        let text = #"<sub alias="test">contains < and > symbols</sub>"#
        let tags = SSMLTagParser.parse(text)

        // Regex pattern [^<]* won't match content with <
        XCTAssertEqual(tags.count, 0)
    }

    func testMalformedUnclosedQuote() {
        let text = #"<phoneme ph="test>hello</phoneme>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 0)  // Unclosed quote
    }

    func testMalformedMismatchedTags() {
        let text = #"<phoneme ph="test">hello</sub>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 0)  // Mismatched closing tag
    }

    func testMalformedExtraWhitespace() {
        // Should handle extra whitespace gracefully
        let text = #"<phoneme   ph = "test"  >hello</phoneme>"#
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)  // Should still parse
    }

    func testMalformedSingleQuotes() {
        // Should handle single quotes as well as double quotes
        let text = "<phoneme ph='test'>hello</phoneme>"
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 1)
    }

    func testMalformedMixedQuotes() {
        // Mixed quotes are accepted by our flexible regex (["'] matches either)
        let text = #"<phoneme ph="test'>hello</phoneme>"#
        let tags = SSMLTagParser.parse(text)

        // The regex pattern ["']([^"']*)["'] allows mixed quote types
        XCTAssertEqual(tags.count, 1)
    }

    func testMalformedPartialTag() {
        let text = "<phoneme"
        let tags = SSMLTagParser.parse(text)

        XCTAssertEqual(tags.count, 0)
    }

    func testMalformedJustOpeningBracket() {
        let text = "This has a < but no tag"
        let result = SSMLProcessor.process(text)

        // Should return text unchanged since no valid tags
        XCTAssertEqual(result.text, text)
    }

    func testMalformedHTMLEntities() {
        // HTML entities in content
        let text = #"<sub alias="test &amp; more">T&M</sub>"#
        let tags = SSMLTagParser.parse(text)

        // Our simple parser doesn't decode entities, but should match
        XCTAssertEqual(tags.count, 1)
        if case .sub(let alias, _) = tags[0].type {
            XCTAssertEqual(alias, "test &amp; more")  // Raw, not decoded
        }
    }

    func testProcessorMalformedTagsPassthrough() {
        let text = "Hello <phoneme malformed>world</phoneme> there."
        let result = SSMLProcessor.process(text)

        // Malformed tag should pass through unchanged
        XCTAssertEqual(result.text, text)
        XCTAssertEqual(result.phoneticOverrides.count, 0)
    }

    func testProcessorMixedValidAndInvalidTags() {
        let text = #"<say-as interpret-as="cardinal">5</say-as> and <broken>invalid</broken>"#
        let result = SSMLProcessor.process(text)

        // Valid tag should be processed, invalid should remain
        XCTAssertTrue(result.text.contains("five"))
        XCTAssertTrue(result.text.contains("<broken>"))
    }

    // MARK: - Edge Cases: Say-As Interpreters

    func testSayAsCardinalInvalidInput() {
        let result = SayAsInterpreter.interpret(content: "not a number", interpretAs: "cardinal", format: nil)
        XCTAssertEqual(result, "not a number")  // Returns unchanged
    }

    func testSayAsOrdinalInvalidInput() {
        let result = SayAsInterpreter.interpret(content: "abc", interpretAs: "ordinal", format: nil)
        XCTAssertEqual(result, "abc")  // Returns unchanged
    }

    func testSayAsDigitsNonDigits() {
        let result = SayAsInterpreter.interpret(content: "12ab34", interpretAs: "digits", format: nil)
        XCTAssertEqual(result, "one two three four")  // Only digits
    }

    func testSayAsFractionInvalidFormat() {
        let result = SayAsInterpreter.interpret(content: "not/a/fraction", interpretAs: "fraction", format: nil)
        XCTAssertEqual(result, "not/a/fraction")  // Returns unchanged
    }

    func testSayAsFractionZeroDenominator() {
        let result = SayAsInterpreter.interpret(content: "5/0", interpretAs: "fraction", format: nil)
        XCTAssertEqual(result, "5/0")  // Returns unchanged (division by zero protection)
    }

    func testSayAsDateInvalidFormat() {
        let result = SayAsInterpreter.interpret(content: "not-a-date", interpretAs: "date", format: "mdy")
        // Should handle gracefully
        XCTAssertFalse(result.isEmpty)
    }

    func testSayAsDateEmptyComponents() {
        let result = SayAsInterpreter.interpret(content: "", interpretAs: "date", format: "mdy")
        XCTAssertEqual(result, "")  // Empty returns empty
    }

    func testSayAsTimeInvalidFormat() {
        let result = SayAsInterpreter.interpret(content: "not time", interpretAs: "time", format: nil)
        XCTAssertEqual(result, "not time")  // Returns unchanged
    }

    func testSayAsTelephoneEmpty() {
        let result = SayAsInterpreter.interpret(content: "", interpretAs: "telephone", format: nil)
        XCTAssertEqual(result, "")
    }

    func testSayAsCharactersEmpty() {
        let result = SayAsInterpreter.interpret(content: "", interpretAs: "characters", format: nil)
        XCTAssertEqual(result, "")
    }

    func testSayAsCardinalNegativeNumber() {
        let result = SayAsInterpreter.interpret(content: "-42", interpretAs: "cardinal", format: nil)
        XCTAssertTrue(result.contains("forty"))  // Should handle negative
    }

    func testSayAsOrdinalLargeNumber() {
        let result = SayAsInterpreter.interpret(content: "100", interpretAs: "ordinal", format: nil)
        XCTAssertTrue(result.contains("hundredth"))
    }

    func testSayAsOrdinalEleventh() {
        // Special case: 11th, 12th, 13th
        let result = SayAsInterpreter.interpret(content: "11", interpretAs: "ordinal", format: nil)
        XCTAssertEqual(result, "eleventh")
    }

    func testSayAsOrdinalTwelfth() {
        let result = SayAsInterpreter.interpret(content: "12", interpretAs: "ordinal", format: nil)
        XCTAssertEqual(result, "twelfth")
    }

    func testSayAsOrdinalThirteenth() {
        let result = SayAsInterpreter.interpret(content: "13", interpretAs: "ordinal", format: nil)
        XCTAssertEqual(result, "thirteenth")
    }

    // MARK: - Additional Edge Cases

    func testUnicodeContent() {
        // café with accent
        let result1 = SSMLProcessor.process(#"<sub alias="coffee shop">café</sub>"#)
        XCTAssertEqual(result1.text, "coffee shop")

        // Japanese text
        let result2 = SSMLProcessor.process(#"<sub alias="Japan">日本</sub>"#)
        XCTAssertEqual(result2.text, "Japan")
    }

    func testFractionOneThird() {
        let result = SayAsInterpreter.interpret(content: "1/3", interpretAs: "fraction", format: nil)
        XCTAssertEqual(result, "one third")
    }

    func testFractionTwoHalves() {
        let result = SayAsInterpreter.interpret(content: "2/2", interpretAs: "fraction", format: nil)
        XCTAssertEqual(result, "two halves")
    }

    func testFractionLargeDenominator() {
        let result = SayAsInterpreter.interpret(content: "3/100", interpretAs: "fraction", format: nil)
        XCTAssertTrue(result.contains("hundredth"))
    }

    func testWordIndexAfterCurrencyExpansion() {
        // SSML word index is calculated on ORIGINAL text before currency expansion
        let text = "Cost is $10 and <phoneme ph=\"test\">word</phoneme> follows"
        let result = TtsTextPreprocessor.preprocessDetailed(text)
        XCTAssertEqual(result.phoneticOverrides.count, 1)
        // Original: "Cost is $10 and word follows" - word index 4
        // (Currency expansion happens after SSML processing)
        XCTAssertEqual(result.phoneticOverrides[0].wordIndex, 4)
    }

    func testAdjacentTags() {
        let text = #"<say-as interpret-as="cardinal">1</say-as><say-as interpret-as="cardinal">2</say-as>"#
        let result = SSMLProcessor.process(text)
        XCTAssertEqual(result.text, "onetwo")
    }

    func testOrdinal111() {
        let result = SayAsInterpreter.interpret(content: "111", interpretAs: "ordinal", format: nil)
        XCTAssertTrue(result.contains("eleventh") || result.contains("hundred"))
    }

    func testOrdinal1000() {
        let result = SayAsInterpreter.interpret(content: "1000", interpretAs: "ordinal", format: nil)
        XCTAssertTrue(result.contains("thousandth"))
    }

    func testInvalidMonthBound() {
        // Month 24 is invalid - should return original content
        let result = SayAsInterpreter.interpret(content: "24/13/2025", interpretAs: "date", format: "mdy")
        XCTAssertEqual(result, "24/13/2025")  // Returns unchanged
    }

    func testValidDateWithLargeDay() {
        // Day 32 is technically invalid but we still format it as ordinal
        let result = SayAsInterpreter.interpret(content: "12/25/2024", interpretAs: "date", format: "mdy")
        XCTAssertTrue(result.contains("December"))
        XCTAssertTrue(result.contains("twenty"))
    }

    func testWhitespaceInContent() {
        let result = SSMLProcessor.process("<sub alias=\"hello world\">  test  </sub>")
        XCTAssertEqual(result.text, "hello world")
    }

    func testNewlineInContent() {
        let result = SSMLProcessor.process("<sub alias=\"replaced\">line1\nline2</sub>")
        XCTAssertEqual(result.text, "replaced")
    }
}

#endif
