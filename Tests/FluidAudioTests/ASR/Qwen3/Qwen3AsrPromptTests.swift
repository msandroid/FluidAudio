import Foundation
import XCTest

@testable import FluidAudio

final class Qwen3AsrPromptTests: XCTestCase {

    private let asrText = Int32(Qwen3AsrConfig.asrTextTokenId)

    /// Official template with an empty system turn, as tokenized by the Qwen/Qwen3-ASR-0.6B tokenizer:
    /// `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>{audio}<|audio_end|><|im_end|>\n`
    /// `<|im_start|>assistant\n`
    private func officialBase(numAudioFrames: Int) -> [Int32] {
        [151_644, 8948, 198, 151_645, 198, 151_644, 872, 198, 151_669]
            + Array(repeating: Int32(151_676), count: numAudioFrames)
            + [151_670, 151_645, 198, 151_644, 77_091, 198]
    }

    func testAutoLanguageMatchesOfficialTemplate() {
        XCTAssertEqual(Qwen3AsrPrompt.promptTokens(numAudioFrames: 3, language: nil), officialBase(numAudioFrames: 3))
    }

    func testForcedLanguageAppendsOfficialPrefixAfterAssistantHeader() {
        let tokens = Qwen3AsrPrompt.promptTokens(numAudioFrames: 2, language: .japanese)
        // `language Japanese<asr_text>`
        XCTAssertEqual(tokens, officialBase(numAudioFrames: 2) + [11_528, 10_769, 151_704])
    }

    func testSystemTurnStaysEmptyForEveryLanguage() {
        let emptySystem: [Int32] = [151_644, 8948, 198, 151_645, 198]
        for language in Qwen3AsrConfig.Language.allCases {
            let tokens = Qwen3AsrPrompt.promptTokens(numAudioFrames: 1, language: language)
            XCTAssertEqual(Array(tokens.prefix(emptySystem.count)), emptySystem, "\(language)")
        }
    }

    func testEveryLanguageHasForcedPrefix() {
        for language in Qwen3AsrConfig.Language.allCases {
            guard let prefix = Qwen3AsrPrompt.forcedLanguagePrefixTokens[language] else {
                XCTFail("missing forced prefix for \(language)")
                continue
            }
            XCTAssertEqual(prefix.first, Qwen3AsrPrompt.languageWordTokenId, "\(language) must start with `language`")
            XCTAssertEqual(prefix.last, asrText, "\(language) must end with <asr_text>")
            XCTAssertGreaterThanOrEqual(prefix.count, 3, "\(language) needs at least one name token")
            let nameTokens = prefix.dropFirst().dropLast()
            XCTAssertFalse(nameTokens.contains(asrText), "\(language)")
            XCTAssertTrue(
                nameTokens.allSatisfy { $0 >= 0 && $0 < 151_643 }, "\(language) name must be ordinary BPE tokens")
        }
    }

    func testForcedPrefixesAreDistinct() {
        let prefixes = Qwen3AsrConfig.Language.allCases.compactMap { Qwen3AsrPrompt.forcedLanguagePrefixTokens[$0] }
        XCTAssertEqual(Set(prefixes).count, prefixes.count)
    }

    func testAutoLanguagePromptHasNoAsrTextToken() {
        XCTAssertFalse(Qwen3AsrPrompt.promptTokens(numAudioFrames: 4, language: nil).contains(asrText))
    }
}
