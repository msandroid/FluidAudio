import XCTest
@testable import FluidAudio

final class StreamingEouAsrManagerApproachingTests: XCTestCase {
    func testApproachingForcedEouTokenThresholdUsesFraction() async {
        let manager = StreamingEouAsrManager(
            maxTokensBeforeForcedEou: 480,
            debugFeatures: false
        )
        let threshold = await manager.approachingForcedEouTokenThreshold()
        XCTAssertEqual(threshold, 384)
    }

    func testSentenceBoundaryTokenSplitIndexAtPeriod() throws {
        let vocabURL = try makeTempVocab([
            "1": "one",
            "2": "\u{2581}two",
            "3": ".",
            "4": "\u{2581}three",
            "5": "\u{2581}four",
            "6": "\u{2581}five",
        ])
        let tokenizer = try Tokenizer(vocabPath: vocabURL)
        let ids = [1, 2, 3, 4, 5, 6]
        let split = StreamingEouAsrManager.sentenceBoundaryTokenSplitIndex(ids: ids, tokenizer: tokenizer)
        XCTAssertEqual(split, 3)
    }

    func testSentenceBoundaryTokenSplitIndexRejectsShortRemainder() throws {
        let vocabURL = try makeTempVocab([
            "1": "hello",
            "2": ".",
            "3": "tail",
        ])
        let tokenizer = try Tokenizer(vocabPath: vocabURL)
        let ids = [1, 2, 3]
        let split = StreamingEouAsrManager.sentenceBoundaryTokenSplitIndex(
            ids: ids,
            tokenizer: tokenizer,
            minRemainderTokens: 3
        )
        XCTAssertNil(split)
    }

    private func makeTempVocab(_ entries: [String: String]) throws -> URL {
        let data = try JSONSerialization.data(withJSONObject: entries, options: [])
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("parakeet-eou-test-vocab-\(UUID().uuidString).json")
        try data.write(to: url)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url
    }
}
