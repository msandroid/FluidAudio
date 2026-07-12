import Foundation

/// Greedy vocabulary lookup tokenization for CTC custom vocabulary (ja / SentencePiece exports).
public enum CtcVocabularyTokenizer: Sendable {

    /// Tokenize `text` into CTC token ids using longest-match greedy scan over `vocabulary`.
    public static func tokenize(_ text: String, vocabulary: [Int: String]) -> [Int] {
        var tokenToId: [String: Int] = [:]
        tokenToId.reserveCapacity(vocabulary.count)
        for (id, token) in vocabulary {
            tokenToId[token] = id
        }

        let normalizedText = text.lowercased()
        guard !normalizedText.isEmpty else { return [] }

        var result: [Int] = []
        var position = normalizedText.startIndex
        var isWordStart = true

        while position < normalizedText.endIndex {
            var matched = false
            let remaining = normalizedText.distance(from: position, to: normalizedText.endIndex)
            var matchLength = min(20, remaining)

            while matchLength > 0 {
                let endPos = normalizedText.index(position, offsetBy: matchLength)
                let substring = String(normalizedText[position..<endPos])
                let withPrefix =
                    isWordStart ? ASRConstants.sentencePieceWordBoundary + substring : substring

                if let tokenId = tokenToId[withPrefix] {
                    result.append(tokenId)
                    position = endPos
                    isWordStart = false
                    matched = true
                    break
                }
                if let tokenId = tokenToId[substring] {
                    result.append(tokenId)
                    position = endPos
                    isWordStart = false
                    matched = true
                    break
                }

                matchLength -= 1
            }

            if !matched {
                let char = normalizedText[position]
                if char == " " {
                    isWordStart = true
                } else {
                    isWordStart = false
                }
                position = normalizedText.index(after: position)
            }
        }

        return result
    }

    /// Build a ``CustomVocabularyContext`` with pre-tokenized ``CustomVocabularyTerm/ctcTokenIds``.
    public static func buildContext(
        terms: [String],
        vocabulary: [Int: String],
        minTermLength: Int = 2,
        weight: Float = 10.0
    ) -> CustomVocabularyContext {
        var vocabTerms: [CustomVocabularyTerm] = []
        vocabTerms.reserveCapacity(terms.count)

        for raw in terms {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.count >= minTermLength else { continue }
            let tokenIds = tokenize(trimmed, vocabulary: vocabulary)
            guard !tokenIds.isEmpty else { continue }
            vocabTerms.append(
                CustomVocabularyTerm(
                    text: trimmed,
                    weight: weight,
                    aliases: nil,
                    tokenIds: nil,
                    ctcTokenIds: tokenIds
                )
            )
        }

        return CustomVocabularyContext(terms: vocabTerms, minTermLength: minTermLength)
    }
}
