import Foundation

/// CTC keyword spotting from pre-computed log-probabilities (model-agnostic blank id).
public struct CtcVocabularySpotter: Sendable {

    public let blankId: Int

    public init(blankId: Int) {
        self.blankId = blankId
    }

    /// Spot vocabulary terms in CTC log-probs (no additional Core ML inference).
    public func spotKeywords(
        logProbs: [[Float]],
        frameDuration: Double,
        customVocabulary: CustomVocabularyContext,
        minScore: Float? = nil
    ) -> [CtcKeywordSpotter.KeywordDetection] {
        let totalFrames = logProbs.count
        guard totalFrames > 0 else { return [] }

        var results: [CtcKeywordSpotter.KeywordDetection] = []

        for term in customVocabulary.terms {
            guard term.text.count >= customVocabulary.minTermLength else { continue }
            let ids = term.ctcTokenIds ?? term.tokenIds
            guard let ids, !ids.isEmpty else { continue }

            let tokenCount = ids.count
            let adjustedThreshold: Float =
                minScore.map { base in
                    let extraTokens = max(0, tokenCount - ContextBiasingConstants.baselineTokenCountForThreshold)
                    return base - Float(extraTokens) * ContextBiasingConstants.thresholdRelaxationPerToken
                } ?? ContextBiasingConstants.defaultMinSpotterScore

            let multipleDetections = CtcDPAlgorithm.ctcWordSpotMultiple(
                logProbs: logProbs,
                keywordTokens: ids,
                minScore: adjustedThreshold,
                mergeOverlap: true,
                blankId: blankId
            )

            for (score, start, end) in multipleDetections {
                let startTime = TimeInterval(start) * frameDuration
                let endTime = TimeInterval(end) * frameDuration
                results.append(
                    CtcKeywordSpotter.KeywordDetection(
                        term: term,
                        score: score,
                        totalFrames: totalFrames,
                        startFrame: start,
                        endFrame: end,
                        startTime: startTime,
                        endTime: endTime
                    )
                )
            }
        }

        return results
    }
}
