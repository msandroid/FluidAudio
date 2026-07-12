import Foundation

/// Applies CTC keyword detections to a greedy transcript (substring fuzzy replace; Japanese-safe).
public enum CtcTranscriptVocabularyBiasing: Sendable {

  /// Minimum acoustic score for a detection to influence the transcript.
  public static let defaultMinDetectionScore: Float = ContextBiasingConstants.defaultMinSpotterScore

  /// Minimum normalized similarity (0…1) between a transcript span and a vocabulary term.
  public static let defaultMinReplaceSimilarity: Float = 0.62

  /// Replace near-miss spans in `transcript` when acoustic evidence supports vocabulary terms.
  public static func applyDetections(
    to transcript: String,
    detections: [CtcKeywordSpotter.KeywordDetection],
    minDetectionScore: Float = defaultMinDetectionScore,
    minReplaceSimilarity: Float = defaultMinReplaceSimilarity
  ) -> String {
    let valid = detections
      .filter { $0.score >= minDetectionScore }
      .sorted { $0.term.text.count > $1.term.text.count }

    guard !valid.isEmpty else { return transcript }

    var text = transcript
    var appliedTerms = Set<String>()

    for detection in valid {
      let keyword = detection.term.text
      let key = keyword.lowercased()
      guard !appliedTerms.contains(key) else { continue }

      if text.contains(keyword) {
        appliedTerms.insert(key)
        continue
      }

      guard let replacement = bestFuzzyReplacement(
        in: text,
        target: keyword,
        minSimilarity: minReplaceSimilarity
      ) else {
        continue
      }

      if let range = text.range(of: replacement.span) {
        text.replaceSubrange(range, with: keyword)
        appliedTerms.insert(key)
      }
    }

    return text
  }

  // MARK: - Fuzzy span search

  private struct FuzzyMatch {
    let span: String
    let similarity: Float
  }

  private static func bestFuzzyReplacement(
    in transcript: String,
    target: String,
    minSimilarity: Float
  ) -> FuzzyMatch? {
    let trimmedTarget = target.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedTarget.isEmpty, !transcript.isEmpty else { return nil }

    let targetLen = trimmedTarget.count
    let minLen = max(1, targetLen - 2)
    let maxLen = targetLen + 2
    var best: FuzzyMatch?

    let chars = Array(transcript)
    let n = chars.count
    guard n >= minLen else { return nil }

    for length in minLen...min(maxLen, n) {
      for start in 0...(n - length) {
        let span = String(chars[start..<(start + length)])
        let normalizedSpan = normalizeForCompare(span)
        guard !normalizedSpan.isEmpty else { continue }

        let similarity = stringSimilarity(normalizedSpan, normalizeForCompare(trimmedTarget))
        guard similarity >= minSimilarity else { continue }

        if let current = best {
          if similarity > current.similarity {
            best = FuzzyMatch(span: span, similarity: similarity)
          }
        } else {
          best = FuzzyMatch(span: span, similarity: similarity)
        }
      }
    }

    return best
  }

  private static func normalizeForCompare(_ text: String) -> String {
    text
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
      .replacingOccurrences(of: ASRConstants.sentencePieceWordBoundary, with: "")
  }

  private static func stringSimilarity(_ a: String, _ b: String) -> Float {
    guard !a.isEmpty, !b.isEmpty else { return a.isEmpty && b.isEmpty ? 1 : 0 }
    let distance = StringUtils.levenshteinDistance(a, b)
    let maxLen = max(a.count, b.count)
    guard maxLen > 0 else { return 1 }
    return 1 - Float(distance) / Float(maxLen)
  }
}
