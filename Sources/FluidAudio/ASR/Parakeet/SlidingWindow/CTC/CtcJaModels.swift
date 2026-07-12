@preconcurrency import CoreML
import Foundation

/// Configuration for Japanese CTC models (`FluidInference/parakeet-0.6b-ja-coreml`).
public enum CtcJaConfig: ParakeetLanguageModelConfig {
    public static let blankId: Int = 3072
    public static let repository: Repo = .parakeetJa
    public static let languageLabel: String = "CTC ja"
    public static let loggerCategory: String = "CtcJaModels"

    public static let preprocessorFile: String = ModelNames.CTCJa.preprocessorFile
    public static let encoderFile: String = ModelNames.CTCJa.encoderFile
    public static let decoderFile: String = ModelNames.CTCJa.decoderFile
    public static let vocabularyFile: String = ModelNames.CTCJa.vocabularyFile
    public static let jointFile: String? = nil

    public static let supportsInt8Encoder: Bool = false
    public static let encoderFp32File: String? = nil
}

/// Container for Parakeet CTC Japanese CoreML models (full pipeline).
public typealias CtcJaModels = ParakeetLanguageModels<CtcJaConfig>
