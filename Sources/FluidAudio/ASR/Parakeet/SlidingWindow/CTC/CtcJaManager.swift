@preconcurrency import CoreML
import Foundation

/// Manager for Parakeet CTC Japanese transcription.
///
/// Pipeline (FluidInference `parakeet-0.6b-ja-coreml`; feature names resolved from loaded graphs):
/// 1. Preprocessor: `audio_signal` + (`length` | `audio_length`) → `mel_features` + `mel_length`
/// 2. Encoder: `mel_features` + `mel_length` → (`encoder` | `encoder_output`)
/// 3. CtcDecoder: encoder tensor → `ctc_logits` (raw logits)
/// 4. Greedy CTC decode via `ctcGreedyDecodeFromRawLogits`
public struct CtcJaTranscriptionResult: Sendable {
    public let text: String
    /// CTC log-probabilities `[T, V]` after log-softmax and blank bias.
    public let logProbs: [[Float]]
    /// Duration of each CTC frame in seconds.
    public let frameDuration: Double

    public init(text: String, logProbs: [[Float]], frameDuration: Double) {
        self.text = text
        self.logProbs = logProbs
        self.frameDuration = frameDuration
    }
}

public actor CtcJaManager {

    private let models: CtcJaModels
    private let maxAudioSamples: Int
    private let sampleRate: Int
    private let preprocessorLengthFeatureKey: String
    private let preprocessorMelOutputName: String
    private let preprocessorMelLengthOutputName: String
    private let encoderMelInputName: String
    private let encoderMelLengthInputName: String
    private let encoderOutputName: String
    private let ctcDecoderInputName: String
    private let ctcLogitsOutputName: String

    private static let logger = AppLogger(category: "CtcJaManager")

    public init(models: CtcJaModels, maxAudioSamples: Int = 240_000, sampleRate: Int = 16_000) {
        self.models = models
        self.maxAudioSamples = maxAudioSamples
        self.sampleRate = sampleRate

        let preInputs = Set(models.preprocessor.modelDescription.inputDescriptionsByName.keys)
        if preInputs.contains("audio_length") {
            preprocessorLengthFeatureKey = "audio_length"
        } else if preInputs.contains("length") {
            preprocessorLengthFeatureKey = "length"
        } else {
            preprocessorLengthFeatureKey = "length"
        }

        preprocessorMelOutputName = Self.multiArrayOutputName(
            from: models.preprocessor,
            preferred: ["mel_features", "mel", "melspectrogram_features"]
        ) ?? "mel_features"
        preprocessorMelLengthOutputName = Self.multiArrayOutputName(
            from: models.preprocessor,
            preferred: ["mel_length", "length"]
        ) ?? "mel_length"
        encoderMelInputName = Self.multiArrayInputName(
            from: models.encoder,
            preferred: ["mel_features", "mel", "audio_signal"]
        ) ?? "mel_features"
        encoderMelLengthInputName = Self.multiArrayInputName(
            from: models.encoder,
            preferred: ["mel_length", "length"]
        ) ?? "mel_length"
        encoderOutputName = Self.multiArrayOutputName(
            from: models.encoder,
            preferred: ["encoder_output", "encoder", "encoded", "output", "y"]
        ) ?? "encoder"
        ctcDecoderInputName = Self.multiArrayInputName(
            from: models.decoder,
            preferred: ["encoder_output", "encoder", "input", "x", "features"]
        ) ?? encoderOutputName
        ctcLogitsOutputName = Self.multiArrayOutputName(
            from: models.decoder,
            preferred: ["ctc_logits", "logits", "output", "y"]
        ) ?? "ctc_logits"
    }

    public static func load(
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CtcJaManager {
        let models = try await CtcJaModels.downloadAndLoad(
            configuration: configuration,
            progressHandler: progressHandler
        )
        return CtcJaManager(models: models)
    }

    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CtcJaManager {
        let models = try await CtcJaModels.load(
            from: directory,
            configuration: configuration,
            progressHandler: progressHandler
        )
        return CtcJaManager(models: models)
    }

    public func transcribe(
        audio: [Float],
        audioLength: Int? = nil
    ) throws -> String {
        let actualLength = audioLength ?? audio.count
        let paddedAudio = padOrTruncateAudio(audio, targetLength: maxAudioSamples)

        let melOutput = try runPreprocessor(audio: paddedAudio, audioLength: actualLength)
        let encoderOutput = try runEncoder(mel: melOutput.mel, melLength: melOutput.melLength)
        let ctcLogits = try runCtcDecoder(encoderOutput: encoderOutput)

        return ctcGreedyDecodeFromRawLogits(
            ctcLogits,
            vocabulary: models.vocabulary,
            blankId: models.blankId
        )
    }

    public func transcribe(audioURL: URL) throws -> String {
        let converter = AudioConverter(sampleRate: Double(sampleRate))
        let samples = try converter.resampleAudioFile(audioURL)
        return try transcribe(audio: samples)
    }

    /// Parakeet CTC ja vocabulary (token id → piece) for custom-term tokenization.
    public var vocabulary: [Int: String] {
        models.vocabulary
    }

    /// CTC blank token id (3072 for ja).
    public var blankId: Int {
        models.blankId
    }

    /// Greedy decode plus CTC log-probabilities for keyword spotting / vocabulary biasing.
    public func transcribeWithLogProbs(
        audio: [Float],
        audioLength: Int? = nil
    ) throws -> CtcJaTranscriptionResult {
        let actualLength = audioLength ?? audio.count
        let paddedAudio = padOrTruncateAudio(audio, targetLength: maxAudioSamples)

        let melOutput = try runPreprocessor(audio: paddedAudio, audioLength: actualLength)
        let encoderOutput = try runEncoder(mel: melOutput.mel, melLength: melOutput.melLength)
        let ctcLogits = try runCtcDecoder(encoderOutput: encoderOutput)

        let text = ctcGreedyDecodeFromRawLogits(
            ctcLogits,
            vocabulary: models.vocabulary,
            blankId: models.blankId
        )

        let rawLogProbs = extractLogProbs(from: ctcLogits)
        let trimmed = trimJaLogProbs(rawLogProbs, audioSampleCount: min(actualLength, maxAudioSamples))
        let frameCount = trimmed.count
        let frameDuration =
            frameCount > 0
            ? Double(min(actualLength, maxAudioSamples)) / Double(frameCount) / Double(sampleRate)
            : 0

        return CtcJaTranscriptionResult(
            text: text,
            logProbs: trimmed,
            frameDuration: frameDuration
        )
    }

    // MARK: - Private Pipeline

    private struct MelOutput {
        let mel: MLMultiArray
        let melLength: MLMultiArray
    }

    private func runPreprocessor(audio: [Float], audioLength _: Int) throws -> MelOutput {
        let audioArray = try MLMultiArray(shape: [1, maxAudioSamples as NSNumber], dataType: .float32)
        audioArray.withUnsafeMutableBytes { rawBuffer, _ in
            guard let base = rawBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
            let copyCount = min(audio.count, maxAudioSamples)
            if copyCount > 0 {
                audio.withUnsafeBufferPointer { src in
                    base.initialize(from: src.baseAddress!, count: copyCount)
                }
            }
            if copyCount < maxAudioSamples {
                (base + copyCount).initialize(repeating: 0, count: maxAudioSamples - copyCount)
            }
        }

        // Parakeet ja Preprocessor is a fixed ~15s Core ML graph. Short `audio_length` values
        // trigger `ios17.slice_by_index: zero shape` and omit mel outputs on device.
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: maxAudioSamples)

        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                "audio_signal": MLFeatureValue(multiArray: audioArray),
                preprocessorLengthFeatureKey: MLFeatureValue(multiArray: lengthArray),
            ]
        )
        let output = try models.preprocessor.prediction(from: input)

        guard
            let mel = output.featureValue(for: preprocessorMelOutputName)?.multiArrayValue,
            let melLength = output.featureValue(for: preprocessorMelLengthOutputName)?.multiArrayValue
        else {
            let available = output.featureNames.sorted().joined(separator: ", ")
            throw ASRError.processingFailed(
                "Failed to extract \(preprocessorMelOutputName) or \(preprocessorMelLengthOutputName) from preprocessor output (available: \(available))"
            )
        }

        return MelOutput(mel: mel, melLength: melLength)
    }

    private func runEncoder(mel: MLMultiArray, melLength: MLMultiArray) throws -> MLMultiArray {
        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                encoderMelInputName: MLFeatureValue(multiArray: mel),
                encoderMelLengthInputName: MLFeatureValue(multiArray: melLength),
            ]
        )
        let output = try models.encoder.prediction(from: input)

        guard let encoderOutput = output.featureValue(for: encoderOutputName)?.multiArrayValue else {
            throw ASRError.processingFailed("Failed to extract \(encoderOutputName) from encoder")
        }

        return encoderOutput
    }

    private func runCtcDecoder(encoderOutput: MLMultiArray) throws -> MLMultiArray {
        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                ctcDecoderInputName: MLFeatureValue(multiArray: encoderOutput)
            ]
        )
        let output = try models.decoder.prediction(from: input)

        guard let ctcLogits = output.featureValue(for: ctcLogitsOutputName)?.multiArrayValue else {
            throw ASRError.processingFailed("Failed to extract \(ctcLogitsOutputName) from CtcDecoder")
        }

        return ctcLogits
    }

    private static func multiArrayOutputName(from model: MLModel, preferred: [String]) -> String? {
        let out = model.modelDescription.outputDescriptionsByName
        for name in preferred where out[name]?.type == .multiArray {
            return name
        }
        return out.first { $0.value.type == .multiArray }?.key
    }

    private static func multiArrayInputName(from model: MLModel, preferred: [String]) -> String? {
        let input = model.modelDescription.inputDescriptionsByName
        for name in preferred where input[name]?.type == .multiArray {
            return name
        }
        return input.first { $0.value.type == .multiArray }?.key
    }

    private func extractLogProbs(from ctcLogits: MLMultiArray) -> [[Float]] {
        let rank = ctcLogits.shape.count
        guard rank >= 3 else { return [] }

        let d1 = ctcLogits.shape[1].intValue
        let d2 = ctcLogits.shape[2].intValue
        let vocabSize = models.blankId + 1

        let strides = ctcLogits.strides.map { $0.intValue }
        guard strides.count >= 3 else { return [] }
        let s1 = strides[1]
        let s2 = strides[2]
        let ptr = ctcLogits.dataPointer.assumingMemoryBound(to: Float.self)

        var rawRows: [[Float]] = []

        if d2 == vocabSize {
            let timeSteps = d1
            rawRows.reserveCapacity(timeSteps)
            for t in 0..<timeSteps {
                let base = t * s1
                var row = [Float](repeating: 0, count: vocabSize)
                for v in 0..<vocabSize {
                    row[v] = ptr[base + v * s2]
                }
                rawRows.append(row)
            }
        } else if d1 == vocabSize {
            let timeSteps = d2
            rawRows.reserveCapacity(timeSteps)
            for t in 0..<timeSteps {
                var row = [Float](repeating: 0, count: vocabSize)
                for v in 0..<vocabSize {
                    row[v] = ptr[v * s1 + t * s2]
                }
                rawRows.append(row)
            }
        }

        return CtcKeywordSpotter.applyLogSoftmax(
            rawLogits: rawRows,
            blankId: models.blankId
        )
    }

    private func trimJaLogProbs(_ logProbs: [[Float]], audioSampleCount: Int) -> [[Float]] {
        guard !logProbs.isEmpty else { return logProbs }
        let totalFrames = logProbs.count
        if audioSampleCount >= maxAudioSamples {
            return logProbs
        }
        let samplesPerFrame = Double(maxAudioSamples) / Double(totalFrames)
        let validFrames = Int(ceil(Double(audioSampleCount) / samplesPerFrame))
        let clampedFrames = max(1, min(validFrames, totalFrames))
        return Array(logProbs.prefix(clampedFrames))
    }

    private func padOrTruncateAudio(_ audio: [Float], targetLength: Int) -> [Float] {
        var result = audio
        if result.count < targetLength {
            result.append(contentsOf: Array(repeating: 0.0, count: targetLength - result.count))
        } else if result.count > targetLength {
            result = Array(result.prefix(targetLength))
        }
        return result
    }
}
