@preconcurrency import CoreML
import Foundation

/// Manager for SenseVoiceSmall transcription.
///
/// Pipeline: waveform → [Preprocessor fp32/CPU] → 560-d features → pad to the
/// smallest enumerated encoder bucket → [encoder+CTC fp16/ANE] → greedy CTC
/// decode (drop blank 0, collapse) → SentencePiece detokenize → strip the
/// leading `<|lang|><|emo|><|event|><|itn|>` tags.
public actor SenseVoiceManager {

    private let models: SenseVoiceModels
    private let language: Int32
    private let textNorm: Int32
    private static let logger = AppLogger(category: "SenseVoiceManager")

    public init(
        models: SenseVoiceModels,
        language: Int32 = SenseVoiceConfig.defaultLanguage,
        textNorm: Int32 = SenseVoiceConfig.defaultTextNorm
    ) {
        self.models = models
        self.language = language
        self.textNorm = textNorm
    }

    /// Load models from the default cache (downloading if needed), then build a manager.
    public static func load(
        precision: SenseVoiceEncoderPrecision = .fp16,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> SenseVoiceManager {
        let models = try await SenseVoiceModels.downloadAndLoad(
            precision: precision, progressHandler: progressHandler)
        return SenseVoiceManager(models: models)
    }

    /// Transcribe a 16 kHz mono audio file.
    public func transcribe(audioURL: URL, language: Int32? = nil) throws -> String {
        let converter = AudioConverter(sampleRate: Double(SenseVoiceConfig.sampleRate))
        let samples = try converter.resampleAudioFile(audioURL)
        return try transcribe(audio: samples, language: language)
    }

    /// Transcribe 16 kHz mono float samples (in [-1, 1]).
    public func transcribe(audio: [Float], language: Int32? = nil) throws -> String {
        let features = try runPreprocessor(audio: audio)
        let (logits, validFrames) = try runEncoder(features: features, language: language)
        return decode(logits: logits, validFrames: validFrames)
    }

    // MARK: - Pipeline

    /// waveform [1, N] (scaled to int16 range) → features [1, T, 560].
    private func runPreprocessor(audio: [Float]) throws -> MLMultiArray {
        guard !audio.isEmpty else {
            throw ASRError.processingFailed("SenseVoice preprocessor received empty audio")
        }
        let minSamples = 3200
        let maxSamples = 480000
        let effectiveAudio: [Float]
        if audio.count < minSamples {
            effectiveAudio = audio + [Float](repeating: 0, count: minSamples - audio.count)
        } else if audio.count > maxSamples {
            effectiveAudio = Array(audio.suffix(maxSamples))
        } else {
            effectiveAudio = audio
        }
        let n = effectiveAudio.count
        let waveform = try MLMultiArray(shape: [1, n as NSNumber], dataType: .float32)
        let scale = SenseVoiceConfig.waveformScale
        let wptr = waveform.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<n { wptr[i] = effectiveAudio[i] * scale }

        let input = try MLDictionaryFeatureProvider(
            dictionary: ["waveform": MLFeatureValue(multiArray: waveform)])
        let out = try models.preprocessor.prediction(from: input)
        guard let features = out.featureValue(for: "features")?.multiArrayValue else {
            throw ASRError.processingFailed("SenseVoice preprocessor produced no `features`")
        }
        return features
    }

    /// features [1, T, 560] → (ctc_logits [1, bucket+4, V], validFrames = 4 + T).
    private func runEncoder(features: MLMultiArray, language: Int32? = nil) throws -> (MLMultiArray, Int) {
        let dim = SenseVoiceConfig.featureDim
        var t = features.shape.count >= 2 ? features.shape[1].intValue : 0
        guard t > 0 else {
            throw ASRError.processingFailed("SenseVoice preprocessor produced empty feature sequence")
        }
        if t > SenseVoiceConfig.maxFrames {
            Self.logger.warning("Audio exceeds max length; truncating \(t) → \(SenseVoiceConfig.maxFrames) frames")
            t = SenseVoiceConfig.maxFrames
        }
        let bucket = SenseVoiceConfig.pickBucket(forFrames: t)

        // Zero-padded [1, bucket, 560] with the first T feature frames copied in.
        let speech = try MLMultiArray(shape: [1, bucket as NSNumber, dim as NSNumber], dataType: .float32)
        let sptr = speech.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(sptr, 0, bucket * dim * MemoryLayout<Float32>.size)
        
        let totalElements = min(t * dim, features.count)
        if features.dataType == .float32 && features.strides.count >= 3 && features.strides[2].intValue == 1 {
            let copyBytes = totalElements * MemoryLayout<Float32>.size
            memcpy(sptr, features.dataPointer, copyBytes)
        } else if features.dataType == .float32 {
            let fptr = features.dataPointer.assumingMemoryBound(to: Float32.self)
            let fStride1 = features.strides.count >= 2 ? features.strides[1].intValue : dim
            for i in 0..<t {
                let srcBase = i * fStride1
                let dstBase = i * dim
                for j in 0..<dim {
                    sptr[dstBase + j] = fptr[srcBase + j]
                }
            }
        } else {
            for i in 0..<totalElements {
                sptr[i] = features[i].floatValue
            }
        }

        let lengths = try MLMultiArray(shape: [1], dataType: .int32)
        lengths[0] = NSNumber(value: t)
        let lang = try MLMultiArray(shape: [1], dataType: .int32)
        lang[0] = NSNumber(value: language ?? self.language)
        let tn = try MLMultiArray(shape: [1], dataType: .int32)
        tn[0] = NSNumber(value: textNorm)

        // Defense-in-depth, not a fix for a bug found in this file: every scalar
        // input here is already rank 1 by construction (`shape: [1]`), and that
        // was verified both statically and by an actual CoreML run against the
        // shipped encoder — this file has never produced the rank mismatch. The
        // real historical case of "speech_lengths must be of rank 1, instead got
        // rank 2" traced to a *different* caller: TranslateBlue's own legacy
        // fallback in `SenseVoiceASRService.swift` bound `inputDesc.keys.first`
        // from an unordered Swift `Dictionary` (hash order is randomized per
        // process) directly to the raw [1, N] waveform buffer — on the unlucky
        // launch where that key happened to be "speech_lengths", CoreML reported
        // exactly this message, and it read like a FluidAudio defect. Fixed
        // upstream in TranslateBlue at commit 4b614034 ("Fail the SenseVoice
        // legacy load instead of transcribing nothing"), which now refuses to
        // load a multi-input model through that single-input path at all.
        // These guards exist so that if a *future* change to this file (or a
        // caller building its own MLMultiArray by hand) ever reintroduces a
        // rank/shape mismatch here, it fails loudly at this call site with a
        // FluidAudio-attributed message instead of a bare, easily-misattributed
        // CoreML string.
        for (featureName, array, expectedShape) in [
            ("speech", speech, [1, bucket, dim]),
            ("speech_lengths", lengths, [1]),
            ("language", lang, [1]),
            ("textnorm", tn, [1]),
        ] {
            let actualShape = array.shape.map(\.intValue)
            guard actualShape == expectedShape else {
                throw ASRError.processingFailed(
                    "SenseVoice encoder input `\(featureName)` has shape \(actualShape), "
                        + "expected \(expectedShape) — internal FluidAudio construction bug"
                )
            }
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "speech": MLFeatureValue(multiArray: speech),
            "speech_lengths": MLFeatureValue(multiArray: lengths),
            "language": MLFeatureValue(multiArray: lang),
            "textnorm": MLFeatureValue(multiArray: tn),
        ])
        let out = try models.encoder.prediction(from: input)
        guard let logits = out.featureValue(for: "ctc_logits")?.multiArrayValue else {
            throw ASRError.processingFailed("SenseVoice encoder produced no `ctc_logits`")
        }
        return (logits, SenseVoiceConfig.numQueryTokens + t)
    }

    /// Greedy CTC over the first `validFrames` (drop blank 0, collapse repeats),
    /// detokenize, then strip the `<|...|>` meta tags.
    private func decode(logits: MLMultiArray, validFrames: Int) -> String {
        guard logits.shape.count >= 3 else { return "" }
        let vocab = logits.shape[2].intValue
        let frames = min(validFrames, logits.shape[1].intValue)
        guard vocab > 0, frames > 0 else { return "" }
        var ids: [Int] = []
        var prev = -1

        func appendArgmax(frameBase: (Int) -> Float) {
            var best = 0
            var bestVal = frameBase(0)
            for v in 1..<vocab {
                let x = frameBase(v)
                if x > bestVal {
                    bestVal = x
                    best = v
                }
            }
            if best != SenseVoiceConfig.blankId && best != prev { ids.append(best) }
            prev = best
        }

        let s1 = logits.strides.count >= 2 ? logits.strides[1].intValue : vocab
        if logits.dataType == .float32 {
            let p = logits.dataPointer.assumingMemoryBound(to: Float32.self)
            for t in 0..<frames {
                let base = t * s1
                appendArgmax { p[base + $0] }
            }
        } else if logits.dataType == .float16 {
            let p = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            for t in 0..<frames {
                let base = t * s1
                appendArgmax { Float(p[base + $0]) }
            }
        } else {
            for t in 0..<frames {
                appendArgmax { logits[[0, t as NSNumber, $0 as NSNumber]].floatValue }
            }
        }

        let raw = decodeCtcTokenIds(ids, vocabulary: models.vocabulary)
        return
            raw
            .replacingOccurrences(of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }
}
