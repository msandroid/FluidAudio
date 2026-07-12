#if os(iOS) || os(macOS)
import CoreML
import Foundation

/// Nemotron Speech Streaming ASR — shared loader (same policy as `fluidaudio nemotron-transcribe` / TranslateBlue `NemotronASRService`).
public struct NemotronAsrCommand {
    /// Hub / cache folder basename for the 80 ms chunk variant (skips Neural Engine in Core ML fallback).
    public static let neuralEngineSkippedVariantDirectoryName = "nemotron_coreml_80ms"

    /// `true` when `modelDir` contains the files `StreamingNemotronAsrManager.loadModels(modelDir:)` expects.
    public static func isVariantDirectoryComplete(at directory: URL) -> Bool {
        let dir = directory.standardizedFileURL
        let fm = FileManager.default
        let metadata = dir.appendingPathComponent("metadata.json").path
        let tokenizer = dir.appendingPathComponent("tokenizer.json").path
        guard fm.fileExists(atPath: metadata), fm.fileExists(atPath: tokenizer) else { return false }

        // Hub Core ML bundles may ship ML Program weights (`weights/weight.bin`) without legacy `coremldata.bin`.
        // Reject truncated downloads before Core ML fails with "Could not open ... weight.bin".
        func mlmodelcReady(_ relative: String, minimumWeightBytes: Int64) -> Bool {
            let bundle = dir.appendingPathComponent(relative, isDirectory: true)
            let weightBin = bundle.appendingPathComponent("weights/weight.bin", isDirectory: false)
            let legacy = bundle.appendingPathComponent("coremldata.bin", isDirectory: false)
            if fm.fileExists(atPath: weightBin.path) {
                guard let attrs = try? fm.attributesOfItem(atPath: weightBin.path),
                      let sizeNum = attrs[.size] as? NSNumber,
                      sizeNum.int64Value >= minimumWeightBytes else { return false }
                return Self.canReadFirstByte(of: weightBin)
            }
            if fm.fileExists(atPath: legacy.path) {
                guard let attrs = try? fm.attributesOfItem(atPath: legacy.path),
                      let sizeNum = attrs[.size] as? NSNumber,
                      sizeNum.int64Value > 0 else { return false }
                return Self.canReadFirstByte(of: legacy)
            }
            return false
        }

        // Size floors vs FluidInference model card (approximate; catches empty Git-LFS pointers / partial sync).
        return mlmodelcReady("preprocessor.mlmodelc", minimumWeightBytes: 32 * 1024)
            && mlmodelcReady("decoder.mlmodelc", minimumWeightBytes: 512 * 1024)
            && mlmodelcReady("joint.mlmodelc", minimumWeightBytes: 256 * 1024)
            && mlmodelcReady("encoder/encoder_int8.mlmodelc", minimumWeightBytes: 20 * 1024 * 1024)
    }

    private static func canReadFirstByte(of url: URL) -> Bool {
        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            _ = try handle.read(upToCount: 1)
            return true
        } catch {
            return false
        }
    }

    /// Loads `StreamingNemotronAsrManager` using the same Core ML compute-unit fallback as the TranslateBlue app.
    public static func loadStreamingManager(modelDir: URL) async throws -> StreamingNemotronAsrManager {
        let dir = modelDir.standardizedFileURL
        guard isVariantDirectoryComplete(at: dir) else {
            throw ASRError.modelLoadFailed
        }
        let variantName = dir.lastPathComponent
        let needsNeuralEngineSkip = variantName == neuralEngineSkippedVariantDirectoryName
        let computeCandidates: [MLComputeUnits] = needsNeuralEngineSkip
            ? [.cpuAndGPU, .cpuOnly]
            : [.cpuAndNeuralEngine, .cpuAndGPU, .cpuOnly]
        var lastError: Error?
        for units in computeCandidates {
            let mlConfig = MLModelConfiguration()
            mlConfig.computeUnits = units
            let manager = StreamingNemotronAsrManager(configuration: mlConfig)
            do {
                try await manager.loadModels(from: dir)
                return manager
            } catch {
                lastError = error
            }
        }
        throw lastError ?? ASRError.modelLoadFailed
    }
}
#endif
