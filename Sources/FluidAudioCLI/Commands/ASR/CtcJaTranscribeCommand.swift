#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

enum CtcJaTranscribeCommand {
    private static let logger = AppLogger(category: "CtcJaTranscribe")

    static func run(arguments: [String]) async {
        var audioPath: String?
        var verbose = false

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--verbose", "-v":
                verbose = true
            case "--help", "-h":
                printUsage()
                return
            default:
                if audioPath == nil {
                    audioPath = arg
                }
            }
            i += 1
        }

        guard let audioPath = audioPath else {
            logger.error("Error: No audio file specified")
            printUsage()
            return
        }

        let audioURL = URL(fileURLWithPath: audioPath)
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            logger.error("Error: Audio file not found: \(audioPath)")
            return
        }

        do {
            logger.info("Loading CTC ja (Japanese) models...")

            let manager = try await CtcJaManager.load(
                progressHandler: verbose ? createProgressHandler() : nil
            )

            logger.info("Transcribing: \(audioPath)")

            let startTime = Date()
            let text = try await manager.transcribe(audioURL: audioURL)
            let elapsed = Date().timeIntervalSince(startTime)

            logger.info("Transcription completed in \(String(format: "%.2f", elapsed))s")
            logger.info("")
            logger.info("Result:")
            print(text)

        } catch {
            logger.error("Transcription failed: \(error.localizedDescription)")
            if verbose {
                logger.error("Error details: \(String(describing: error))")
            }
        }
    }

    private static func createProgressHandler() -> DownloadUtils.ProgressHandler {
        return { progress in
            let percentage = progress.fractionCompleted * 100.0
            switch progress.phase {
            case .listing:
                logger.info("Listing files from repository...")
            case .downloading(let completed, let total):
                logger.info(
                    "Downloading models: \(completed)/\(total) files (\(String(format: "%.1f", percentage))%)"
                )
            case .compiling(let modelName):
                logger.info("Compiling \(modelName)...")
            }
        }
    }

    private static func printUsage() {
        logger.info(
            """
            CTC ja Transcribe - Japanese speech recognition

            Usage: fluidaudiocli ctc-ja-transcribe <audio_file> [options]

            Arguments:
                <audio_file>    Path to audio file (WAV, MP3, M4A, etc.)

            Options:
                --verbose, -v   Show download progress and detailed logs
                --help, -h      Show this help message

            Examples:
                # Basic transcription
                fluidaudiocli ctc-ja-transcribe audio.mp3

                # Show download progress
                fluidaudiocli ctc-ja-transcribe audio.mp3 --verbose

            Model Info:
                - Language: Japanese (日本語)
                - Architecture: Hybrid FastConformer-TDT-CTC
                - Vocabulary: 3,072 SentencePiece BPE tokens
                - Max audio: 15 seconds per chunk
                - Model: FluidInference/parakeet-0.6b-ja-coreml

            Performance:
                - CER: ~10.29% on FLEURS Japanese
                - RTFx: ~136.85x on M-series chips

            Note: Models auto-download from HuggingFace on first use (~1-2 GB).
            """
        )
    }
}
#endif
