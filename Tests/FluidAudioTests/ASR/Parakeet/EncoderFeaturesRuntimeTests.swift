import AVFoundation
@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// RUNTIME verification for opt-in encoder-feature export against the REAL
/// Parakeet TDT v3 CoreML models in the local FluidAudio cache.
///
/// This is intentionally NOT part of the normal unit run — it requires the models
/// on disk and a specific WAV fixture, and it runs CoreML inference. Gated behind
/// the `PARLEQ_ENCODER_RUNTIME=1` environment variable so a plain `swift test` skips
/// it. Run explicitly with:
///
///     PARLEQ_ENCODER_RUNTIME=1 swift test --filter EncoderFeaturesRuntimeTests
///
/// It loads the models, transcribes the fixture WAV with `captureEncoderFeatures`
/// enabled, computes `pooledEmbedding` for three time spans, and writes them as JSON
/// for cross-checking against a Python reference.
final class EncoderFeaturesRuntimeTests: XCTestCase {

    private let modelsDirOnDisk = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3")

    private let wavPath = "/Users/jonyoder/.parleq/flywheel/audio/03785502-F858-4B30-A02D-048617F297E6.wav"

    private let outputPath =
        "/private/tmp/claude-501/-Users-jonyoder-Dev-parleq-speech/2511b616-12ec-4462-9ddb-231fd081f549/scratchpad/swift_emb.json"

    func testEncoderEmbeddingsAgainstRealModels() async throws {
        guard ProcessInfo.processInfo.environment["PARLEQ_ENCODER_RUNTIME"] == "1" else {
            throw XCTSkip("Set PARLEQ_ENCODER_RUNTIME=1 to run the real-model runtime verification.")
        }

        let fm = FileManager.default
        try XCTSkipUnless(
            fm.fileExists(atPath: modelsDirOnDisk.path), "Models not found at \(modelsDirOnDisk.path)")
        try XCTSkipUnless(fm.fileExists(atPath: wavPath), "WAV fixture not found at \(wavPath)")

        // The fork's loader resolves the repo cache as
        // <parent>/<Repo.parakeetV3.folderName> == ".../parakeet-tdt-0.6b-v3-coreml".
        // The on-disk cache (Parleq's naming) is ".../parakeet-tdt-0.6b-v3". Bridge the
        // two by building a temp parent dir whose expected-named subdir symlinks to the
        // real cache, so loading is fully offline (no download).
        let expectedFolder = Repo.parakeetV3.folderName  // "parakeet-tdt-0.6b-v3-coreml"
        let tempParent = fm.temporaryDirectory.appendingPathComponent(
            "EncoderFeaturesRuntime-\(UUID().uuidString)")
        try fm.createDirectory(at: tempParent, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempParent) }

        let linkedRepoDir = tempParent.appendingPathComponent(expectedFolder)
        try fm.createSymbolicLink(at: linkedRepoDir, withDestinationURL: modelsDirOnDisk)

        // Force offline so a folder-name/precision mismatch never triggers a network
        // fetch — it should fail loudly instead.
        let priorOffline = DownloadUtils.enforceOffline
        DownloadUtils.enforceOffline = true
        defer { DownloadUtils.enforceOffline = priorOffline }

        let models = try await AsrModels.load(from: linkedRepoDir, version: .v3)

        let asrManager = AsrManager(config: .default)
        try await asrManager.loadModels(models)
        await asrManager.setCaptureEncoderFeatures(true)

        // Transcribe via the raw-samples path so we exercise the single-chunk path
        // for this ~8.4s clip.
        let samples = try AudioConverter().resampleAudioFile(path: wavPath)
        var decoderState = TdtDecoderState.make(decoderLayers: await asrManager.decoderLayerCount)
        decoderState.reset()

        let result = try await asrManager.transcribe(samples, decoderState: &decoderState)

        // Runtime layout facts (deferred until now).
        #if DEBUG
        let shape = await asrManager.lastEncoderArrayShape
        let strides = await asrManager.lastEncoderArrayStrides
        print("ENCODER_SHAPE=\(shape ?? [])")
        print("ENCODER_STRIDES=\(strides ?? [])")
        #endif

        guard let features = result.encoderFeatures else {
            return XCTFail("encoderFeatures was nil despite captureEncoderFeatures = true")
        }
        print("WINDOW_COUNT=\(features.windows.count)")
        print("HIDDEN_SIZE=\(features.hiddenSize)")
        print("SECONDS_PER_FRAME=\(features.secondsPerFrame)")
        print("TRANSCRIPT=\(result.text)")
        let totalFrames = features.windows.reduce(0) { $0 + $1.frames.count }
        print("TOTAL_FRAMES=\(totalFrames)")

        let spans: [(label: String, start: Double, end: Double)] = [
            ("3.28-3.60", 3.28, 3.60),
            ("5.52-6.00", 5.52, 6.00),
            ("7.60-8.00", 7.60, 8.00),
        ]

        var jsonObject: [String: [Float]] = [:]
        for span in spans {
            guard let pooled = features.pooledEmbedding(startSeconds: span.start, endSeconds: span.end)
            else {
                return XCTFail("pooledEmbedding(\(span.label)) returned nil")
            }
            XCTAssertEqual(pooled.count, 1024, "Embedding for \(span.label) should be 1024-dim")
            let first3 = pooled.prefix(3).map { $0 }
            print("EMB[\(span.label)] len=\(pooled.count) first3=\(first3)")
            jsonObject[span.label] = pooled
        }

        // Write the JSON file: { "3.28-3.60": [...1024 floats...], ... }
        // Use NaN-safe serialization via manual encoding to preserve full precision.
        var encodableMap: [String: [Double]] = [:]
        for (k, v) in jsonObject { encodableMap[k] = v.map { Double($0) } }
        let data = try JSONSerialization.data(
            withJSONObject: encodableMap, options: [.sortedKeys])
        let outURL = URL(fileURLWithPath: outputPath)
        try fm.createDirectory(
            at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: outURL)
        print("WROTE_JSON=\(outputPath)")

        XCTAssertTrue(fm.fileExists(atPath: outputPath))
    }
}
