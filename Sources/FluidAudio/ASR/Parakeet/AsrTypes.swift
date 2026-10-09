import Foundation

// MARK: - Configuration

public struct ASRConfig: Sendable {
    public let sampleRate: Int
    public let tdtConfig: TdtConfig

    /// Encoder hidden dimension (1024 for 0.6B, 512 for 110m)
    public let encoderHiddenSize: Int

    /// Number of long-form chunks to transcribe concurrently.
    /// Applies only to stateless chunked transcription paths.
    public let parallelChunkConcurrency: Int

    /// Enable streaming mode for large files to reduce memory usage.
    /// When enabled, files larger than `streamingThreshold` samples will be processed
    /// using streaming to maintain constant memory usage.
    public let streamingEnabled: Bool

    /// File size threshold in samples for enabling streaming.
    /// Files with more samples than this threshold will use streaming mode.
    /// Default: 480,000 samples (~30 seconds at 16kHz)
    public let streamingThreshold: Int

    /// 80ms mel-context prepend on non-first long-form chunks (PR #264
    /// blank-boundary fix). `nil` (default) resolves per model version:
    /// `false` on v3 — the no-mel path's silence-aligned chunk starts avoid
    /// both the multilingual drift (issue #594) and quiet-speech drops near
    /// long silence runs (issue #803) — and `true` elsewhere. Set via the
    /// `melChunkContext:` init parameter. See "Current Paths" in
    /// Documentation/ASR/LongTranscription.md.
    public let melChunkContextOverride: Bool?

    /// Legacy Boolean view of `melChunkContextOverride`. Reports the explicit
    /// setting, or the non-v3 default (`true`) when unset — it cannot see the
    /// version-aware resolution applied at model load (v3 resolves to `false`).
    @available(*, deprecated, renamed: "melChunkContextOverride")
    public var melChunkContext: Bool { melChunkContextOverride ?? true }

    /// Opt-in probe-then-commit chunking arbitration for the v3 + no-mel
    /// batch path (default `false`) — strategies, commitment rationale, and
    /// cost in "Current Paths" (Documentation/ASR/LongTranscription.md).
    public let dualDecodeArbitration: Bool

    /// Post-merge repair pass for chunk-seam content drops in long-form
    /// batch transcription (issue #758, default `true`) — mechanics, cost,
    /// and limitations in "Post-Merge Repair Pass"
    /// (Documentation/ASR/LongTranscription.md).
    public let seamGapRepair: Bool

    /// Minimum inter-token gap, in seconds, that triggers a seam-gap repair
    /// probe when `seamGapRepair` is enabled.
    public let seamGapRepairMinGapSeconds: Double

    /// End-align the final long-form batch window at the last speech-bearing
    /// frame, backfilling it with real audio decoded as a suppressed prefix
    /// (issue #747, default `true`). Set `false` to restore the pre-#800
    /// zero-padded final window. Applies to the default batch path on
    /// decoders that support prefix suppression; the opt-in
    /// `dualDecodeArbitration` path always end-aligns.
    public let endAlignFinalWindow: Bool

    public static let `default` = ASRConfig()

    public init(
        sampleRate: Int = 16000,
        tdtConfig: TdtConfig = .default,
        encoderHiddenSize: Int = ASRConstants.encoderHiddenSize,
        parallelChunkConcurrency: Int = 4,
        streamingEnabled: Bool = true,
        streamingThreshold: Int = 480_000,
        melChunkContext: Bool? = nil,
        dualDecodeArbitration: Bool = false,
        seamGapRepair: Bool = true,
        seamGapRepairMinGapSeconds: Double = 1.5,
        endAlignFinalWindow: Bool = true
    ) {
        self.sampleRate = sampleRate
        self.tdtConfig = tdtConfig
        self.encoderHiddenSize = encoderHiddenSize
        self.parallelChunkConcurrency = max(1, parallelChunkConcurrency)
        self.streamingEnabled = streamingEnabled
        self.streamingThreshold = streamingThreshold
        self.melChunkContextOverride = melChunkContext
        self.dualDecodeArbitration = dualDecodeArbitration
        self.seamGapRepair = seamGapRepair
        self.seamGapRepairMinGapSeconds = max(0.5, seamGapRepairMinGapSeconds)
        self.endAlignFinalWindow = endAlignFinalWindow
    }

    /// Resolve the mel-context tri-state against the loaded model version.
    func resolvedMelChunkContext(for modelVersion: AsrModelVersion?) -> Bool {
        melChunkContextOverride ?? (modelVersion?.isV3Family != true)
    }
}

// MARK: - Results

public struct ASRResult: Codable, Sendable {
    public let text: String
    public let confidence: Float
    public let duration: TimeInterval
    public let processingTime: TimeInterval
    public let tokenTimings: [TokenTiming]?
    public let performanceMetrics: ASRPerformanceMetrics?
    public let ctcDetectedTerms: [String]?
    public let ctcAppliedTerms: [String]?

    /// Opt-in acoustic encoder features for this transcription. Non-`nil` only when
    /// `AsrManager.captureEncoderFeatures` was enabled for the call that produced this
    /// result. Lets a consumer mean-pool a fixed-dimension embedding over a time span
    /// (see `EncoderFeatureSequence.pooledEmbedding`). Excluded from `Codable` (the raw
    /// encoder frames are large and not meant for serialization), so decoded results
    /// always carry `nil` here.
    public let encoderFeatures: EncoderFeatureSequence?

    private enum CodingKeys: String, CodingKey {
        case text, confidence, duration, processingTime, tokenTimings
        case performanceMetrics, ctcDetectedTerms, ctcAppliedTerms
        // `encoderFeatures` is intentionally omitted from coding.
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text = try container.decode(String.self, forKey: .text)
        confidence = try container.decode(Float.self, forKey: .confidence)
        duration = try container.decode(TimeInterval.self, forKey: .duration)
        processingTime = try container.decode(TimeInterval.self, forKey: .processingTime)
        tokenTimings = try container.decodeIfPresent([TokenTiming].self, forKey: .tokenTimings)
        performanceMetrics = try container.decodeIfPresent(
            ASRPerformanceMetrics.self, forKey: .performanceMetrics)
        ctcDetectedTerms = try container.decodeIfPresent([String].self, forKey: .ctcDetectedTerms)
        ctcAppliedTerms = try container.decodeIfPresent([String].self, forKey: .ctcAppliedTerms)
        encoderFeatures = nil
    }

    public init(
        text: String, confidence: Float, duration: TimeInterval, processingTime: TimeInterval,
        tokenTimings: [TokenTiming]? = nil,
        performanceMetrics: ASRPerformanceMetrics? = nil,
        ctcDetectedTerms: [String]? = nil,
        ctcAppliedTerms: [String]? = nil,
        encoderFeatures: EncoderFeatureSequence? = nil
    ) {
        self.text = text
        self.confidence = confidence
        self.duration = duration
        self.processingTime = processingTime
        self.tokenTimings = tokenTimings
        self.performanceMetrics = performanceMetrics
        self.ctcDetectedTerms = ctcDetectedTerms
        self.ctcAppliedTerms = ctcAppliedTerms
        self.encoderFeatures = encoderFeatures
    }

    /// Real-time factor (RTFx) - how many times faster than real-time
    public var rtfx: Float {
        Float(duration) / Float(processingTime)
    }

    /// Create a copy of this result with rescored text and CTC metadata from vocabulary boosting.
    ///
    /// - Parameters:
    ///   - text: The rescored transcript text
    ///   - detected: Vocabulary terms detected by CTC (candidates considered for replacement)
    ///   - applied: Vocabulary terms actually applied as replacements
    /// - Returns: A new ASRResult with updated text and CTC metadata
    public func withRescoring(text: String, detected: [String]?, applied: [String]?) -> ASRResult {
        ASRResult(
            text: text,
            confidence: confidence,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: tokenTimings,
            performanceMetrics: performanceMetrics,
            ctcDetectedTerms: detected,
            ctcAppliedTerms: applied,
            encoderFeatures: encoderFeatures
        )
    }
}

public struct TokenTiming: Codable, Sendable {
    public let token: String
    public let tokenId: Int
    public let startTime: TimeInterval
    public let endTime: TimeInterval
    public let confidence: Float

    public init(
        token: String, tokenId: Int, startTime: TimeInterval, endTime: TimeInterval,
        confidence: Float
    ) {
        self.token = token
        self.tokenId = tokenId
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
    }
}

/// Word-level timing, aggregated from a sequence of `TokenTiming`s by grouping
/// SentencePiece sub-word tokens on their word-boundary markers (`▁` / leading space).
public struct WordTiming: Codable, Sendable {
    public let word: String
    public let startTime: TimeInterval
    public let endTime: TimeInterval

    public init(word: String, startTime: TimeInterval, endTime: TimeInterval) {
        self.word = word
        self.startTime = startTime
        self.endTime = endTime
    }
}

/// Build word-level timings from token timings (e.g. from
/// `StreamingUnifiedAsrManager.consumeTokenTimings()`).
///
/// Tokens whose raw piece starts with a word-boundary marker (`▁` or a leading
/// space) begin a new word; the rest are appended to the current word. The
/// resulting word spans from the first sub-word token's `startTime` to the last
/// sub-word token's `endTime`.
public func buildWordTimings(from tokenTimings: [TokenTiming]) -> [WordTiming] {
    var wordTimings: [WordTiming] = []
    var currentWord = ""
    var wordStart: TimeInterval = 0
    var wordEnd: TimeInterval = 0

    func flush() {
        let trimmed = currentWord.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        wordTimings.append(WordTiming(word: trimmed, startTime: wordStart, endTime: wordEnd))
    }

    for timing in tokenTimings {
        let token = timing.token
        if token.isEmpty || token == "<blank>" || token == "<pad>" {
            continue
        }

        let startsNewWord = isWordBoundary(token) || currentWord.isEmpty
        if startsNewWord && !currentWord.isEmpty {
            flush()
            currentWord = ""
        }

        if startsNewWord {
            currentWord = stripWordBoundaryPrefix(token)
            wordStart = timing.startTime
        } else {
            currentWord += token
        }
        wordEnd = timing.endTime
    }

    flush()
    return wordTimings
}

// MARK: - Errors

public enum ASRError: Error, LocalizedError {
    case notInitialized
    case invalidAudioData
    case modelLoadFailed
    case processingFailed(String)
    case modelCompilationFailed
    case unsupportedPlatform(String)
    case streamingConversionFailed(Error)
    case fileAccessFailed(URL, Error)
    case encoderInstantiationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "AsrManager not initialized. Call initialize() first."
        case .invalidAudioData:
            return "Invalid audio data provided. Must be at least 300ms of 16kHz audio."
        case .modelLoadFailed:
            return "Failed to load Parakeet CoreML models."
        case .processingFailed(let message):
            return "ASR processing failed: \(message)"
        case .modelCompilationFailed:
            return "CoreML model compilation failed after recovery attempts."
        case .unsupportedPlatform(let message):
            return message
        case .streamingConversionFailed(let error):
            return "Streaming audio conversion failed: \(error.localizedDescription)"
        case .fileAccessFailed(let url, let error):
            return "Failed to access audio file at \(url.path): \(error.localizedDescription)"
        case .encoderInstantiationFailed(let message):
            return "Encoder ANE program failed to instantiate: \(message)"
        }
    }
}
