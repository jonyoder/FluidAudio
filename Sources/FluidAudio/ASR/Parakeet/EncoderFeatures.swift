import CoreML
import Foundation

/// Opt-in acoustic feature export from the Parakeet TDT encoder.
///
/// The encoder produces a `[1, T, hiddenSize]` (or `[1, hiddenSize, T]`, depending on
/// the CoreML export's axis layout) feature tensor per sliding window that the TDT
/// decoder normally consumes and discards. When `AsrManager.captureEncoderFeatures`
/// is enabled, each window's valid encoder frames are copied out (frame-major,
/// `frames[t]` is a `hiddenSize`-vector) and collected here, tagged with the window's
/// `globalFrameOffset` so spans can be located across overlapping windows.
///
/// Consumers can then mean-pool a fixed-dimension embedding over an arbitrary time
/// span (e.g. for voice enrollment / acoustic disambiguation) via `pooledEmbedding`.
///
/// Frames are stored as `[Float]` copies (not `MLMultiArray`, which isn't `Sendable`),
/// so the whole structure is `Sendable` and safe to hand across the actor boundary.
public struct EncoderFeatureSequence: Sendable {

    /// One sliding window's worth of encoder frames.
    public struct Window: Sendable {
        /// Valid encoder frames, frame-major: `frames[t]` is a `hiddenSize`-vector.
        public let frames: [[Float]]
        /// Global encoder-frame index of this window's first frame.
        public let globalFrameOffset: Int

        public init(frames: [[Float]], globalFrameOffset: Int) {
            self.frames = frames
            self.globalFrameOffset = globalFrameOffset
        }
    }

    /// The captured windows, in transcription order.
    public let windows: [Window]
    /// Encoder hidden dimension (e.g. 1024 for the 0.6B model).
    public let hiddenSize: Int
    /// Seconds of audio per encoder frame (the encoder frame stride).
    public let secondsPerFrame: Double

    public init(windows: [Window], hiddenSize: Int, secondsPerFrame: Double) {
        self.windows = windows
        self.hiddenSize = hiddenSize
        self.secondsPerFrame = secondsPerFrame
    }

    /// Build one window's frame vectors from a raw encoder `MLMultiArray`.
    ///
    /// Uses runtime axis detection (`EncoderFrameView` matches the non-batch axis whose
    /// dimension equals `hiddenSize` as the hidden axis and the other as the time axis)
    /// and `.strides`-correct copying, so it is correct regardless of whether the export
    /// reports `[1, T, hiddenSize]` or `[1, hiddenSize, T]`.
    ///
    /// - Parameters:
    ///   - encoderOutput: The raw encoder feature tensor for this window.
    ///   - validFrameCount: Number of valid (non-padding) frames to copy.
    ///   - hiddenSize: Expected encoder hidden dimension.
    /// - Returns: Frame-major `[[Float]]` (length `min(validFrameCount, availableFrames)`),
    ///   each inner array of length `hiddenSize`.
    static func frames(
        from encoderOutput: MLMultiArray,
        validFrameCount: Int,
        hiddenSize: Int
    ) throws -> [[Float]] {
        let view = try EncoderFrameView(
            encoderOutput: encoderOutput,
            validLength: validFrameCount,
            expectedHiddenSize: hiddenSize
        )

        var frames = [[Float]]()
        frames.reserveCapacity(view.count)
        for index in 0..<view.count {
            var frame = [Float](repeating: 0, count: view.hiddenSize)
            try frame.withUnsafeMutableBufferPointer { buffer in
                try view.copyFrame(at: index, into: buffer.baseAddress!, destinationStride: 1)
            }
            frames.append(frame)
        }
        return frames
    }

    /// Mean-pool a `hiddenSize`-dimension embedding over a time span.
    ///
    /// The requested `[startSeconds, endSeconds)` span is converted to a half-open
    /// global encoder-frame range `[floor(start / secondsPerFrame), ceil(end / secondsPerFrame))`
    /// (clamped to at least one frame). The window covering the most of that range is
    /// selected (handling overlapping windows), and its frame vectors over the covered
    /// portion are averaged element-wise.
    ///
    /// - Returns: The pooled `hiddenSize`-vector, or `nil` if no captured window covers
    ///   any of the requested span.
    public func pooledEmbedding(startSeconds: Double, endSeconds: Double) -> [Float]? {
        guard secondsPerFrame > 0, !windows.isEmpty else { return nil }

        // Convert seconds → half-open global frame range, at least one frame wide.
        let lo = max(0, Int((startSeconds / secondsPerFrame).rounded(.down)))
        var hi = Int((endSeconds / secondsPerFrame).rounded(.up))
        if hi <= lo { hi = lo + 1 }

        // Find the window covering the most of [lo, hi).
        var bestWindowIndex: Int?
        var bestOverlap = 0
        var bestLocalRange: Range<Int> = 0..<0
        for (windowIndex, window) in windows.enumerated() {
            let windowStart = window.globalFrameOffset
            let windowEnd = window.globalFrameOffset + window.frames.count
            let overlapStart = max(lo, windowStart)
            let overlapEnd = min(hi, windowEnd)
            let overlap = overlapEnd - overlapStart
            if overlap > bestOverlap {
                bestOverlap = overlap
                bestWindowIndex = windowIndex
                bestLocalRange = (overlapStart - windowStart)..<(overlapEnd - windowStart)
            }
        }

        guard let windowIndex = bestWindowIndex, bestOverlap > 0 else { return nil }
        let frames = windows[windowIndex].frames
        let localRange = bestLocalRange
        guard !localRange.isEmpty else { return nil }

        var accumulator = [Float](repeating: 0, count: hiddenSize)
        var counted = 0
        for frameIndex in localRange {
            let frame = frames[frameIndex]
            guard frame.count == hiddenSize else { continue }
            for h in 0..<hiddenSize {
                accumulator[h] += frame[h]
            }
            counted += 1
        }
        guard counted > 0 else { return nil }

        let inverse = 1.0 / Float(counted)
        for h in 0..<hiddenSize {
            accumulator[h] *= inverse
        }
        return accumulator
    }
}
