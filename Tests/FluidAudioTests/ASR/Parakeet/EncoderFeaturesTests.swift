import CoreML
import XCTest

@testable import FluidAudio

/// Unit tests for opt-in encoder-feature export (`EncoderFeatureSequence`).
///
/// These exercise the pure logic — runtime axis detection in the frame builder and
/// the time-span → frame-range pooling — without requiring model inference or audio
/// fixtures. Runtime verification against a real Parakeet encoder is covered by the
/// integration-style check in the task report (deferred when models aren't present).
final class EncoderFeaturesTests: XCTestCase {

    private let hiddenSize = 4  // small stand-in for ASRConstants.encoderHiddenSize

    /// Build an MLMultiArray laid out as [1, T, H] (time-major) with frame `t`
    /// holding value `Float(baseValue + t)` in every hidden slot.
    private func makeTimeMajorArray(frames: Int, baseValue: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: frames), NSNumber(value: hiddenSize)], dataType: .float32)
        for t in 0..<frames {
            for h in 0..<hiddenSize {
                array[t * hiddenSize + h] = NSNumber(value: Float(baseValue + t))
            }
        }
        return array
    }

    /// Build an MLMultiArray laid out as [1, H, T] (hidden-major) with frame `t`
    /// holding value `Float(baseValue + t)` in every hidden slot.
    private func makeHiddenMajorArray(frames: Int, baseValue: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: hiddenSize), NSNumber(value: frames)], dataType: .float32)
        for h in 0..<hiddenSize {
            for t in 0..<frames {
                array[h * frames + t] = NSNumber(value: Float(baseValue + t))
            }
        }
        return array
    }

    // MARK: - Runtime axis detection

    func testFrameBuilderTimeMajorLayout() throws {
        let array = try makeTimeMajorArray(frames: 3, baseValue: 10)
        let frames = try EncoderFeatureSequence.frames(
            from: array, validFrameCount: 3, hiddenSize: hiddenSize)

        XCTAssertEqual(frames.count, 3)
        XCTAssertEqual(frames[0], [10, 10, 10, 10])
        XCTAssertEqual(frames[1], [11, 11, 11, 11])
        XCTAssertEqual(frames[2], [12, 12, 12, 12])
    }

    func testFrameBuilderHiddenMajorLayout() throws {
        // Same logical frames, transposed physical layout — must produce identical result.
        let array = try makeHiddenMajorArray(frames: 3, baseValue: 10)
        let frames = try EncoderFeatureSequence.frames(
            from: array, validFrameCount: 3, hiddenSize: hiddenSize)

        XCTAssertEqual(frames.count, 3)
        XCTAssertEqual(frames[0], [10, 10, 10, 10])
        XCTAssertEqual(frames[1], [11, 11, 11, 11])
        XCTAssertEqual(frames[2], [12, 12, 12, 12])
    }

    func testFrameBuilderHonorsValidFrameCount() throws {
        let array = try makeTimeMajorArray(frames: 5, baseValue: 0)
        // Only the first 2 frames are valid.
        let frames = try EncoderFeatureSequence.frames(
            from: array, validFrameCount: 2, hiddenSize: hiddenSize)
        XCTAssertEqual(frames.count, 2)
    }

    // MARK: - Pooling

    private func makeSequence(windows: [(frames: [[Float]], offset: Int)]) -> EncoderFeatureSequence {
        EncoderFeatureSequence(
            windows: windows.map {
                EncoderFeatureSequence.Window(frames: $0.frames, globalFrameOffset: $0.offset)
            },
            hiddenSize: hiddenSize,
            secondsPerFrame: 0.08
        )
    }

    func testPooledEmbeddingMeanOverRange() {
        // Single window, frames 0..2 with values 0,1,2 in each slot.
        let seq = makeSequence(windows: [
            (frames: [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], offset: 0)
        ])
        // 0.08s/frame → [0, 0.24) maps to frame range [0, 3) → mean of 0,1,2 = 1.
        let pooled = seq.pooledEmbedding(startSeconds: 0.0, endSeconds: 0.24)
        XCTAssertEqual(pooled, [1, 1, 1, 1])
    }

    func testPooledEmbeddingSubRange() {
        let seq = makeSequence(windows: [
            (frames: [[0, 0, 0, 0], [10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]], offset: 0)
        ])
        // [0.08, 0.24) → floor(1)=1 .. ceil(3)=3 → frames 1,2 → mean of 10,20 = 15.
        let pooled = seq.pooledEmbedding(startSeconds: 0.08, endSeconds: 0.24)
        XCTAssertEqual(pooled, [15, 15, 15, 15])
    }

    func testPooledEmbeddingPicksMostCoveringOverlappingWindow() {
        // Two overlapping windows. Requested span sits mostly in the second window.
        let seq = makeSequence(windows: [
            (frames: [[1, 1, 1, 1], [1, 1, 1, 1]], offset: 0),  // global frames 0..1
            (frames: [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]], offset: 1),  // global frames 1..3
        ])
        // [0.08, 0.24) → frames [1, 3): window 0 covers {1} (1 frame), window 1 covers {1,2} (2 frames).
        // Window 1 wins → mean of its frames in local range = 9.
        let pooled = seq.pooledEmbedding(startSeconds: 0.08, endSeconds: 0.24)
        XCTAssertEqual(pooled, [9, 9, 9, 9])
    }

    func testPooledEmbeddingReturnsNilWhenUncovered() {
        let seq = makeSequence(windows: [
            (frames: [[1, 1, 1, 1]], offset: 0)  // only global frame 0
        ])
        // Request far beyond any captured frame.
        XCTAssertNil(seq.pooledEmbedding(startSeconds: 5.0, endSeconds: 6.0))
    }

    func testPooledEmbeddingEmptySequenceReturnsNil() {
        let seq = makeSequence(windows: [])
        XCTAssertNil(seq.pooledEmbedding(startSeconds: 0.0, endSeconds: 1.0))
    }

    func testPooledEmbeddingZeroWidthSpanCoversAtLeastOneFrame() {
        let seq = makeSequence(windows: [
            (frames: [[7, 7, 7, 7], [8, 8, 8, 8]], offset: 0)
        ])
        // start == end at 0.0 → range clamped to [0, 1) → frame 0.
        let pooled = seq.pooledEmbedding(startSeconds: 0.0, endSeconds: 0.0)
        XCTAssertEqual(pooled, [7, 7, 7, 7])
    }
}
