import assert from 'node:assert/strict'
import test from 'node:test'
import { nearestFinalFrame, reviewSeekTimestamp, trajectorySegments } from '../src/components/trackingFinalDebug.ts'

const point = (frame, time, segment = 'segment-a') => ({ identity_id: 'identity-001', segment_id: segment, frame_index: frame, timestamp_s: time, measurement_x: 1, measurement_z: 2, smoothed_x: 1.1, smoothed_z: 2.1, origin: 'observed', speed_mps: 1, acceleration_mps2: 0, position_adjustment_m: 0, position_std_m: .1, kinematics_quality: 'rts_cv' })

test('selects the nearest published V1 frame for the video timestamp', () => {
  const frame = nearestFinalFrame([point(0, 0), point(3, .1), point(6, .2)], .14)
  assert.equal(frame?.frameIndex, 3)
  assert.equal(frame?.timestampS, .1)
})

test('never connects a selected trajectory across a published segment break', () => {
  const segments = trajectorySegments([point(0, 0, 'segment-a'), point(3, .1, 'segment-a'), point(30, 1, 'segment-b'), point(33, 1.1, 'segment-b')], [{ identity_id: 'identity-001', left_frame_index: 3, right_frame_index: 30 }])
  assert.deepEqual(segments.map((segment) => segment.map((item) => item.frame_index)), [[0, 3], [30, 33]])
})

test('review seek uses its published timestamp and does not invent one from a frame', () => {
  assert.equal(reviewSeekTimestamp({ identity_id: 'identity-001', frame_index: 30, timestamp_s: 1.001 }), 1.001)
  assert.equal(reviewSeekTimestamp({ identity_id: 'identity-001', start_frame_index: 30, start_timestamp_s: 1.001 }), 1.001)
  assert.equal(reviewSeekTimestamp({ identity_id: 'identity-001', frame_index: 30 }), null)
})
