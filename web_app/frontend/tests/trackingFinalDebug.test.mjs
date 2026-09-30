import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { nearestFinalFrame, pointCoordinates, reviewSeekTimestamp, trajectoryRenderSegments, trajectorySegments } from '../src/components/trackingFinalDebug.ts'

const point = (frame, time, segment = 'segment-a') => ({ identity_id: 'identity-001', segment_id: segment, frame_index: frame, timestamp_s: time, measurement_x: 1, measurement_z: 2, smoothed_x: 1.1, smoothed_z: 2.1, origin: 'observed', speed_mps: 1, acceleration_mps2: 0, position_adjustment_m: 0, position_std_m: .1, kinematics_quality: 'rts_cv' })
const actorPoint = (actor, frame, time, segment = `${actor}:segment-001`) => ({ actor_id: actor, segment_id: segment, frame_index: frame, timestamp_s: time, x: 1, z: 2, measurement_x: 1.1, measurement_z: 2.1, origin: 'observed' })

test('selects the nearest published V1 frame for the video timestamp', () => {
  const frame = nearestFinalFrame([point(0, 0), point(3, .1), point(6, .2)], .14)
  assert.equal(frame?.frameIndex, 3)
  assert.equal(frame?.timestampS, .1)
})

test('never connects a selected trajectory across a published segment break', () => {
  const segments = trajectorySegments([point(0, 0, 'segment-a'), point(3, .1, 'segment-a'), point(30, 1, 'segment-b'), point(33, 1.1, 'segment-b')], [{ identity_id: 'identity-001', left_frame_index: 3, right_frame_index: 30 }])
  assert.deepEqual(segments.map((segment) => segment.map((item) => item.frame_index)), [[0, 3], [30, 33]])
})

test('Persistent V1 uses published actor points without creating a missing actor point', () => {
  const frame = nearestFinalFrame([actorPoint('actor-001', 0, 0), actorPoint('actor-023', 0, 0)], 0)
  assert.deepEqual(frame?.points.map((item) => item.actor_id), ['actor-001', 'actor-023'])
  assert.equal(frame?.points.some((item) => item.actor_id === 'actor-022'), false)
})

test('Persistent trails keep actor ids and split on published segments', () => {
  const segments = trajectorySegments([actorPoint('actor-001', 0, 0, 'one'), actorPoint('actor-001', 3, .1, 'one'), actorPoint('actor-001', 9, .3, 'two')], [])
  assert.deepEqual(segments.map((segment) => segment.map((item) => item.frame_index)), [[0, 3], [9]])
  assert.equal(segments[0][0].actor_id, 'actor-001')
})

test('Persistent raw omits synthetic null measurements without emitting invalid coordinates', () => {
  const observed = actorPoint('actor-001', 0, 0)
  const interpolated = { ...actorPoint('actor-001', 3, .1), origin: 'interpolated', measurement_x: null, measurement_z: null }
  const predicted = { ...actorPoint('actor-001', 6, .2), origin: 'predicted', measurement_x: null, measurement_z: null }

  assert.deepEqual(pointCoordinates(observed, 'measurement'), { x: 1.1, z: 2.1 })
  assert.equal(pointCoordinates(interpolated, 'measurement'), null)
  assert.equal(pointCoordinates(predicted, 'measurement'), null)
  assert.deepEqual(pointCoordinates(interpolated, 'smoothed'), { x: 1, z: 2 })

  const rawSegments = trajectoryRenderSegments([observed, interpolated, predicted], [], 'measurement')
  assert.deepEqual(rawSegments.map((segment) => segment.map(({ point }) => point.frame_index)), [[0]])
  assert.equal(rawSegments.flat().every(({ coordinates }) => Number.isFinite(coordinates.x) && Number.isFinite(coordinates.z)), true)
})

test('review seek uses its published timestamp and does not invent one from a frame', () => {
  assert.equal(reviewSeekTimestamp({ identity_id: 'identity-001', frame_index: 30, timestamp_s: 1.001 }), 1.001)
  assert.equal(reviewSeekTimestamp({ identity_id: 'identity-001', start_frame_index: 30, start_timestamp_s: 1.001 }), 1.001)
  assert.equal(reviewSeekTimestamp({ identity_id: 'identity-001', frame_index: 30 }), null)
})

test('Tracking Debug connects its presentation clock after FINAL mounts the video', () => {
  const source = readFileSync(new URL('../src/components/TrackingDebugWorkspace.tsx', import.meta.url), 'utf8')
  assert.equal(source.includes('if (!artifact) return'), true)
  assert.equal(source.includes('}, [artifact, masterVideo, source?.id, clip.ultra_base_url, context])'), true)
  const clockEffect = source.slice(source.indexOf('useEffect(() => {\n    if (!artifact) return'), source.indexOf('const seek'))
  assert.equal(clockEffect.includes('selectedIdentityId'), false)
  assert.equal(clockEffect.includes('followPlayer'), false)
})

test('Tracking Debug keeps Physical V1 as default and exposes a separate Persistent selector', () => {
  const source = readFileSync(new URL('../src/components/TrackingDebugWorkspace.tsx', import.meta.url), 'utf8')
  assert.equal(source.includes("useState<TrackingSource>('physical')"), true)
  assert.equal(source.includes('Physical V1'), true)
  assert.equal(source.includes('Persistent V1'), true)
  assert.equal(source.includes('api.trackingPersistent(matchId, clip.id)'), true)
  assert.equal(source.includes('Show point origin'), true)
})

test('changing tracking source leaves the presentation clock state untouched', () => {
  const source = readFileSync(new URL('../src/components/TrackingDebugWorkspace.tsx', import.meta.url), 'utf8')
  const changeSource = source.slice(source.indexOf('const changeTrackingSource'), source.indexOf('const persistentCoverage'))
  assert.equal(changeSource.includes('setTrackingSource(next)'), true)
  assert.equal(changeSource.includes('setTime('), false)
  assert.equal(changeSource.includes('setPlaying('), false)
  assert.equal(changeSource.includes('setView('), false)
})
