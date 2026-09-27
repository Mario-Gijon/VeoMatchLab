import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { presentationDirectionFromVirtualPixel } from '../src/components/trackingProjection.ts'
import { representativeObservation, visualAnchorsByTrack, visualDirectionForTrack } from '../src/components/trackingOverlayData.ts'

const association = [{ frame_index: 12, observations: [
  { observation_id: 7, view_name: 'tight_far_c1', xyxy: [100, 200, 120, 260] },
], measurements: [{ measurement_id: 3, primary_observation_id: 7, observation_ids: [7] }] }]

test('observed track resolves its frame measurement and primary observation', () => {
  const result = representativeObservation({ position_kind: 'observed', measurement_id: '12:3' }, association)
  assert.deepEqual(result, { view_name: 'tight_far_c1', xyxy: [100, 200, 120, 260] })
})

test('predicted tracks do not resolve a video representative', () => {
  assert.equal(representativeObservation({ position_kind: 'predicted', measurement_id: '12:3' }, association), null)
})

const specs = {
  left: { longitude: 0, latitude: 0, vfov: Math.PI / 2 },
  right: { longitude: .3, latitude: 0, vfov: Math.PI / 2 },
}
const visualAssociation = [
  { frame_index: 100, observations: [{ observation_id: 1, view_name: 'left', xyxy: [600, 400, 680, 600] }], measurements: [{ measurement_id: 1, primary_observation_id: 1, observation_ids: [1] }] },
  { frame_index: 102, observations: [{ observation_id: 2, view_name: 'right', xyxy: [600, 400, 680, 600] }], measurements: [{ measurement_id: 2, primary_observation_id: 2, observation_ids: [2] }] },
]
const visualFrames = [
  { frame_index: 100, timestamp_s: 100 / 30, tracks: [{ track_id: 7, position_kind: 'observed', measurement_id: '100:1' }] },
  { frame_index: 101, timestamp_s: 101 / 30, tracks: [{ track_id: 7, position_kind: 'interpolated', measurement_id: null }] },
  { frame_index: 102, timestamp_s: 102 / 30, tracks: [{ track_id: 7, position_kind: 'observed', measurement_id: '102:2' }] },
]

test('observed final samples retain their exact representative visual ray', () => {
  const anchors = visualAnchorsByTrack(visualFrames, visualAssociation, specs)
  const expected = presentationDirectionFromVirtualPixel(specs.left, 640, 600, 1280, 720)
  assert.deepEqual(visualDirectionForTrack(visualFrames[0].tracks[0], 100, 100 / 30, anchors), expected)
})

test('interpolated final samples receive an nlerp visual ray between observed anchors', () => {
  const anchors = visualAnchorsByTrack(visualFrames, visualAssociation, specs)
  const direction = visualDirectionForTrack(visualFrames[1].tracks[0], 101, 101 / 30, anchors)
  assert.ok(direction)
  assert.ok(Math.abs(Math.hypot(...direction) - 1) < 1e-12)
  const [left, right] = anchors.get(7)
  assert.ok(direction[0] > Math.min(left.direction[0], right.direction[0]))
  assert.ok(direction[0] < Math.max(left.direction[0], right.direction[0]))
})

test('anchors can originate from different virtual cameras', () => {
  const anchors = visualAnchorsByTrack(visualFrames, visualAssociation, specs)
  assert.equal(anchors.get(7)?.length, 2)
  assert.notDeepEqual(anchors.get(7)?.[0].direction, anchors.get(7)?.[1].direction)
})

test('predicted samples bridge only short observed-anchor gaps', () => {
  const anchors = visualAnchorsByTrack(visualFrames, visualAssociation, specs)
  const predicted = { track_id: 7, position_kind: 'predicted', measurement_id: null }
  assert.ok(visualDirectionForTrack(predicted, 101, 101 / 30, anchors))
  assert.equal(visualDirectionForTrack(predicted, 101, 101 / 30, anchors, .01), null)
})

test('visual interpolation never extrapolates beyond the observed anchors', () => {
  const anchors = visualAnchorsByTrack(visualFrames, visualAssociation, specs)
  const interpolated = { track_id: 7, position_kind: 'interpolated', measurement_id: null }
  assert.equal(visualDirectionForTrack(interpolated, 99, 99 / 30, anchors), null)
  assert.equal(visualDirectionForTrack(interpolated, 103, 103 / 30, anchors), null)
})

test('overlay visual rays do not use Tracking V2 x/z coordinates', () => {
  const source = readFileSync(new URL('../src/components/trackingOverlayData.ts', import.meta.url), 'utf8')
  assert.equal(source.includes('track.x'), false)
  assert.equal(source.includes('track.z'), false)
})
