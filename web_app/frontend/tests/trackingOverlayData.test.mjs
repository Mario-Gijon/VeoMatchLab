import assert from 'node:assert/strict'
import test from 'node:test'
import { representativeObservation } from '../src/components/trackingOverlayData.ts'

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
