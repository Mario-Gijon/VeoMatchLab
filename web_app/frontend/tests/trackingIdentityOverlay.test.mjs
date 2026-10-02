import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { activeIdentityObservations, identityColor } from '../src/components/trackingIdentityOverlayData.ts'

const observation = (time, bbox, lens = 'left') => ({ timestamp_s: time, bbox, source_lens: lens })
const track = (id, observations, identity = 'person-0047') => ({ tracklet_id: id, identity_id: identity, observations })

test('interpolates only within one tracklet and lens over a short gap', () => {
  const active = activeIdentityObservations([track('trk-a', [observation(1, [10, 20, 30, 40]), observation(1.2, [20, 30, 50, 60])])], 1.1)
  assert.deepEqual(active[0].bbox.map((value) => Number(value.toFixed(6))), [15, 25, 40, 50])
  assert.equal(active[0].interpolated, true)
  assert.equal(active[0].trackletId, 'trk-a')
})

test('does not interpolate across long gaps, lenses, or tracklets sharing an identity', () => {
  assert.equal(activeIdentityObservations([track('long', [observation(0, [0, 0, 1, 1]), observation(.36, [2, 2, 3, 3])])], .18).length, 0)
  assert.equal(activeIdentityObservations([track('lenses', [observation(0, [0, 0, 1, 1], 'left'), observation(.2, [2, 2, 3, 3], 'right')])], .1).length, 0)
  assert.equal(activeIdentityObservations([track('first', [observation(0, [0, 0, 1, 1])]), track('second', [observation(.2, [2, 2, 3, 3])])], .1).length, 0)
})

test('identity colors are deterministic and native bboxes are unmodified', () => {
  assert.equal(identityColor('person-0047'), identityColor('person-0047'))
  assert.notEqual(identityColor('person-0047'), identityColor('person-0048'))
  const active = activeIdentityObservations([track('native', [observation(2, [100, 1100, 300, 1300], 'right')])], 2)
  assert.deepEqual(active[0].bbox, [100, 1100, 300, 1300])
})

test('the source view keeps playback usable when Tracking V1 is unavailable', () => {
  const source = readFileSync(new URL('../src/components/ClipWorkspace.tsx', import.meta.url), 'utf8')
  assert.equal(source.includes("value=\"source\""), true)
  assert.equal(source.includes('Tracking V1 no disponible'), true)
  assert.equal(source.includes('source ? <MatchViewer'), true)
})
