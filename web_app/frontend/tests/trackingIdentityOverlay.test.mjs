import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { activeFixedRosterObservations, activeIdentityObservations, identityColor, identityOverlayStyle } from '../src/components/trackingIdentityOverlayData.ts'

const observation = (time, bbox, lens = 'left', frame = Math.round(time * 30), x = time * 10, z = -time * 10) => ({ frame_index: frame, timestamp_s: time, x, z, bbox, source_lens: lens, representative_view: null, virtual_bbox: null })
const track = (id, observations, identity = 'actor-0007', assignmentStatus = 'assigned') => ({ tracklet_id: id, identity_id: identity, assignment_status: assignmentStatus, observations })

test('interpolates only within one tracklet and lens over a short gap', () => {
  const active = activeIdentityObservations([track('trk-a', [observation(1, [10, 20, 30, 40]), observation(1.2, [20, 30, 50, 60])])], 1.1)
  assert.deepEqual(active[0].bbox.map((value) => Number(value.toFixed(6))), [15, 25, 40, 50])
  assert.equal(Number(active[0].x.toFixed(6)), 11)
  assert.equal(Number(active[0].z.toFixed(6)), -11)
  assert.equal(active[0].interpolated, true)
  assert.equal(active[0].trackletId, 'trk-a')
})

test('does not interpolate across long gaps or tracklets sharing an identity', () => {
  assert.equal(activeIdentityObservations([track('long', [observation(0, [0, 0, 1, 1]), observation(.36, [2, 2, 3, 3])])], .18).length, 0)
  const acrossLenses = activeIdentityObservations([track('lenses', [observation(0, [0, 0, 1, 1], 'left'), observation(.2, [2, 2, 3, 3], 'right')])], .1)
  assert.equal(acrossLenses[0].bbox, null)
  assert.equal(acrossLenses[0].interpolated, true)
  assert.equal(activeIdentityObservations([track('first', [observation(0, [0, 0, 1, 1])]), track('second', [observation(.2, [2, 2, 3, 3])])], .1).length, 0)
})

test('identity colors are deterministic and native bboxes are unmodified', () => {
  assert.equal(identityColor('actor-0007'), identityColor('actor-0007'))
  assert.notEqual(identityColor('actor-0007'), identityColor('actor-0008'))
  assert.equal(identityOverlayStyle('UNASSIGNED', 'unassigned').color, '#d7dce2')
  assert.equal(identityOverlayStyle('UNASSIGNED', 'unassigned').dash, '7 5')
  const active = activeIdentityObservations([track('native', [observation(2, [100, 1100, 300, 1300], 'right')])], 2)
  assert.deepEqual(active[0].bbox, [100, 1100, 300, 1300])
})

test('uses exact physical frames, keeps bbox-null spatial observations, and never carries forward', () => {
  const spatial = track('spatial', [observation(1, null, null, 30, 8, -4), observation(1.2, [1, 2, 3, 4], 'left', 36, 10, -6)])
  assert.deepEqual(activeIdentityObservations([spatial], 1.01, 30)[0].bbox, null)
  const between = activeIdentityObservations([spatial], 1.1)[0]
  assert.deepEqual([Number(between.x.toFixed(6)), Number(between.z.toFixed(6)), between.bbox], [9, -5, null])
  assert.equal(activeIdentityObservations([spatial], 1.7).length, 0)
})

test('fixed roster emits no more than one marker per assigned actor', () => {
  const actorA = track('trk-b', [observation(2, [1, 1, 2, 2], 'left', 60, 30, 10)], 'actor-0007')
  const actorB = track('trk-a', [observation(2, [3, 3, 4, 4], 'left', 60, -30, -10)], 'actor-0007')
  const pendingA = track('trk-u1', [observation(2, null)], 'UNASSIGNED', 'unassigned')
  const pendingB = track('trk-u2', [observation(2, null)], 'UNASSIGNED', 'unassigned')
  const active = activeFixedRosterObservations([actorA, actorB, pendingA, pendingB], 2)
  assert.deepEqual(active.map((item) => item.trackletId), ['trk-a', 'trk-u1', 'trk-u2'])
  assert.equal(active.filter((item) => item.identityId === 'actor-0007').length, 1)
})

test('the source view keeps playback usable when Tracking V1 is unavailable', () => {
  const source = readFileSync(new URL('../src/components/ClipWorkspace.tsx', import.meta.url), 'utf8')
  assert.equal(source.includes("value=\"source\""), true)
  assert.equal(source.includes('Tracking V1 no disponible'), true)
  assert.equal(source.includes('source ? <MatchViewer'), true)
  assert.equal(source.includes("useState<TrackingIdentityOverlaySource>('fixed_roster')"), true)
  assert.equal(source.includes('Fixed roster 7I.4'), true)
  assert.equal(source.includes('Legacy round state'), true)
  assert.equal(source.includes('api.trackingIdentityOverlay(matchId, clip.id, requestedOverlaySource)'), true)
  assert.equal(source.includes('FixedRosterPitch'), true)
  assert.equal(source.includes('FixedRosterTrackingOverlay'), true)
  const changeSource = source.slice(source.indexOf('onChange={(_, source: TrackingIdentityOverlaySource'), source.indexOf('}}><ToggleButton value="fixed_roster"'))
  assert.equal(changeSource.includes('setLocalTime('), false)
  assert.equal(changeSource.includes('setPlaying('), false)
})
