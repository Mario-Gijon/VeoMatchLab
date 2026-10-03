import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { activeActorTrajectoryPoints, presentationDirectionForActorPoint, sourceBboxForActorPoint } from '../src/components/actorTrajectoryData.ts'
import { identityColor } from '../src/components/trackingIdentityOverlayData.ts'

const endpoint = (view, lens = 'left', source = [10, 20, 30, 40]) => ({ detection_id: 'det', frame_index: 0, timestamp_s: 0, representative_view: view, virtual_bbox_xyxy: [600, 300, 640, 400], source_lens: lens, source_bbox_xyxy: source })
const observed = (frame, time) => ({ frame_index: frame, timestamp_s: time, x: 3.5, z: -4.5, origin: 'observed', segment_id: 'a:1', segment_index: 1, source_tracklet_ids: ['trk-a'], visual_anchor: endpoint('cam-a'), visual_interpolation: null })
const interpolated = (frame, time, left = endpoint('cam-a'), right = endpoint('cam-b')) => ({ frame_index: frame, timestamp_s: time, x: 7, z: 8, origin: 'interpolated', segment_id: 'a:1', segment_index: 1, source_tracklet_ids: ['trk-a'], visual_anchor: null, visual_interpolation: { left, right, alpha: .5, visual_interpolation_allowed: true, source_bbox_interpolation_allowed: true } })
const actor = (points) => ({ actor_id: 'actor-0007', tracklet_ids: ['trk-a'], segment_count: 1, points })
const specs = { 'cam-a': { longitude: 0, latitude: 0, vfov: 1 }, 'cam-b': { longitude: .2, latitude: 0, vfov: 1 } }

test('selects exact frame points, uses smoothed x/z, and does not carry through a preserved gap', () => {
  const points = activeActorTrajectoryPoints([actor([observed(0, 0), observed(10, 10 / 30)])], 0, 30)
  assert.deepEqual([points[0].x, points[0].z, points[0].actorId], [3.5, -4.5, 'actor-0007'])
  assert.equal(activeActorTrajectoryPoints([actor([observed(0, 0), observed(10, 10 / 30)])], 5 / 30, 30).length, 0)
})

test('creates observed and cross-camera interpolated presentation rays without virtual-bbox interpolation', () => {
  assert.ok(presentationDirectionForActorPoint(observed(0, 0), specs))
  const direction = presentationDirectionForActorPoint(interpolated(1, 1 / 30), specs)
  assert.ok(direction)
  assert.ok(Math.abs(Math.hypot(...direction) - 1) < 1e-10)
  assert.equal(presentationDirectionForActorPoint({ ...interpolated(1, 1 / 30), visual_interpolation: { ...interpolated(1, 1 / 30).visual_interpolation, visual_interpolation_allowed: false } }, specs), null)
  assert.equal(presentationDirectionForActorPoint({ ...observed(0, 0), visual_anchor: { ...endpoint('cam-a'), virtual_bbox_xyxy: [1, 2, Infinity, 4] } }, specs), null)
})

test('uses physical source bboxes directly and interpolates them only on the same lens', () => {
  assert.deepEqual(sourceBboxForActorPoint(observed(0, 0)), [10, 20, 30, 40])
  const sameLens = interpolated(1, 1 / 30, endpoint('cam-a', 'left', [0, 0, 10, 10]), endpoint('cam-a', 'left', [10, 20, 30, 40]))
  assert.deepEqual(sourceBboxForActorPoint(sameLens), [5, 10, 20, 25])
  assert.equal(sourceBboxForActorPoint(interpolated(1, 1 / 30, endpoint('cam-a', 'left'), endpoint('cam-b', 'right'))), null)
  assert.equal(identityColor('actor-0007'), identityColor('actor-0007'))
})

test('workspace exposes 7I.7 as a read-only actor mode without a second decoder', () => {
  const source = readFileSync(new URL('../src/components/ClipWorkspace.tsx', import.meta.url), 'utf8')
  assert.equal(source.includes('Actor trajectories 7I.7'), true)
  assert.equal(source.includes('api.actorTrajectories(matchId, clip.id)'), true)
  assert.equal(source.includes('ActorTrajectoryOverlay'), true)
  assert.equal(source.includes('ActorTrajectorySourceOverlay'), true)
  assert.equal(source.includes('actorTrajectorySelected} variant="outlined"'), true)
  assert.equal(source.includes('startVideoPresentationClock'), true)
})
