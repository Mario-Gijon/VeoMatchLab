import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'

const storage = new Map()
const context = {
  document: {
    body: { dataset: { matchId: 'match_for_test' } },
    querySelector: () => ({ append() {}, replaceChildren() {} }),
  },
  fetch: () => new Promise(() => {}),
  localStorage: {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
  },
  URLSearchParams,
  console,
}
context.window = context
vm.runInNewContext(readFileSync(new URL('../static/app.js', import.meta.url), 'utf8'), context)

const clip = { start_s: 750, duration_s: 30 }
assert.equal(context.DetectionLabClipStore.saveClip('match_for_test', clip), true)
assert.equal(
  JSON.stringify(context.DetectionLabClipStore.loadClip('match_for_test')),
  JSON.stringify({ match_id: 'match_for_test', ...clip }),
)

const sample = { detections: [
  { lens: 'left', bbox: [1, 2, 3, 4], confidence: 0.1 },
  { lens: 'right', bbox: [5, 6, 7, 8], confidence: 0.3 },
] }
assert.equal(context.DetectionLabPersonView.visibleDetections(sample, 0.25).length, 1)
assert.equal(
  JSON.stringify(context.DetectionLabPersonView.stackedBox(sample.detections[1])),
  JSON.stringify([5, 1030, 7, 1032]),
)
assert.equal(context.DetectionLabPersonView.personPageState(null), 'no-clip')
assert.equal(context.DetectionLabPersonView.personPageState(clip), 'not-run')
assert.equal(context.DetectionLabPersonView.personPageState(clip, { state: 'processing' }), 'processing')
assert.equal(context.DetectionLabPersonView.personPageState(clip, { state: 'cached' }), 'cached')
assert.equal(context.DetectionLabPersonView.personPageState(clip, { state: 'complete' }), 'complete')
assert.equal(context.DetectionLabPersonView.personPageState(clip, { state: 'error' }), 'error')
