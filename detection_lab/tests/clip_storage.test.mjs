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
