import assert from 'node:assert/strict'
import test from 'node:test'
import { startVideoPresentationClock } from '../src/components/videoPresentationClock.ts'

test('requestVideoFrameCallback publishes authoritative mediaTime for every presented frame', () => {
  let callback = null
  let cancelled = null
  const video = {
    currentTime: 99,
    requestVideoFrameCallback(next) { callback = next; return 17 },
    cancelVideoFrameCallback(handle) { cancelled = handle },
  }
  const seen = []
  const stop = startVideoPresentationClock(video, (time) => seen.push(time), {
    requestAnimationFrame: () => { throw new Error('fallback should not run') }, cancelAnimationFrame: () => undefined,
  })
  callback(0, { mediaTime: 3.25 })
  callback(0, { mediaTime: 3.3 })
  assert.deepEqual(seen, [3.25, 3.3])
  stop()
  assert.equal(cancelled, 17)
})

test('requestAnimationFrame fallback follows the same video currentTime on seek/playback', () => {
  let callback = null
  let cancelled = null
  const video = { currentTime: 0 }
  const seen = []
  const stop = startVideoPresentationClock(video, (time) => seen.push(time), {
    requestAnimationFrame(next) { callback = next; return 9 }, cancelAnimationFrame(handle) { cancelled = handle },
  })
  callback(0)
  video.currentTime = 4.5
  callback(0)
  assert.deepEqual(seen, [0, 4.5])
  stop()
  assert.equal(cancelled, 9)
})
