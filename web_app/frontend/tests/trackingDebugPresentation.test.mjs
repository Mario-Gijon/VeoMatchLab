import assert from 'node:assert/strict'
import test from 'node:test'
import { activePresentation, controlPresentationPlayback, hasActivePresentation, synchronizePresentations, timeForViewChange } from '../src/components/trackingDebugPresentation.ts'
import { startVideoPresentationClock } from '../src/components/videoPresentationClock.ts'

const presentation = (time) => ({ currentTime: time, pauses: 0, plays: 0, pause() { this.pauses++ }, play() { this.plays++; return Promise.resolve() } })

test('changing Tracking Debug views preserves the active presentation timestamp', () => {
  const original = presentation(7.35), ultra = presentation(0), interactive = presentation(0)
  const views = { original, ultra, interactive }
  const time = timeForViewChange('original', views, 0, 90)
  assert.equal(time, 7.35)
  synchronizePresentations(views, time)
  assert.equal(ultra.currentTime, 7.35)
  assert.equal(interactive.currentTime, 7.35)
})

test('seek synchronizes every mounted view and only the active presentation plays', () => {
  const original = presentation(0), ultra = presentation(0), interactive = presentation(0)
  const views = { original, ultra, interactive }
  synchronizePresentations(views, 12.5)
  assert.deepEqual([original.currentTime, ultra.currentTime, interactive.currentTime], [12.5, 12.5, 12.5])
  const active = activePresentation('interactive', views)
  controlPresentationPlayback(views, active, true)
  assert.equal(interactive.plays, 1)
  assert.equal(original.pauses, 1)
  assert.equal(ultra.pauses, 1)
})

test('an unselected player still plays and publishes Tracking Debug presentation time', () => {
  const selectedIdentityId = null
  let callback = null
  const original = { ...presentation(0), requestVideoFrameCallback(next) { callback = next; return 1 }, cancelVideoFrameCallback() {} }
  const views = { original, ultra: null, interactive: null }
  assert.equal(selectedIdentityId, null)
  assert.equal(hasActivePresentation('original', { hasSource: true, hasUltra: false, hasContext: false, ultraFailed: false }), true)
  controlPresentationPlayback(views, original, true)
  const seen = []
  const stop = startVideoPresentationClock(original, (time) => seen.push(time), { requestAnimationFrame: () => 0, cancelAnimationFrame: () => undefined })
  callback(0, { mediaTime: 4.2 })
  stop()
  assert.equal(original.plays, 1)
  assert.deepEqual(seen, [4.2])
})

test('view availability is unchanged by follow toggles and does not own playback state', () => {
  const availability = { hasSource: true, hasUltra: true, hasContext: true, ultraFailed: false }
  let playing = true, time = 7.35
  for (const followPlayer of [true, false]) {
    assert.equal(typeof followPlayer, 'boolean')
    assert.equal(hasActivePresentation('interactive', availability), true)
  }
  assert.equal(playing, true)
  assert.equal(time, 7.35)
})
