import assert from 'node:assert/strict'
import test from 'node:test'
import { activePresentation, controlPresentationPlayback, synchronizePresentations, timeForViewChange } from '../src/components/trackingDebugPresentation.ts'

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
