import { test } from 'node:test'
import assert from 'node:assert/strict'
import { frameSampleIndex, latestSampleAtMediaTime, measurementTrack, nearestSample, sampleAtMediaTime, stackedLensBox } from '../src/components/clipDebug.ts'

test('source lens evidence maps left/right coordinates into stacked dual-lens pixels', () => {
  assert.deepEqual(stackedLensBox({ lens: 'left', x_min: 0, y_min: 0, x_max: 2048, y_max: 1024 }), { left: 0, top: 0, width: 100, height: 50 })
  assert.equal(stackedLensBox({ lens: 'right', x_min: 100, y_min: 20, x_max: 300, y_max: 220 }).top, 50.9765625)
})

test('debug helpers select the nearest real sample and connect measurement to observed track', () => {
  assert.equal(nearestSample([{ timestamp_s: 0 }, { timestamp_s: .2 }], .14)?.timestamp_s, .2)
  assert.equal(measurementTrack([{ track_id: 17, measurement_id: '72:8' }], 72, 8)?.track_id, 17)
})

test('final samples use physical frame indices for ordinary playback', () => {
  const rows = [{ frame_index: 100, timestamp_s: 100 / 30 }, { frame_index: 101, timestamp_s: 101 / 30 }]
  const indexed = frameSampleIndex(rows)
  assert.equal(sampleAtMediaTime(rows, indexed, 100.49 / 30, 30)?.frame_index, 100)
  assert.equal(sampleAtMediaTime(rows, indexed, 100.51 / 30, 30)?.frame_index, 101)
})

test('sample lookup falls back to timestamp when a physical frame is unavailable', () => {
  const rows = [{ frame_index: 100, timestamp_s: 100 / 30 }, { frame_index: 102, timestamp_s: 102 / 30 }]
  assert.equal(sampleAtMediaTime(rows, frameSampleIndex(rows), 101 / 30, 30)?.frame_index, 100)
  assert.equal(sampleAtMediaTime(rows, frameSampleIndex(rows), 102 / 30, null)?.frame_index, 102)
})

test('sparse preview playback uses the last published sample without interpolation', () => {
  const rows = [{ frame_index: 0, timestamp_s: 0 }, { frame_index: 2, timestamp_s: 2 / 30 }]
  assert.equal(latestSampleAtMediaTime(rows, 1 / 30)?.frame_index, 0)
  assert.equal(latestSampleAtMediaTime(rows, 2 / 30)?.frame_index, 2)
  assert.equal(latestSampleAtMediaTime(rows, -.01), null)
})
