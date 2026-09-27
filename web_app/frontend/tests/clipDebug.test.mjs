import { test } from 'node:test'
import assert from 'node:assert/strict'
import { measurementTrack, nearestSample, stackedLensBox } from '../src/components/clipDebug.ts'

test('source lens evidence maps left/right coordinates into stacked dual-lens pixels', () => {
  assert.deepEqual(stackedLensBox({ lens: 'left', x_min: 0, y_min: 0, x_max: 2048, y_max: 1024 }), { left: 0, top: 0, width: 100, height: 50 })
  assert.equal(stackedLensBox({ lens: 'right', x_min: 100, y_min: 20, x_max: 300, y_max: 220 }).top, 50.9765625)
})

test('debug helpers select the nearest real sample and connect measurement to observed track', () => {
  assert.equal(nearestSample([{ timestamp_s: 0 }, { timestamp_s: .2 }], .14)?.timestamp_s, .2)
  assert.equal(measurementTrack([{ track_id: 17, measurement_id: '72:8' }], 72, 8)?.track_id, 17)
})
