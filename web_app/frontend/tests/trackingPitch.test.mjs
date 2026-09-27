import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { trackingPitchPoint, trackingPitchViewBox } from '../src/components/trackingPitchGeometry.ts'

test('tracking pitch keeps the center at the SVG center', () => assert.deepEqual(trackingPitchPoint(0, 0), { x: 0, y: 0 }))
test('tracking pitch preserves horizontal field extremes', () => {
  assert.equal(trackingPitchPoint(-105 / 2, 0).x, -105 / 2)
  assert.equal(trackingPitchPoint(105 / 2, 0).x, 105 / 2)
})
test('tracking pitch flips only the visual vertical axis', () => {
  assert.equal(trackingPitchPoint(0, 68 / 2).y, -68 / 2)
  assert.equal(trackingPitchPoint(0, -68 / 2).y, 68 / 2)
})
test('negative coordinates are never clamped', () => assert.equal(trackingPitchPoint(-12, -8).x, -12))
test('viewBox includes a small margin around the canonical field', () => assert.equal(trackingPitchViewBox(105, 68), '-54 -35.5 108 71'))
test('tracking pitch deliberately renders all position kinds with one marker style', () => {
  const source = readFileSync(new URL('../src/components/TrackingPitch.tsx', import.meta.url), 'utf8')
  assert.equal(source.includes('position_kind'), false)
})
