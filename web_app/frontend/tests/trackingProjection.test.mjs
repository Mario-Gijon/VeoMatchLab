import assert from 'node:assert/strict'
import test from 'node:test'
import { presentationDirectionFromVirtualPixel, projectVirtualPixelToUltra } from '../src/components/trackingProjection.ts'

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
const context = { uniforms: {
  camera2world: { value: identity }, camera2presentation: { value: identity },
  va_left: { value: -1 }, va_right: { value: 1 }, va_top: { value: 1 }, va_bottom: { value: -1 },
}, field_length: 105, field_width: 68 }

test('Ultra projection maps presentation center to video center', () => {
  assert.deepEqual(projectVirtualPixelToUltra(context, { longitude: 0, latitude: 0, vfov: Math.PI / 2 }, 50, 50, 100, 100), { u: .5, v: .5 })
})

test('representative virtual pixel uses the canonical camera ray', () => {
  const direction = presentationDirectionFromVirtualPixel({ longitude: 0, latitude: 0, vfov: Math.PI / 2 }, 50, 50, 100, 100)
  assert.ok(direction)
  assert.ok(Math.abs(direction[0]) < 1e-12 && Math.abs(direction[1]) < 1e-12 && direction[2] > 0)
})

test('Ultra projection rejects virtual pixels outside the presentation span', () => {
  assert.equal(projectVirtualPixelToUltra(context, { longitude: 0, latitude: 0, vfov: Math.PI / 2 }, 200, 50, 100, 100), null)
})

test('Ultra projection preserves the visual downward Y inversion', () => {
  const point = projectVirtualPixelToUltra(context, { longitude: 0, latitude: 0, vfov: Math.PI / 2 }, 50, 75, 100, 100)
  assert.ok(point && Number.isFinite(point.v) && point.v > .5)
})
