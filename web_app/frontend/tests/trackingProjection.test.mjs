import assert from 'node:assert/strict'
import test from 'node:test'
import { projectGroundPointToUltra } from '../src/components/trackingProjection.ts'

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
const context = { uniforms: {
  camera2world: { value: identity }, camera2presentation: { value: identity },
  va_left: { value: -1 }, va_right: { value: 1 }, va_top: { value: 1 }, va_bottom: { value: -1 },
}, field_length: 105, field_width: 68 }

test('Ultra projection maps presentation center to video center', () => {
  assert.deepEqual(projectGroundPointToUltra(context, 0, 1, 200, 100), { u: .5, v: .5 })
})

test('Ultra projection rejects points outside the presentation span', () => {
  assert.equal(projectGroundPointToUltra(context, 10, 1, 200, 100), null)
})

test('Ultra projection uses the visual downward Y convention and finite output', () => {
  const elevated = { ...context, uniforms: { ...context.uniforms, camera2world: { value: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 10, 0, 1] } } }
  const point = projectGroundPointToUltra(elevated, 0, 10, 200, 100)
  assert.ok(point && Number.isFinite(point.u) && Number.isFinite(point.v))
  assert.ok(point.v < .5)
})
