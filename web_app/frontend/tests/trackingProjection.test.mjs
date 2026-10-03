import assert from 'node:assert/strict'
import test from 'node:test'
import { initialCamera, interactiveTransform, inverseMatrix, multiplyMatrix } from '../src/components/interactiveCamera.ts'
import { presentationDirectionFromVirtualPixel, projectGroundPointToUltra, projectVirtualPixelToUltra, projectVirtualRayToInteractive } from '../src/components/trackingProjection.ts'

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
const context = { uniforms: {
  camera2world: { value: identity }, camera2presentation: { value: identity },
  va_left: { value: -1 }, va_right: { value: 1 }, va_top: { value: 1 }, va_bottom: { value: -1 },
}, field_length: 105, field_width: 68 }

const rotationY = (angle) => [Math.cos(angle), 0, -Math.sin(angle), 0, 0, 1, 0, 0, Math.sin(angle), 0, Math.cos(angle), 0, 0, 0, 0, 1]

function transform(matrix, vector) {
  return [0, 1, 2, 3].map((row) => vector.reduce((sum, value, column) => sum + matrix[column * 4 + row] * value, 0))
}

function expectedInteractiveProjection(ctx, camera, direction, aspect) {
  const presentationToWorld = multiplyMatrix(ctx.uniforms.camera2world.value, ctx.uniforms.camera2presentation.value)
  const worldDirection = transform(presentationToWorld, [...direction, 0])
  const worldToView = inverseMatrix(interactiveTransform(ctx, camera))
  assert.ok(worldToView)
  const view = transform(worldToView, worldDirection)
  const height = 2 * Math.tan(camera.fov / 2)
  const viewportWidth = height * aspect
  return { u: .5 + view[0] / view[2] / viewportWidth, v: .5 + view[1] / view[2] / height }
}

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

test('fixed roster ground points project deterministically without invalid values', () => {
  assert.deepEqual(projectGroundPointToUltra(context, 0, 10), { u: .5, v: .5 })
  assert.equal(projectGroundPointToUltra(context, 100, 10), null)
  const point = projectGroundPointToUltra(context, 2, 10)
  assert.ok(point && Number.isFinite(point.u) && Number.isFinite(point.v))
})

test('interactive projection converts presentation rays to world before applying the view', () => {
  const ctx = { ...context, uniforms: { ...context.uniforms,
    camera2world: { value: identity }, camera2presentation: { value: rotationY(.35) } } }
  const direction = presentationDirectionFromVirtualPixel({ longitude: .1, latitude: 0, vfov: Math.PI / 2 }, 50, 50, 100, 100)
  assert.ok(direction)
  const cameras = [
    initialCamera(ctx),
    { ...initialCamera(ctx), longitude: .25 },
    { ...initialCamera(ctx), longitude: -.2, fov: Math.PI / 2.4 },
  ]
  for (const camera of cameras) {
    const actual = projectVirtualRayToInteractive(ctx, interactiveTransform(ctx, camera), camera.fov, direction, 16 / 9)
    const expected = expectedInteractiveProjection(ctx, camera, direction, 16 / 9)
    assert.ok(actual)
    assert.ok(Math.abs(actual.u - expected.u) < 1e-12)
    assert.ok(Math.abs(actual.v - expected.v) < 1e-12)
  }
})
