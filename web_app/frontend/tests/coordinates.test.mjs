import { test } from 'node:test'
import assert from 'node:assert/strict'
import { naturalPixel, nativeVideoPixel, stackedPixel } from '../src/components/viewCoordinates.ts'

test('responsive clicks select the same natural pixel center', () => {
  for (const scale of [.25, .5, 1, 1.7]) {
    const bounds = { left: 20, top: 50, width: 3840 * scale, height: 2160 * scale }
    assert.deepEqual(naturalPixel(20 + 2345.5 * scale, 50 + 1456.5 * scale, bounds, 3840, 2160), {x: 2345, y: 1456})
  }
})
test('ultra panorama display clicks map to its natural calibrated pixel grid', () => {
  const width = 4077
  const height = 1024
  const bounds = { left: 20, top: 50, width: 1200, height: 302 }
  assert.deepEqual(naturalPixel(20 + 600.5 * 1200 / width, 50 + 512.5 * 302 / height, bounds, width, height), { x: 600, y: 512 })
})
test('edges clamp and crop clicks use stacked coordinates', () => {
  assert.deepEqual(naturalPixel(100, 100, {left: 0, top: 0, width: 100, height: 100}, 2048, 2048), {x: 2047, y: 2047})
  assert.deepEqual(stackedPixel(100, 500, 'left'), {x: 100, y: 500})
  assert.deepEqual(stackedPixel(100, 500, 'right'), {x: 100, y: 1524})
})
test('letterboxed tracking clicks map to native pixels', () => {
  assert.deepEqual(nativeVideoPixel(500, 300, {left: 0, top: 0, width: 1000, height: 600}, 1920, 1080), {x: 960, y: 540})
})

test('Tracking rejects black bars and invalid sizes after resizing', () => {
  for (const scale of [.5, 1, 2]) {
    const bounds = { left: 20, top: 30, width: 1000 * scale, height: 1000 * scale }
    assert.deepEqual(nativeVideoPixel(20 + 500 * scale, 30 + 500 * scale, bounds, 1920, 1080), { x: 960, y: 540 })
    assert.equal(nativeVideoPixel(20 + 500 * scale, 30 + 10 * scale, bounds, 1920, 1080), null)
    assert.equal(nativeVideoPixel(20 + 500 * scale, 30 + 990 * scale, bounds, 1920, 1080), null)
  }
  const wide = { left: 0, top: 0, width: 1000, height: 200 }
  assert.equal(nativeVideoPixel(50, 100, wide, 1920, 1080), null)
  assert.equal(nativeVideoPixel(950, 100, wide, 1920, 1080), null)
  assert.equal(nativeVideoPixel(0, 0, { left: 0, top: 0, width: 0, height: 0 }, 1920, 1080), null)
  assert.equal(nativeVideoPixel(NaN, 0, wide, 1920, 1080), null)
})
