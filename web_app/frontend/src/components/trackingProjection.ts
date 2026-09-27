import type { ViewContext } from '../types'
import { inverseMatrix, multiplyMatrix } from './interactiveCamera.ts'

export function projectGroundPointToUltra(context: ViewContext, x: number, z: number, width: number, height: number) {
  const camera2world = context.uniforms.camera2world?.value
  const camera2presentation = context.uniforms.camera2presentation?.value
  const va = context.uniforms.va_left && context.uniforms.va_right && context.uniforms.va_top && context.uniforms.va_bottom
    ? { left: Number(context.uniforms.va_left.value), right: Number(context.uniforms.va_right.value), top: Number(context.uniforms.va_top.value), bottom: Number(context.uniforms.va_bottom.value) }
    : null
  if (!Array.isArray(camera2world) || !Array.isArray(camera2presentation) || !va || width <= 0 || height <= 0 || ![x, z, ...camera2world, ...camera2presentation, va.left, va.right, va.top, va.bottom].every(Number.isFinite)) return null
  const effectiveCamera2Presentation = inverseMatrix(camera2presentation)
  if (!effectiveCamera2Presentation) return null
  const cameraToWorldPresentation = multiplyMatrix(camera2world, effectiveCamera2Presentation)
  const worldToPresentation = inverseMatrix(cameraToWorldPresentation)
  if (!worldToPresentation) return null
  const origin = camera2world.slice(12, 15)
  const direction = [x - origin[0], -origin[1], z - origin[2], 0]
  const presentation = [0, 1, 2].map((row) => direction.reduce((sum, value, col) => sum + worldToPresentation[col * 4 + row] * value, 0))
  const norm = Math.hypot(...presentation)
  if (!(norm > 1e-12) || !presentation.every(Number.isFinite)) return null
  const lon = Math.atan2(presentation[0], presentation[2])
  const lat = Math.asin(Math.max(-1, Math.min(1, -presentation[1] / norm)))
  const horizontalSpan = va.right - va.left
  const verticalSpan = va.top - va.bottom
  if (!(horizontalSpan > 0 && verticalSpan > 0)) return null
  const u = (lon - va.left) / horizontalSpan
  const v = 1 - (lat - va.bottom) / verticalSpan
  return [u, v].every(Number.isFinite) && u >= 0 && u <= 1 && v >= 0 && v <= 1 ? { u, v } : null
}
