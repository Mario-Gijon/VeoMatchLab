import type { ViewContext } from '../types'
import { inverseMatrix } from './interactiveCamera.ts'

export interface VirtualCameraSpec { longitude: number; latitude: number; vfov: number }

export function presentationDirectionFromVirtualPixel(spec: VirtualCameraSpec, pixelX: number, pixelY: number, width: number, height: number) {
  if (!(width > 0 && height > 0 && [spec.longitude, spec.latitude, spec.vfov, pixelX, pixelY].every(Number.isFinite))) return null
  const tan = Math.tan(spec.vfov / 2)
  const local = [(2 * pixelX / width - 1) * tan * width / height, (2 * pixelY / height - 1) * tan, 1]
  const cy = Math.cos(spec.longitude), sy = Math.sin(spec.longitude), cx = Math.cos(spec.latitude), sx = Math.sin(spec.latitude)
  const rotation = [[cy, sy * sx, sy * cx], [0, cx, -sx], [-sy, cy * sx, cy * cx]]
  const direction = [0, 1, 2].map((row) => local.reduce((sum, value, column) => sum + value * rotation[row][column], 0))
  const norm = Math.hypot(...direction)
  return norm > 1e-12 && direction.every(Number.isFinite) ? direction.map((value) => value / norm) : null
}

export function projectVirtualPixelToUltra(context: ViewContext, spec: VirtualCameraSpec, pixelX: number, pixelY: number, width: number, height: number) {
  const direction = presentationDirectionFromVirtualPixel(spec, pixelX, pixelY, width, height)
  const va = context.uniforms.va_left && context.uniforms.va_right && context.uniforms.va_top && context.uniforms.va_bottom
    ? { left: Number(context.uniforms.va_left.value), right: Number(context.uniforms.va_right.value), top: Number(context.uniforms.va_top.value), bottom: Number(context.uniforms.va_bottom.value) }
    : null
  if (!direction || !va) return null
  const horizontalSpan = va.right - va.left, verticalSpan = va.top - va.bottom
  if (!(horizontalSpan > 0 && verticalSpan > 0)) return null
  const lon = Math.atan2(direction[0], direction[2])
  const lat = Math.asin(Math.max(-1, Math.min(1, -direction[1])))
  const u = (lon - va.left) / horizontalSpan, v = 1 - (lat - va.bottom) / verticalSpan
  return [u, v].every(Number.isFinite) && u >= 0 && u <= 1 && v >= 0 && v <= 1 ? { u, v } : null
}

export function projectVirtualRayToInteractive(viewToWorld: number[], fov: number, direction: number[], aspect: number) {
  if (direction.length !== 3 || !(aspect > 0) || !Number.isFinite(fov) || !direction.every(Number.isFinite)) return null
  const worldToView = inverseMatrix(viewToWorld)
  if (!worldToView) return null
  const view = [direction[0], direction[1], direction[2], 0].map((_, row) => [direction[0], direction[1], direction[2], 0].reduce((sum, value, column) => sum + worldToView[column * 4 + row] * value, 0))
  const height = 2 * Math.tan(fov / 2), viewportWidth = height * aspect
  if (!(view[2] > 1e-9) || ![...view, height, viewportWidth].every(Number.isFinite)) return null
  const u = .5 + view[0] / view[2] / viewportWidth, v = .5 + view[1] / view[2] / height
  return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? { u, v } : null
}
