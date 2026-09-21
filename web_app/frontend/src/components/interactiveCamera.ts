import type { GroundPoint, ViewContext } from '../types'

/** Radians in the validated Panorama presentation frame; vertical perspective FOV. */
export interface InteractiveCamera { longitude: number; latitude: number; fov: number }
export const PITCH_LIMIT = 85 * Math.PI / 180
export const MIN_FOV = 20 * Math.PI / 180
export const MAX_FOV = 100 * Math.PI / 180
export const DRAG_THRESHOLD = 6 // CSS pixels, independent of resolution/DPR
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
const finite = (v: number, fallback: number) => Number.isFinite(v) ? v : fallback

export function safeCamera(camera: InteractiveCamera): InteractiveCamera {
  const lon = finite(camera.longitude, 0)
  return { longitude: ((lon + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI,
    latitude: clamp(finite(camera.latitude, 0), -PITCH_LIMIT, PITCH_LIMIT),
    fov: clamp(finite(camera.fov, Math.PI / 3), MIN_FOV, MAX_FOV) }
}

export function initialCamera(context: ViewContext): InteractiveCamera {
  const value = (name: string) => Number(context.uniforms[name]?.value ?? 0)
  return safeCamera({ longitude: (value('va_left') + value('va_right')) / 2,
    latitude: (value('va_bottom') + value('va_top')) / 2, fov: Math.PI / 3 })
}

export function dragCamera(start: InteractiveCamera, dx: number, dy: number, width: number, height: number) {
  if (!(width > 0 && height > 0) || !Number.isFinite(dx + dy)) return safeCamera(start)
  const horizontalFov = 2 * Math.atan(Math.tan(start.fov / 2) * width / height)
  return safeCamera({ ...start, longitude: start.longitude - dx * horizontalFov / width,
    latitude: start.latitude + dy * start.fov / height })
}

export function wheelCamera(camera: InteractiveCamera, delta: number) {
  return safeCamera({ ...camera, fov: camera.fov * Math.exp(clamp(finite(delta, 0), -1000, 1000) * .001) })
}

export function isDrag(dx: number, dy: number) { return Math.hypot(dx, dy) >= DRAG_THRESHOLD }

// All matrices are column-major, exactly as supplied to WebGL by view_context.
export function multiplyMatrix(a: number[], b: number[]) {
  return Array.from({ length: 16 }, (_, i) => {
    const row = i % 4, col = Math.floor(i / 4)
    return [0, 1, 2, 3].reduce((sum, k) => sum + a[k * 4 + row] * b[col * 4 + k], 0)
  })
}

export function interactiveTransform(context: ViewContext, camera: InteractiveCamera): number[] {
  const { longitude: lon, latitude: lat } = camera
  const cy = Math.cos(lon), sy = Math.sin(lon), cx = Math.cos(lat), sx = Math.sin(lat)
  const ry = [cy, 0, -sy, 0, 0, 1, 0, 0, sy, 0, cy, 0, 0, 0, 0, 1]
  const rx = [1, 0, 0, 0, 0, cx, sx, 0, 0, -sx, cx, 0, 0, 0, 0, 1]
  return multiplyMatrix(multiplyMatrix(context.uniforms.camera2world.value as number[],
    context.uniforms.camera2presentation.value as number[]), multiplyMatrix(ry, rx))
}

export function inverseMatrix(matrix: number[]): number[] | null {
  const a = Array.from({ length: 4 }, (_, row) => Array.from({ length: 8 }, (_, col) => col < 4 ? matrix[col * 4 + row] : col - 4 === row ? 1 : 0))
  for (let col = 0; col < 4; col++) {
    const pivot = Array.from({ length: 4 - col }, (_, i) => i + col).reduce((best, row) => Math.abs(a[row][col]) > Math.abs(a[best][col]) ? row : best, col)
    if (!Number.isFinite(a[pivot][col]) || Math.abs(a[pivot][col]) < 1e-12) return null
    ;[a[col], a[pivot]] = [a[pivot], a[col]]
    const value = a[col][col]; for (let j = 0; j < 8; j++) a[col][j] /= value
    for (let row = 0; row < 4; row++) if (row !== col) { const scale = a[row][col]; for (let j = 0; j < 8; j++) a[row][j] -= scale * a[col][j] }
  }
  return Array.from({ length: 16 }, (_, i) => a[i % 4][4 + Math.floor(i / 4)])
}

export function projectGroundPoint(viewToWorld: number[], fov: number, aspect: number, x: number, z: number) {
  const worldToView = inverseMatrix(viewToWorld)
  if (!worldToView || ![fov, aspect, x, z].every(Number.isFinite)) return null
  const point = [x, 0, z, 1]
  const view = [0, 1, 2, 3].map(row => point.reduce((sum, value, col) => sum + worldToView[col * 4 + row] * value, 0))
  const depth = view[2], height = 2 * Math.tan(fov / 2), viewportWidth = height * aspect
  if (!(depth > 1e-9) || ![...view, height, viewportWidth].every(Number.isFinite)) return null
  const u = .5 + view[0] / depth / viewportWidth, v = .5 + view[1] / depth / height
  return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? { u, v } : null
}

/** Same pixel-center and top/bottom conventions as Panorama. No .det involved. */
export function interactiveRay(context: ViewContext, camera: InteractiveCamera, x: number, y: number, width: number, height: number) {
  if (![camera.longitude, camera.latitude, camera.fov, x, y, width, height].every(Number.isFinite)
    || Math.abs(camera.longitude) > Math.PI || Math.abs(camera.latitude) > PITCH_LIMIT
    || camera.fov < MIN_FOV || camera.fov > MAX_FOV
    || width <= 0 || height <= 0 || x < 0 || y < 0 || x >= width || y >= height) return null
  const tan = Math.tan(camera.fov / 2)
  const u = (x + .5) / width, v = (y + .5) / height
  const view = [(2 * u - 1) * tan * width / height, (2 * v - 1) * tan, 1]
  const transform = interactiveTransform(context, camera)
  const direction = [0, 1, 2].map((row) => view.reduce((sum, value, k) => sum + transform[k * 4 + row] * value, 0))
  const world = context.uniforms.camera2world.value as number[]
  const origin = world.slice(12, 15)
  return [...origin, ...direction].every(Number.isFinite) ? { origin, direction, u, v } : null
}

export function interactiveGroundPoint(context: ViewContext, camera: InteractiveCamera, x: number, y: number, width: number, height: number): GroundPoint | null {
  const ray = interactiveRay(context, camera, x, y, width, height)
  if (!ray || Math.abs(ray.direction[1]) < 1e-12) return null
  const t = -ray.origin[1] / ray.direction[1]
  const world = ray.origin.map((value, i) => value + t * ray.direction[i])
  // Conservative ground-only selection: do not turn stands/sky or distant ground
  // extensions into field markers. This is not semantic scene/occlusion detection.
  if (!(t > 0) || !world.every(Number.isFinite) || Math.abs(world[0]) > context.field_length / 2
    || Math.abs(world[2]) > context.field_width / 2) return null
  return { pixel_x: x, pixel_y: y, u: ray.u, v: 1 - ray.v, world_x: world[0], world_z: world[2],
    measured_x: world[0], measured_z: world[2], distance: t * Math.hypot(...ray.direction),
    inside_field: true, status: 'inside_field', landmark: null, landmark_name: null,
    expected_x: null, expected_z: null, delta_x: null, delta_z: null, field_error_m: null }
}

/** Only the output ray changes. Captured lens sampling, compression and blending
 * are retained verbatim; the unmodified Panorama branch remains independently usable. */
export function interactiveShader(fragment: string) {
  const begin = fragment.indexOf('  // *** Calculation for PANORAMA ONLY *** //')
  const endMarker = '  // *** END *** //'
  const endIndex = fragment.indexOf(endMarker, begin)
  if (begin < 0 || endIndex < begin || !fragment.includes('void main(){')) throw new Error('Unsupported Panorama shader layout')
  const end = endIndex + endMarker.length
  const extended = fragment.slice(0, begin) + `
  if (interactive_enabled > 0.5) {
    vec3 view_ray = vec3((2.0 * uv.x - 1.0) * interactive_scale.x,
                         (1.0 - 2.0 * uv.y) * interactive_scale.y, 1.0);
    worldspace_projection_dir = interactive_to_world * vec4(view_ray, 0.0);
  } else {
` + fragment.slice(begin, end) + '\n  }\n' + fragment.slice(end)
  // The captured expression divides by ray.y even at the exact horizon. Guard
  // only the new perspective mode; Panorama keeps its original expression.
  const projection = 'vec4 projection_direction = worldspace_projection_dir * (up_down * (-worldspace_cam_pos.y/worldspace_projection_dir.y) + (1.0-up_down)  * projection_bubble_radius);'
  if (!extended.includes(projection)) throw new Error('Unsupported Panorama projection layout')
  return extended.replace(projection, `vec4 projection_direction;
  if (interactive_enabled > 0.5 && abs(worldspace_projection_dir.y) < 0.000001) {
    projection_direction = worldspace_projection_dir * projection_bubble_radius;
  } else {
    ${projection.replace('vec4 ', '')}
  }`)
}

// Declarations follow precision, before main; captured helpers stay untouched.
export const INTERACTIVE_UNIFORMS = `
uniform float interactive_enabled;
uniform mat4 interactive_to_world;
uniform vec2 interactive_scale;
`
