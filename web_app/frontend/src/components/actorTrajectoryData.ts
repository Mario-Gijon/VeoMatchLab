import type { ActorTrajectory, ActorTrajectoryPoint, ActorTrajectoryVisualEndpoint } from '../types/actorTrajectory'
import type { VirtualCameraSpec } from './trackingProjection'
import { presentationDirectionForObservation } from './trackingOverlayData.ts'

const TIME_TOLERANCE_S = .02

export interface ActiveActorTrajectoryPoint extends ActorTrajectoryPoint {
  actorId: string
  trackletIds: string[]
}

function closestPoint(points: ActorTrajectoryPoint[], mediaTime: number, fps?: number | null) {
  if (!points.length || !Number.isFinite(mediaTime)) return null
  const frame = fps && Number.isFinite(fps) && fps > 0 ? Math.round(mediaTime * fps) : null
  if (frame !== null) {
    let left = 0, right = points.length
    while (left < right) {
      const middle = Math.floor((left + right) / 2)
      if (points[middle].frame_index < frame) left = middle + 1
      else right = middle
    }
    if (points[left]?.frame_index === frame) return points[left]
  }
  let low = 0, high = points.length
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (points[middle].timestamp_s < mediaTime) low = middle + 1
    else high = middle
  }
  const candidates = [points[low - 1], points[low]].filter((point): point is ActorTrajectoryPoint => !!point)
  return candidates.reduce<ActorTrajectoryPoint | null>((best, point) => !best || Math.abs(point.timestamp_s - mediaTime) < Math.abs(best.timestamp_s - mediaTime) ? point : best, null)
}

/** Lookup is exact by video frame when available and never carries an actor through a preserved gap. */
export function activeActorTrajectoryPoints(actors: ActorTrajectory[], mediaTime: number, fps?: number | null): ActiveActorTrajectoryPoint[] {
  const active: ActiveActorTrajectoryPoint[] = []
  for (const actor of actors) {
    const point = closestPoint(actor.points, mediaTime, fps)
    if (!point || Math.abs(point.timestamp_s - mediaTime) > TIME_TOLERANCE_S) continue
    active.push({ ...point, actorId: actor.actor_id, trackletIds: actor.tracklet_ids })
  }
  return active.sort((left, right) => left.actorId.localeCompare(right.actorId))
}

function directionForEndpoint(endpoint: ActorTrajectoryVisualEndpoint | null | undefined, cameraSpecs: Record<string, VirtualCameraSpec>) {
  return endpoint ? presentationDirectionForObservation(endpoint.representative_view, endpoint.virtual_bbox_xyxy, cameraSpecs) : null
}

function normalizedLerp(left: number[], right: number[], alpha: number): [number, number, number] | null {
  if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1 || left.length !== 3 || right.length !== 3 || !left.every(Number.isFinite) || !right.every(Number.isFinite)) return null
  const direction = left.map((value, index) => value * (1 - alpha) + right[index] * alpha)
  const norm = Math.hypot(...direction)
  return norm > 1e-12 && direction.every(Number.isFinite) ? [direction[0] / norm, direction[1] / norm, direction[2] / norm] : null
}

/** Visual placement follows evidence rays; field x/z is deliberately not part of this path. */
export function presentationDirectionForActorPoint(point: ActorTrajectoryPoint, cameraSpecs: Record<string, VirtualCameraSpec>) {
  if (point.origin === 'observed') return directionForEndpoint(point.visual_anchor, cameraSpecs)
  const interpolation = point.visual_interpolation
  if (!interpolation?.visual_interpolation_allowed) return null
  const left = directionForEndpoint(interpolation.left, cameraSpecs), right = directionForEndpoint(interpolation.right, cameraSpecs)
  return left && right ? normalizedLerp(left, right, interpolation.alpha) : null
}

/** Source bboxes remain native physical-composite coordinates and only interpolate within one lens. */
export function sourceBboxForActorPoint(point: ActorTrajectoryPoint) {
  if (point.origin === 'observed') return point.visual_anchor?.source_bbox_xyxy ?? null
  const interpolation = point.visual_interpolation
  if (!interpolation?.source_bbox_interpolation_allowed || interpolation.left.source_lens !== interpolation.right.source_lens) return null
  const left = interpolation.left.source_bbox_xyxy, right = interpolation.right.source_bbox_xyxy
  if (!left || !right || !Number.isFinite(interpolation.alpha)) return null
  return left.map((value, index) => value * (1 - interpolation.alpha) + right[index] * interpolation.alpha) as [number, number, number, number]
}
