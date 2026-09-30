import type { FinalPositionMode, SegmentBreak, TrackingFieldPoint, TrackingReview } from '../types/trackingFinal'

export interface FinalFrame { frameIndex: number; timestampS: number; points: TrackingFieldPoint[] }

/** Selects the nearest *published* Tracking V1 frame, never an interpolated UI value. */
export function nearestFinalFrame(points: TrackingFieldPoint[], timestampS: number): FinalFrame | null {
  const frames = new Map<number, FinalFrame>()
  for (const point of points) {
    const frame = frames.get(point.frame_index)
    if (frame) frame.points.push(point)
    else frames.set(point.frame_index, { frameIndex: point.frame_index, timestampS: point.timestamp_s, points: [point] })
  }
  return [...frames.values()].reduce<FinalFrame | null>((closest, frame) => !closest || Math.abs(frame.timestampS - timestampS) < Math.abs(closest.timestampS - timestampS) ? frame : closest, null)
}

export function pointCoordinates(point: TrackingFieldPoint, mode: FinalPositionMode) {
  if ('actor_id' in point) return mode === 'smoothed' ? { x: point.x, z: point.z } : { x: point.measurement_x, z: point.measurement_z }
  return mode === 'smoothed' ? { x: point.smoothed_x, z: point.smoothed_z } : { x: point.measurement_x, z: point.measurement_z }
}

/** Builds SVG polylines from actual points, splitting at each published break. */
export function trajectorySegments(points: TrackingFieldPoint[], segmentBreaks: SegmentBreak[]): TrackingFieldPoint[][] {
  const breakAfter = new Set(segmentBreaks.map((item) => `${item.identity_id}:${item.left_frame_index}:${item.right_frame_index}`))
  const ordered = [...points].sort((a, b) => a.frame_index - b.frame_index)
  const groups: TrackingFieldPoint[][] = []
  for (const point of ordered) {
    const current = groups.at(-1)
    const previous = current?.at(-1)
    const entityId = 'actor_id' in point ? point.actor_id : point.identity_id
    const separated = !!previous && (previous.segment_id !== point.segment_id || breakAfter.has(`${entityId}:${previous.frame_index}:${point.frame_index}`))
    if (!current || separated) groups.push([point])
    else current.push(point)
  }
  return groups
}

export function reviewSeekTimestamp(review: TrackingReview): number | null {
  const timestamp = review.timestamp_s ?? review.start_timestamp_s
  return typeof timestamp === 'number' && Number.isFinite(timestamp) ? timestamp : null
}
