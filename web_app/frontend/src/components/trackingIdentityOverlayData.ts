import type { TrackingAssignmentStatus, TrackingIdentityObservation, TrackingIdentityTracklet } from '../types/trackingIdentityOverlay'

export interface ActiveIdentityObservation extends TrackingIdentityObservation {
  identityId: string
  trackletId: string
  assignmentStatus: TrackingAssignmentStatus
  interpolated: boolean
}

const interpolationGapS = .35
const timeEpsilonS = 1e-4

export function identityColor(identityId: string) {
  let hash = 2166136261
  for (let index = 0; index < identityId.length; index += 1) {
    hash ^= identityId.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return `hsl(${(hash >>> 0) % 360} 78% 58%)`
}

export function identityOverlayStyle(identityId: string, assignmentStatus: TrackingAssignmentStatus) {
  return assignmentStatus === 'unassigned'
    ? { color: '#d7dce2', dash: '7 5' }
    : { color: identityColor(identityId), dash: undefined }
}

function interpolate(first: TrackingIdentityObservation, second: TrackingIdentityObservation, time: number): TrackingIdentityObservation | null {
  const gap = second.timestamp_s - first.timestamp_s
  if (gap <= 0 || gap > interpolationGapS) return null
  const ratio = (time - first.timestamp_s) / gap
  if (ratio < 0 || ratio > 1) return null
  const bbox = first.bbox && second.bbox && first.source_lens === second.source_lens
    ? first.bbox.map((value, index) => value + (second.bbox![index] - value) * ratio) as TrackingIdentityObservation['bbox']
    : null
  return {
    frame_index: Math.round(first.frame_index + (second.frame_index - first.frame_index) * ratio),
    timestamp_s: time,
    x: first.x + (second.x - first.x) * ratio,
    z: first.z + (second.z - first.z) * ratio,
    bbox,
    source_lens: bbox ? first.source_lens : null,
  }
}

function observationAtTime(observations: TrackingIdentityObservation[], time: number, fps?: number | null) {
  if (fps && fps > 0) {
    const frame = Math.round(time * fps)
    const exact = observations.find((observation) => observation.frame_index === frame)
    if (exact) return { observation: exact, interpolated: false }
  }
  let left = 0, right = observations.length
  while (left < right) {
    const middle = Math.floor((left + right) / 2)
    if (observations[middle].timestamp_s < time) left = middle + 1
    else right = middle
  }
  const next = observations[left]
  const previous = observations[left - 1]
  if (next && Math.abs(next.timestamp_s - time) <= timeEpsilonS) return { observation: next, interpolated: false }
  if (previous && Math.abs(previous.timestamp_s - time) <= timeEpsilonS) return { observation: previous, interpolated: false }
  const result = previous && next ? interpolate(previous, next, time) : null
  return result ? { observation: result, interpolated: true } : null
}

/** Lookup is bounded per tracklet; it never carries a box beyond its evidence. */
export function activeIdentityObservations(tracks: TrackingIdentityTracklet[], time: number, fps?: number | null): ActiveIdentityObservation[] {
  if (!Number.isFinite(time)) return []
  const result: ActiveIdentityObservation[] = []
  for (const track of tracks) {
    const active = observationAtTime(track.observations, time, fps)
    if (active) result.push({ ...active.observation, identityId: track.identity_id, trackletId: track.tracklet_id, assignmentStatus: track.assignment_status, interpolated: active.interpolated })
  }
  return result
}
