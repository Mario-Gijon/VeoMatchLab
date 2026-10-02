import type { TrackingIdentityObservation, TrackingIdentityTracklet } from '../types/trackingIdentityOverlay'

export interface ActiveIdentityObservation extends TrackingIdentityObservation {
  identityId: string
  trackletId: string
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

function interpolate(first: TrackingIdentityObservation, second: TrackingIdentityObservation, time: number): TrackingIdentityObservation | null {
  const gap = second.timestamp_s - first.timestamp_s
  if (gap <= 0 || gap > interpolationGapS || first.source_lens !== second.source_lens) return null
  const ratio = (time - first.timestamp_s) / gap
  if (ratio < 0 || ratio > 1) return null
  return {
    timestamp_s: time,
    bbox: first.bbox.map((value, index) => value + (second.bbox[index] - value) * ratio) as TrackingIdentityObservation['bbox'],
    source_lens: first.source_lens,
  }
}

function observationAtTime(observations: TrackingIdentityObservation[], time: number) {
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
export function activeIdentityObservations(tracks: TrackingIdentityTracklet[], time: number): ActiveIdentityObservation[] {
  if (!Number.isFinite(time)) return []
  const result: ActiveIdentityObservation[] = []
  for (const track of tracks) {
    const active = observationAtTime(track.observations, time)
    if (active) result.push({ ...active.observation, identityId: track.identity_id, trackletId: track.tracklet_id, interpolated: active.interpolated })
  }
  return result
}
