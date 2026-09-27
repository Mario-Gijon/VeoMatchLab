import { presentationDirectionFromVirtualPixel, type VirtualCameraSpec } from './trackingProjection.ts'

export interface OverlayAssociationSample {
  frame_index: number
  observations: Array<{ observation_id: number; view_name: string; xyxy: number[] }>
  measurements: Array<{ measurement_id: number; primary_observation_id: number; observation_ids: number[] }>
}

export interface RepresentativeObservation { view_name: string; xyxy: [number, number, number, number] }
export interface OverlayTrack { track_id: number; position_kind: string; measurement_id?: string | null }
export interface OverlayTrackFrame { frame_index?: number; timestamp_s: number; tracks: OverlayTrack[] }
export interface VisualAnchor { frame_index: number; timestamp_s: number; direction: [number, number, number] }

export const VISUAL_ANCHOR_MAX_GAP_S = .5

export function representativeObservation(track: { position_kind: string; measurement_id?: string | null }, association: OverlayAssociationSample[]): RepresentativeObservation | null {
  if (track.position_kind !== 'observed' || !track.measurement_id) return null
  const separator = track.measurement_id.indexOf(':')
  if (separator < 1) return null
  const frameIndex = Number(track.measurement_id.slice(0, separator))
  const measurementId = Number(track.measurement_id.slice(separator + 1))
  const sample = association.find((item) => item.frame_index === frameIndex)
  const measurement = sample?.measurements.find((item) => item.measurement_id === measurementId)
  const observation = measurement && sample?.observations.find((item) => item.observation_id === measurement.primary_observation_id)
  if (!observation || observation.xyxy.length !== 4 || !observation.xyxy.every(Number.isFinite)) return null
  return { view_name: observation.view_name, xyxy: [observation.xyxy[0], observation.xyxy[1], observation.xyxy[2], observation.xyxy[3]] }
}

/** Precompute source-image ray anchors once per immutable tracking/debug sequence. */
export function visualAnchorsByTrack(trackFrames: OverlayTrackFrame[], association: OverlayAssociationSample[], cameraSpecs: Record<string, VirtualCameraSpec>) {
  const result = new Map<number, VisualAnchor[]>()
  for (const frame of trackFrames) {
    const frameIndex = frame.frame_index
    if (typeof frameIndex !== 'number' || !Number.isInteger(frameIndex)) continue
    for (const track of frame.tracks) {
      const representative = representativeObservation(track, association)
      const spec = representative && cameraSpecs[representative.view_name]
      const foot = representative && { x: (representative.xyxy[0] + representative.xyxy[2]) / 2, y: representative.xyxy[3] }
      const direction = spec && foot ? presentationDirectionFromVirtualPixel(spec, foot.x, foot.y, 1280, 720) : null
      if (!direction) continue
      const anchors = result.get(track.track_id) ?? []
      anchors.push({ frame_index: frameIndex, timestamp_s: frame.timestamp_s, direction: [direction[0], direction[1], direction[2]] })
      result.set(track.track_id, anchors)
    }
  }
  for (const anchors of result.values()) anchors.sort((left, right) => left.frame_index - right.frame_index)
  return result
}

function normalizedLerp(left: number[], right: number[], alpha: number): [number, number, number] | null {
  const direction = left.map((value, index) => value * (1 - alpha) + right[index] * alpha)
  const norm = Math.hypot(...direction)
  return norm > 1e-12 && direction.every(Number.isFinite) ? [direction[0] / norm, direction[1] / norm, direction[2] / norm] : null
}

/** Return an exact observed ray, or a short bracketing nlerp ray for display only. */
export function visualDirectionForTrack(track: OverlayTrack, frameIndex: number | undefined, timestamp: number | undefined, anchorsByTrack: Map<number, VisualAnchor[]>, maxGapSeconds = VISUAL_ANCHOR_MAX_GAP_S): [number, number, number] | null {
  if (typeof frameIndex !== 'number' || !Number.isInteger(frameIndex) || typeof timestamp !== 'number' || !Number.isFinite(timestamp)) return null
  const anchors = anchorsByTrack.get(track.track_id)
  if (!anchors?.length) return null
  let low = 0, high = anchors.length
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (anchors[middle].frame_index <= frameIndex) low = middle + 1
    else high = middle
  }
  const nextIndex = low
  const exact = anchors[nextIndex - 1]
  if (track.position_kind === 'observed' && exact?.frame_index === frameIndex) return exact.direction
  if (track.position_kind !== 'interpolated' && track.position_kind !== 'predicted') return null
  const previous = anchors[nextIndex - 1], following = anchors[nextIndex]
  if (!previous || !following || following.timestamp_s - previous.timestamp_s > maxGapSeconds) return null
  const span = following.timestamp_s - previous.timestamp_s
  const alpha = span > 1e-9 ? (timestamp - previous.timestamp_s) / span : (frameIndex - previous.frame_index) / (following.frame_index - previous.frame_index)
  return alpha >= 0 && alpha <= 1 ? normalizedLerp(previous.direction, following.direction, alpha) : null
}
