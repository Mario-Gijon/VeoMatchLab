export interface OverlayAssociationSample {
  frame_index: number
  observations: Array<{ observation_id: number; view_name: string; xyxy: number[] }>
  measurements: Array<{ measurement_id: number; primary_observation_id: number; observation_ids: number[] }>
}

export interface RepresentativeObservation { view_name: string; xyxy: [number, number, number, number] }

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
