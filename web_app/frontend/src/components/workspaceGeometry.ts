import type { GroundPoint } from '../types'

export const VIEW_MODES = ['tracking', 'interactive', 'panorama', 'dual_lens'] as const
export type ViewMode = typeof VIEW_MODES[number]

export function viewLabel(view: ViewMode) {
  return view === 'tracking' ? 'Seguimiento de balón' : view === 'interactive' ? 'Interactivo' : view === 'panorama' ? 'Panorámica' : 'Doble lente'
}

/** The media element is the Workspace clock, not React's timeupdate cadence. */
export function matchTimestamp(video: { currentTime: number } | null, fallback: number) {
  return video && Number.isFinite(video.currentTime) ? video.currentTime : fallback
}

export function validPitchPoint(point: GroundPoint) {
  return point.measured_x !== null && point.measured_z !== null
    && Number.isFinite(point.measured_x) && Number.isFinite(point.measured_z)
    && (point.status === 'inside_field' || point.status === 'outside_field')
}

/** All stored points that have a usable canonical field position. */
export function pitchPoints(points: GroundPoint[]) {
  return points.filter(validPitchPoint)
}

/** The newest valid canonical point drives the live Tracking/Interactive overlay. */
export function latestSelectedPitchPoint(points: GroundPoint[]) {
  return pitchPoints(points).at(-1) ?? null
}

export function validationStatistics(points: GroundPoint[]) {
  const errors = points.map((point) => point.field_error_m).filter((value): value is number => value !== null && Number.isFinite(value))
  const ordered = [...errors].sort((a, b) => a - b)
  const landmarkPoints = points.filter((point) => point.field_error_m !== null && Number.isFinite(point.field_error_m))
  return {
    n: errors.length,
    median: errors.length ? (ordered[Math.floor((errors.length - 1) / 2)] + ordered[Math.floor(errors.length / 2)]) / 2 : null,
    rms: errors.length ? Math.sqrt(errors.reduce((sum, value) => sum + value ** 2, 0) / errors.length) : null,
    max: errors.length ? Math.max(...errors) : null,
    mean: errors.length ? errors.reduce((sum, value) => sum + value, 0) / errors.length : null,
    meanDeltaX: landmarkPoints.length ? landmarkPoints.reduce((sum, point) => sum + (point.delta_x ?? 0), 0) / landmarkPoints.length : null,
    meanDeltaZ: landmarkPoints.length ? landmarkPoints.reduce((sum, point) => sum + (point.delta_z ?? 0), 0) / landmarkPoints.length : null,
  }
}

export function appendMeasurement(points: GroundPoint[], point: GroundPoint) {
  if ((point.view === 'tracking' || point.view === 'interactive')
    && (!validPitchPoint(point) || point.status !== 'inside_field')) return points
  return [...points, { ...point, point_index: points.length + 1 }]
}

export function canSelect(selectionEnabled: boolean, isValidGround: boolean) {
  return selectionEnabled && isValidGround
}
