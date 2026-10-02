import type { Calibration, Clip, ClipJob, GroundPoint, MatchWorkspace, MediaItem, RenderedPanorama, VeoInspection, VeoJob, ViewContext } from '../types'
import type { TrackingFinalArtifact, TrackingPersistentArtifact, TrackingReviews } from '../types/trackingFinal'
import type { TrackingIdentityOverlayArtifact } from '../types/trackingIdentityOverlay'

// Production and LAN development use the Vite origin. The optional override
// remains useful for isolated development or an externally hosted API.
const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/+$/, '')
const baseUrl = configuredBaseUrl ?? ''
const url = (path: string) => `${baseUrl}${path.startsWith('/') ? path : `/${path}`}`

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url(path), init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: { message?: string } | string }
    const detail = typeof body.detail === 'object' ? body.detail?.message : body.detail
    throw new Error(detail || `Request failed (${response.status}).`)
  }
  return response.json() as Promise<T>
}

export const api = {
  viewContext: (id: string) => request<ViewContext>(`/api/matches/${id}/view-context`),
  setRole: (id: string, role: string) => request<MediaItem>(`/api/media/${id}/role`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role }) }),
  viewPoint: (contextId: string, view: string, x: number, y: number, width: number, height: number, landmarkId: string | null) => request<GroundPoint>('/api/view-point', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ context_id: contextId, view, x, y, width, height, landmark_id: landmarkId }) }),
  trackingPoint: (matchId: string, timestamp: number, x: number, y: number, width: number, height: number) => request<GroundPoint>(`/api/matches/${matchId}/tracking-point`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ timestamp_s: timestamp, x, y, width, height }) }),
  url,
  health: () => request<{ status: string }>('/api/health'),
  calibrations: () => request<Calibration[]>('/api/calibrations'),
  matches: () => request<MatchWorkspace[]>('/api/matches'),
  match: (id: string) => request<MatchWorkspace>(`/api/matches/${id}`),
  clips: (id: string) => request<Clip[]>(`/api/matches/${id}/clips`),
  createClip: (matchId: string, sourceMediaId: string, startS: number, durationS: number, name: string) => request<ClipJob>(`/api/matches/${matchId}/clips`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source_media_id: sourceMediaId, start_s: startS, duration_s: durationS, name }) }),
  materializeClip: (matchId: string, clipId: string) => request<ClipJob>(`/api/matches/${matchId}/clips/${clipId}/materialize`, { method: 'POST' }),
  startClipAnalysis: (matchId: string, clipId: string, action: 'all' | 'tracking', trackingMode: 'baseline' | 'v8_canonical' = 'baseline', detectionThreshold = 0.40) => request<ClipJob>(`/api/matches/${matchId}/clips/${clipId}/analysis`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action, tracking_mode: trackingMode, detection_threshold: detectionThreshold }) }),
  clipJob: (jobId: string) => request<ClipJob>(`/api/clip-jobs/${jobId}`),
  cancelClipJob: (jobId: string) => request<ClipJob>(`/api/clip-jobs/${jobId}`, { method: 'DELETE' }),
  deleteClip: (matchId: string, clipId: string) => request<{ deleted: boolean }>(`/api/matches/${matchId}/clips/${clipId}`, { method: 'DELETE' }),
  clipViews: (matchId: string, clipId: string, timestamp: number, view: 'cameras' | 'ultra') => request<{ timestamp_s: number; cameras: Record<string, string>; ultra_panorama: string | null }>(`/api/matches/${matchId}/clips/${clipId}/development-views?timestamp_s=${encodeURIComponent(timestamp)}&view=${view}`),
  clipTracks: (matchId: string, clipId: string, timestamp: number) => request<{ timestamp_s: number; field_length: number; field_width: number; tracks: Array<{ track_id: number; x: number; z: number; lifecycle: string; position_kind: string }> }>(`/api/matches/${matchId}/clips/${clipId}/tracks?timestamp_s=${encodeURIComponent(timestamp)}`),
  clipDebugFrames: (matchId: string, clipId: string, trackingMode: 'actual' | 'v28_preview' | 'v8_canonical' = 'actual') => request<{ tracks: Array<{ timestamp_s: number; frame_index?: number; tracks: Array<{ track_id: number; x: number; z: number; lifecycle: string; position_kind: string; measurement_id?: string | null; vx?: number; vz?: number; hits?: number; missed_seconds?: number }> }>; association: Array<{ timestamp_s: number; frame_index: number; observations: Array<{ observation_id: number; view_name: string; xyxy: number[]; confidence: number; ownership_role: string; status: string; source_lenses: Array<{ lens: string; x_min: number; y_min: number; x_max: number; y_max: number }> }>; measurements: Array<{ measurement_id: number; primary_observation_id: number; observation_ids: number[] }> }>; camera_specs: Record<string, { longitude: number; latitude: number; vfov: number }> }>(`/api/matches/${matchId}/clips/${clipId}/debug-frames?tracking_mode=${encodeURIComponent(trackingMode)}`),
  trackingFinal: (matchId: string, clipId: string) => request<TrackingFinalArtifact>(`/api/tracking/matches/${matchId}/clips/${clipId}/final`),
  trackingFinalSummary: (matchId: string, clipId: string) => request<Record<string, unknown>>(`/api/tracking/matches/${matchId}/clips/${clipId}/final/summary`),
  trackingFinalReviews: (matchId: string, clipId: string) => request<TrackingReviews>(`/api/tracking/matches/${matchId}/clips/${clipId}/final/reviews`),
  trackingPersistent: (matchId: string, clipId: string) => request<TrackingPersistentArtifact>(`/api/tracking/matches/${matchId}/clips/${clipId}/persistent`),
  trackingIdentityOverlay: (matchId: string, clipId: string, state = 'latest') => request<TrackingIdentityOverlayArtifact>(`/api/tracking/matches/${matchId}/clips/${clipId}/identity-overlay?state=${encodeURIComponent(state)}`),
  renameMatch: (id: string, name: string) => request<MatchWorkspace>(`/api/matches/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) }),
  deleteMatch: (id: string) => request(`/api/matches/${id}`, { method: 'DELETE' }),
  removeMedia: (id: string) => request(`/api/media/${id}`, { method: 'DELETE' }),
  removeCalibration: (id: string) => request(`/api/matches/${id}/calibration`, { method: 'DELETE' }),
  createMatch: (name: string, description: string, calibrationId: string | null) => request<MatchWorkspace>('/api/matches', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, description, calibration_id: calibrationId }) }),
  importCalibrationPath: (matchId: string, path: string) => request<MatchWorkspace>(`/api/matches/${matchId}/calibration/path`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) }),
  uploadCalibration: (matchId: string, file: File) => { const body = new FormData(); body.append('file', file); return request<MatchWorkspace>(`/api/matches/${matchId}/calibration/upload`, { method: 'POST', body }) },
  importMediaPath: (matchId: string, path: string, role = 'unknown') => request<MediaItem>(`/api/matches/${matchId}/media/path`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path, role }) }),
  uploadMedia: (matchId: string, file: File, role = 'unknown') => { const body = new FormData(); body.append('file', file); body.append('role', role); return request<MediaItem>(`/api/matches/${matchId}/media/upload`, { method: 'POST', body }) },
  extractFrame: (mediaId: string, timestamp: number) => request<MediaItem>('/api/media/frame', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ media_id: mediaId, timestamp_s: timestamp }) }),
  render: (matchId: string, mediaId: string, width: number, height: number) => request<RenderedPanorama>('/api/panorama/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ match_id: matchId, media_id: mediaId, width, height }) }),
  groundPoint: (renderId: string, x: number, y: number, landmarkId: string | null) => request<GroundPoint>('/api/ground-point', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ render_id: renderId, x, y, landmark_id: landmarkId }) }),
  inspectVeo: (matchUrl: string) => request<VeoInspection>('/api/veo/inspect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ match_url: matchUrl }) }),
  importVeo: (inspectionId: string, selectedAssets: Record<string, string>, matchId?: string | null) => request<VeoJob>('/api/veo/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ inspection_id: inspectionId, selected_assets: selectedAssets, match_id: matchId }) }),
  veoJob: (jobId: string) => request<VeoJob>(`/api/veo/import/${jobId}`),
  matchJob: (id: string) => request<VeoJob | null>(`/api/matches/${id}/import-job`),
  cancelVeoJob: (jobId: string) => request<VeoJob>(`/api/veo/import/${jobId}`, { method: 'DELETE' }),
}
