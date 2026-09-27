import type { Calibration, Clip, ClipJob, GroundPoint, MatchWorkspace, MediaItem, RenderedPanorama, VeoInspection, VeoJob, ViewContext } from '../types'

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
  startClipAnalysis: (matchId: string, clipId: string, action: 'all' | 'tracking') => request<ClipJob>(`/api/matches/${matchId}/clips/${clipId}/analysis`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }) }),
  clipJob: (jobId: string) => request<ClipJob>(`/api/clip-jobs/${jobId}`),
  cancelClipJob: (jobId: string) => request<ClipJob>(`/api/clip-jobs/${jobId}`, { method: 'DELETE' }),
  deleteClip: (matchId: string, clipId: string) => request<{ deleted: boolean }>(`/api/matches/${matchId}/clips/${clipId}`, { method: 'DELETE' }),
  clipViews: (matchId: string, clipId: string, timestamp: number) => request<{ timestamp_s: number; cameras: Record<string, string>; ultra_panorama: string }>(`/api/matches/${matchId}/clips/${clipId}/development-views?timestamp_s=${encodeURIComponent(timestamp)}`),
  clipTracks: (matchId: string, clipId: string, timestamp: number) => request<{ timestamp_s: number; field_length: number; field_width: number; tracks: Array<{ track_id: number; x: number; z: number; lifecycle: string; position_kind: string }> }>(`/api/matches/${matchId}/clips/${clipId}/tracks?timestamp_s=${encodeURIComponent(timestamp)}`),
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
