import { useEffect, useRef, useState } from 'react'
import { Alert, Box, Button, LinearProgress, Stack, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { api } from '../api/client'
import { formatMediaTime } from './mediaTime'
import type { Clip, ClipJob, MediaItem } from '../types'

type View = 'source' | 'cameras' | 'ultra' | 'tracking' | 'field'
interface Tracks { field_length: number; field_width: number; tracks: Array<{ track_id: number; x: number; z: number; lifecycle: string; position_kind: string }> }
interface Props { matchId: string; clip: Clip; source?: MediaItem; onBack: () => void; onChanged: () => void }

export default function ClipWorkspace({ matchId, clip, source, onBack, onChanged }: Props) {
  const [view, setView] = useState<View>('source')
  const [localTime, setLocalTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [renders, setRenders] = useState<{ cameras: Record<string, string>; ultra_panorama: string } | null>(null)
  const [tracks, setTracks] = useState<Tracks | null>(null)
  const [job, setJob] = useState<ClipJob | null>(null)
  const [error, setError] = useState('')
  const video = useRef<HTMLVideoElement>(null)
  const duration = clip.duration_s ?? 0
  const absoluteTime = (clip.source_start_s ?? 0) + localTime
  const seek = (value: number) => { const next = Math.max(0, Math.min(duration, value)); setLocalTime(next); if (video.current) video.current.currentTime = next }
  const startJob = async (action: 'all' | 'tracking') => { try { setJob(await api.startClipAnalysis(matchId, clip.id, action)) } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo iniciar el análisis.') } }

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return
    const timer = window.setInterval(() => api.clipJob(job.job_id).then((next) => { setJob(next); if (next.status === 'completed') onChanged() }).catch((reason: Error) => setError(reason.message)), 1000)
    return () => window.clearInterval(timer)
  }, [job, onChanged])
  useEffect(() => {
    if (view !== 'cameras' && view !== 'ultra') return
    const timer = window.setTimeout(() => api.clipViews(matchId, clip.id, localTime).then((next) => setRenders(next)).catch((reason: Error) => setError(reason.message)), 180)
    return () => window.clearTimeout(timer)
  }, [clip.id, localTime, matchId, view])
  useEffect(() => {
    if (view !== 'field' || !clip.analysis?.tracking_artifact_id) return
    const timer = window.setTimeout(() => api.clipTracks(matchId, clip.id, localTime).then(setTracks).catch((reason: Error) => setError(reason.message)), 100)
    return () => window.clearTimeout(timer)
  }, [clip.analysis?.tracking_artifact_id, clip.id, localTime, matchId, view])

  const cameraNames = ['tight_far_c1', 'tight_far_c2', 'tight_far_c3', 'tight_far_c4', 'tight_near_c1', 'tight_near_c2']
  return <Box sx={{ maxWidth: 1280, mx: 'auto', py: 2 }}>
    <Button onClick={onBack}>← Clips</Button><Typography variant="h5">{clip.name || 'Clip'}</Typography><Typography color="text.secondary">{formatMediaTime(clip.source_start_s)} → {formatMediaTime((clip.source_start_s ?? 0) + duration)} · {formatMediaTime(duration)}</Typography>
    <ToggleButtonGroup sx={{ mt: 2 }} size="small" exclusive value={view} onChange={(_, next: View | null) => next && setView(next)}><ToggleButton value="source">Fuente</ToggleButton><ToggleButton value="cameras">6 cámaras</ToggleButton><ToggleButton value="ultra">Ultra panorama</ToggleButton><ToggleButton value="tracking">Tracking</ToggleButton><ToggleButton value="field">Campo 2D</ToggleButton></ToggleButtonGroup>
    {error && <Alert severity="error" sx={{ mt: 1 }} onClose={() => setError('')}>{error}</Alert>}
    <Box sx={{ mt: 1, background: '#080a0c', minHeight: 320, display: 'grid', placeItems: 'center' }}>
      {view === 'source' && source && <video ref={video} controls preload="metadata" src={api.url(source.url)} style={{ width: '100%', maxHeight: '64vh' }} onLoadedMetadata={() => seek(localTime)} onTimeUpdate={() => setLocalTime(Math.min(duration, Math.max(0, video.current?.currentTime ?? 0)))} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => { setPlaying(false); setLocalTime(duration) }} onError={() => setError('No se puede reproducir source.mp4.')} />}
      {view === 'source' && !source && <Alert severity="info">Clip antiguo — archivo no generado.</Alert>}
      {view === 'cameras' && <Box sx={{ width: '100%', p: 1, display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 1 }}>{cameraNames.map((name) => <Box key={name}><Typography variant="caption" color="common.white">{name}</Typography>{renders?.cameras[name] ? <img src={renders.cameras[name]} alt={name} style={{ width: '100%', display: 'block' }} /> : <LinearProgress />}</Box>)}</Box>}
      {view === 'ultra' && (renders?.ultra_panorama ? <img src={renders.ultra_panorama} alt="Ultra panorama" style={{ width: '100%', maxHeight: '64vh', objectFit: 'contain' }} /> : <LinearProgress sx={{ width: '80%' }} />)}
      {view === 'tracking' && (clip.debug_video_url ? <video controls src={api.url(clip.debug_video_url)} style={{ width: '100%', maxHeight: '64vh' }} /> : <Alert severity="info">Ejecuta el análisis para generar el vídeo de tracking.</Alert>)}
      {view === 'field' && (tracks ? <Box sx={{ width: '90%', aspectRatio: `${tracks.field_length} / ${tracks.field_width}`, maxHeight: '60vh', position: 'relative', border: '2px solid #eee', background: '#2e7d32' }}>{tracks.tracks.map((track) => <Box key={track.track_id} sx={{ position: 'absolute', left: `${Math.max(0, Math.min(100, track.x / tracks.field_length * 100))}%`, top: `${Math.max(0, Math.min(100, track.z / tracks.field_width * 100))}%`, transform: 'translate(-50%, -50%)', color: 'white', fontSize: 12, border: '1px solid white', borderRadius: '50%', px: .5, py: .25, background: track.position_kind === 'observed' ? '#1565c0' : '#6a1b9a', opacity: track.lifecycle === 'tentative' ? .55 : 1 }}>{track.track_id}<br />{track.position_kind[0].toUpperCase()}/{track.lifecycle[0].toUpperCase()}</Box>)}</Box> : <Alert severity="info">No hay tracks disponibles para este instante.</Alert>)}
    </Box>
    <Stack direction="row" gap={1} mt={1} alignItems="center" flexWrap="wrap"><Button onClick={() => { video.current?.pause(); seek(localTime - 10) }}>−10 s</Button><Button onClick={() => { if (!video.current) return; if (video.current.paused) void video.current.play(); else video.current.pause() }}>{playing ? 'Pausar' : 'Reproducir'}</Button><Button onClick={() => { video.current?.pause(); seek(localTime + 10) }}>+10 s</Button><input aria-label="Tiempo del clip" type="range" min="0" max={duration} step="0.05" value={localTime} onChange={(event) => seek(Number(event.target.value))} style={{ flex: 1, minWidth: 180 }} /><Typography sx={{ fontVariantNumeric: 'tabular-nums' }}>{formatMediaTime(localTime)} / {formatMediaTime(duration)} · partido {formatMediaTime(absoluteTime)}</Typography></Stack>
    <Box sx={{ mt: 2, borderTop: 1, borderColor: 'divider', pt: 1 }}><Typography variant="subtitle1">Análisis</Typography><Typography variant="body2">Detection: {clip.analysis?.detection_artifact_id ? 'completed' : 'pending'} · Tracking: {clip.analysis?.tracking_artifact_id ? 'completed' : 'pending'}</Typography><Stack direction="row" gap={1} mt={1}><Button variant="outlined" onClick={() => void startJob('all')}>Ejecutar todo</Button><Button disabled={!clip.analysis?.detection_artifact_id} onClick={() => void startJob('tracking')}>Rehacer tracking</Button></Stack>{job && <Box mt={1}><Typography variant="caption">{job.status} · {job.stage} · {job.processed_frames} / {job.total_frames} frames</Typography><LinearProgress variant="determinate" value={Math.min(100, job.progress * 100)} /></Box>}</Box>
  </Box>
}
