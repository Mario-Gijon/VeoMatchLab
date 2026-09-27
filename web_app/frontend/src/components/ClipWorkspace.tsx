import { useCallback, useEffect, useRef, useState } from 'react'
import { Alert, Box, Button, LinearProgress, Stack, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { api } from '../api/client'
import MatchViewer from './MatchViewer'
import type { InteractiveCamera } from './interactiveCamera'
import { formatMediaTime } from './mediaTime'
import { nearestSample } from './clipDebug'
import type { Clip, ClipJob, MediaItem, ViewContext } from '../types'

type View = 'ultra' | 'interactive'
type Debug = Awaited<ReturnType<typeof api.clipDebugFrames>>
interface Props { matchId: string; clip: Clip; source?: MediaItem; context: ViewContext | null; onBack: () => void; onChanged: () => void }

export default function ClipWorkspace({ matchId, clip, source, context, onBack, onChanged }: Props) {
  const [view, setView] = useState<View>('ultra'), [localTime, setLocalTime] = useState(0), [playing, setPlaying] = useState(false), [debug, setDebug] = useState<Debug | null>(null), [job, setJob] = useState<ClipJob | null>(null), [error, setError] = useState(''), [selectedTrackId, setSelectedTrackId] = useState<number | null>(null)
  const clock = useRef<HTMLVideoElement>(null), ultra = useRef<HTMLVideoElement>(null)
  const interactiveCamera = useRef<InteractiveCamera | null>(null)
  const switchingView = useRef(false)
  const duration = clip.duration_s ?? 0, matchTime = (clip.source_start_s ?? 0) + localTime
  const fieldLength = context?.field_length ?? 105, fieldWidth = context?.field_width ?? 68
  const sample = debug ? nearestSample(debug.tracks, localTime) : null
  const selected = sample?.tracks.find((item) => item.track_id === selectedTrackId)
  const seek = useCallback((value: number) => { const next = Math.max(0, Math.min(duration, value)); setLocalTime(next); if (clock.current) clock.current.currentTime = next; if (ultra.current) ultra.current.currentTime = next }, [duration])
  const start = async (action: 'all' | 'tracking') => { try { setJob(await api.startClipAnalysis(matchId, clip.id, action)) } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo iniciar el análisis.') } }
  useEffect(() => { if (!clip.analysis?.tracking_artifact_id) return; api.clipDebugFrames(matchId, clip.id).then(setDebug).catch((reason: Error) => setError(reason.message)) }, [clip.analysis?.tracking_artifact_id, clip.id, matchId])
  useEffect(() => { if (!job || !['queued', 'running'].includes(job.status)) return; const timer = window.setInterval(() => api.clipJob(job.job_id).then((next) => { setJob(next); if (next.status === 'completed') onChanged() }).catch((reason: Error) => setError(reason.message)), 1000); return () => window.clearInterval(timer) }, [job, onChanged])
  useEffect(() => {
    const presentation = ultra.current
    if (presentation && Math.abs(presentation.currentTime - localTime) > 1 / 15) presentation.currentTime = localTime
  }, [localTime, view])
  useEffect(() => {
    const presentation = ultra.current
    if (!presentation) return
    if (playing) void presentation.play().catch(() => undefined)
    else presentation.pause()
  }, [playing, view])
  useEffect(() => {
    const sourceClock = clock.current
    if (!sourceClock) return
    if (playing) void sourceClock.play().catch(() => undefined)
    else sourceClock.pause()
  }, [playing, view])
  const active = !!job && ['queued', 'running'].includes(job.status)
  const handlePlayback = (value: boolean) => { if (value || !switchingView.current) setPlaying(value) }
  const changeView = (next: View) => {
    const current = clock.current?.currentTime
    if (current !== undefined) setLocalTime(Math.min(duration, current))
    switchingView.current = true
    setView(next)
  }
  useEffect(() => { switchingView.current = false }, [view])
  return <Box sx={{ width: '100%', px: { xs: 1, md: 3 }, py: 2 }}><Button onClick={onBack}>← Clips</Button><Typography variant="h5">{clip.name || 'Clip'}</Typography><Typography color="text.secondary">Clip {formatMediaTime(localTime)} / {formatMediaTime(duration)} · Partido {formatMediaTime(matchTime)}</Typography>
    <ToggleButtonGroup sx={{ mt: 1 }} size="small" exclusive value={view} onChange={(_, value: View | null) => value && changeView(value)}><ToggleButton value="ultra">Ultra panorama</ToggleButton><ToggleButton value="interactive">Interactivo</ToggleButton></ToggleButtonGroup>
    {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
    {view === 'ultra' && <video ref={clock} preload="metadata" src={source ? api.url(source.url) : undefined} style={{ display: 'none' }} onLoadedMetadata={(event) => { event.currentTarget.currentTime = localTime }} onTimeUpdate={() => setLocalTime(Math.min(duration, clock.current?.currentTime ?? 0))} onPlay={() => handlePlayback(true)} onPause={() => handlePlayback(false)} onEnded={() => setPlaying(false)} />}
    <Box sx={{ mt: 1, bgcolor: '#080a0c', minHeight: '52vh', display: 'grid', placeItems: 'center' }}>{view === 'ultra' && (clip.ultra_base_url ? <video ref={ultra} muted playsInline src={api.url(clip.ultra_base_url)} onLoadedMetadata={(event) => { if (Math.abs(event.currentTarget.currentTime - localTime) > 1 / 15) event.currentTarget.currentTime = localTime }} style={{ width: '100%', maxHeight: '66vh', objectFit: 'contain' }} /> : <Alert severity="info">Ejecuta el análisis para generar el ultra panorama.</Alert>)}{view === 'interactive' && (source && context ? <MatchViewer media={source} mode="interactive" crop="stacked" context={context} width={1920} videoRef={clock} initialTime={localTime} cameraRef={interactiveCamera} onTime={(value) => setLocalTime(Math.min(duration, value))} onPlayback={handlePlayback} points={[]} selectionEnabled={false} onClick={() => undefined} /> : <Alert severity="info">{!source ? 'Falta el vídeo físico del clip.' : 'Falta la calibración del partido.'}</Alert>)}</Box>
    <Stack direction="row" gap={1} alignItems="center" mt={1}><Button onClick={() => { clock.current?.pause(); seek(localTime - 10) }}>−10</Button><Button disabled={!source} onClick={() => { if (clock.current?.paused) void clock.current.play(); else clock.current?.pause() }}>{playing ? 'Pausar' : 'Reproducir'}</Button><Button onClick={() => { clock.current?.pause(); seek(localTime + 10) }}>+10</Button><input aria-label="Tiempo del clip" type="range" min="0" max={duration} step=".01" value={localTime} onChange={(event) => seek(Number(event.target.value))} style={{ flex: 1 }} /></Stack>
    <Box sx={{ mt: 2, minHeight: 280, border: 1, borderColor: 'divider', bgcolor: '#2e7d32', position: 'relative', aspectRatio: `${fieldLength} / ${fieldWidth}` }}>{sample?.tracks.map((track) => <Button key={track.track_id} size="small" onClick={() => setSelectedTrackId(track.track_id)} sx={{ position: 'absolute', left: `${Math.max(0, Math.min(100, track.x / fieldLength * 100))}%`, top: `${Math.max(0, Math.min(100, track.z / fieldWidth * 100))}%`, transform: 'translate(-50%, -50%)', minWidth: 0, color: 'white', bgcolor: track.position_kind === 'observed' ? '#1565c0' : '#6a1b9a' }}>{track.track_id}</Button>)}</Box>
    {selected && <Alert severity="info" sx={{ mt: 1 }}>T{selected.track_id} · {selected.position_kind} · {selected.lifecycle} · X {selected.x.toFixed(2)} Z {selected.z.toFixed(2)} · Vx {selected.vx?.toFixed(2) ?? '—'} Vz {selected.vz?.toFixed(2) ?? '—'} · hits {selected.hits ?? '—'} · missed {selected.missed_seconds?.toFixed(2) ?? '—'}</Alert>}
    <Box sx={{ mt: 2, borderTop: 1, borderColor: 'divider', pt: 1 }}><Stack direction="row" gap={1}><Button disabled={active} variant="outlined" onClick={() => void start('all')}>Ejecutar todo</Button><Button disabled={active || !clip.analysis?.detection_artifact_id} onClick={() => void start('tracking')}>Rehacer tracking</Button></Stack>{job && <Box mt={1}><Typography variant="caption">{job.status} · {job.stage} · {job.processed_frames} / {job.total_frames}</Typography><LinearProgress variant="determinate" value={Math.min(100, job.progress * 100)} />{job.status === 'failed' && <Alert severity="error">{job.error || 'Análisis fallido'}</Alert>}</Box>}</Box>
  </Box>
}
