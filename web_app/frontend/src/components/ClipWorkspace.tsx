import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Alert, Box, Button, LinearProgress, Stack, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { api } from '../api/client'
import MatchViewer from './MatchViewer'
import TrackingPitch from './TrackingPitch'
import TrackingOverlay from './trackingOverlay'
import type { InteractiveCamera } from './interactiveCamera'
import { formatMediaTime } from './mediaTime'
import { frameSampleIndex, sampleAtMediaTime } from './clipDebug'
import { startVideoPresentationClock } from './videoPresentationClock'
import type { Clip, ClipJob, MediaItem, ViewContext } from '../types'

type View = 'ultra' | 'interactive'
type Debug = Awaited<ReturnType<typeof api.clipDebugFrames>>
interface Props { matchId: string; clip: Clip; source?: MediaItem; context: ViewContext | null; onBack: () => void; onChanged: () => void }

export default function ClipWorkspace({ matchId, clip, source, context, onBack, onChanged }: Props) {
  const [view, setView] = useState<View>('ultra'), [localTime, setLocalTime] = useState(0), [playing, setPlaying] = useState(false), [debug, setDebug] = useState<Debug | null>(null), [job, setJob] = useState<ClipJob | null>(null), [error, setError] = useState(''), [ultraError, setUltraError] = useState(''), [selectedTrackId, setSelectedTrackId] = useState<number | null>(null)
  const clock = useRef<HTMLVideoElement>(null), ultra = useRef<HTMLVideoElement>(null)
  const interactiveCamera = useRef<InteractiveCamera | null>(null)
  const switchingView = useRef(false)
  const playingRef = useRef(false)
  const duration = clip.duration_s ?? 0, matchTime = (clip.source_start_s ?? 0) + localTime
  const fieldLength = context?.field_length ?? 105, fieldWidth = context?.field_width ?? 68
  const tracksByFrame = useMemo(() => frameSampleIndex(debug?.tracks ?? []), [debug])
  const sample = debug ? sampleAtMediaTime(debug.tracks, tracksByFrame, localTime, source?.fps) : null
  const selected = sample?.tracks.find((item) => item.track_id === selectedTrackId)
  const presentTime = useCallback((value: number) => {
    const next = Math.max(0, Math.min(duration, value))
    setLocalTime((current) => Math.abs(current - next) < 1e-9 ? current : next)
  }, [duration])
  const masterVideo = useCallback(() => view === 'ultra' ? ultra.current : clock.current, [view])
  const seek = useCallback((value: number) => { const next = Math.max(0, Math.min(duration, value)); presentTime(next); if (clock.current) clock.current.currentTime = next; if (ultra.current) ultra.current.currentTime = next }, [duration, presentTime])
  const start = async (action: 'all' | 'tracking') => { try { setJob(await api.startClipAnalysis(matchId, clip.id, action)) } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo iniciar el análisis.') } }
  useEffect(() => { if (!clip.analysis?.tracking_artifact_id) return; api.clipDebugFrames(matchId, clip.id).then(setDebug).catch((reason: Error) => setError(reason.message)) }, [clip.analysis?.tracking_artifact_id, clip.id, matchId])
  useEffect(() => { if (!job || !['queued', 'running'].includes(job.status)) return; const timer = window.setInterval(() => api.clipJob(job.job_id).then((next) => { setJob(next); if (next.status === 'completed') onChanged() }).catch((reason: Error) => setError(reason.message)), 1000); return () => window.clearInterval(timer) }, [job, onChanged])
  useEffect(() => { playingRef.current = playing }, [playing])
  useEffect(() => {
    const presentation = masterVideo()
    return presentation ? startVideoPresentationClock(presentation, presentTime) : undefined
  }, [masterVideo, presentTime, source?.id, clip.ultra_base_url, ultraError, view])
  useEffect(() => {
    const presentation = ultra.current
    if (presentation && Math.abs(presentation.currentTime - localTime) > 1 / 30) presentation.currentTime = localTime
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
  const handlePlayback = useCallback((kind: View, value: boolean) => {
    if (kind === view && (value || !switchingView.current)) setPlaying(value)
  }, [view])
  const coarseTime = useCallback((value: number) => { if (!playingRef.current) presentTime(value) }, [presentTime])
  const changeView = (next: View) => {
    const current = masterVideo()?.currentTime
    if (current !== undefined) setLocalTime(Math.min(duration, current))
    switchingView.current = true
    setView(next)
  }
  useEffect(() => { switchingView.current = false }, [view])
  return <Box sx={{ width: '100%', px: { xs: 1, md: 3 }, py: 2 }}><Button onClick={onBack}>← Clips</Button><Typography variant="h5">{clip.name || 'Clip'}</Typography><Typography color="text.secondary">Clip {formatMediaTime(localTime)} / {formatMediaTime(duration)} · Partido {formatMediaTime(matchTime)}</Typography>
    <ToggleButtonGroup sx={{ mt: 1 }} size="small" exclusive value={view} onChange={(_, value: View | null) => value && changeView(value)}><ToggleButton value="ultra">Ultra panorama</ToggleButton><ToggleButton value="interactive">Interactivo</ToggleButton></ToggleButtonGroup>
    {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
    {view === 'ultra' && <video ref={clock} preload="metadata" src={source ? api.url(source.url) : undefined} style={{ display: 'none' }} onLoadedMetadata={(event) => { event.currentTarget.currentTime = localTime }} onSeeked={(event) => coarseTime(event.currentTarget.currentTime)} onTimeUpdate={(event) => coarseTime(event.currentTarget.currentTime)} onPlay={() => handlePlayback('interactive', true)} onPause={() => handlePlayback('interactive', false)} onEnded={() => setPlaying(false)} />}
    <Box sx={{ mt: 1, display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 3fr) minmax(300px, 1fr)' }, gap: { xs: 1.5, lg: 2 }, alignItems: 'stretch' }}>
      <Box sx={{ minWidth: 0, height: '100%', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', '& > div': { width: '100%' }, '& canvas': { width: '100% !important', maxWidth: '100%' } }}>{view === 'ultra' && (clip.ultra_base_url ? ultraError ? <Alert severity="error">No se puede reproducir el Ultra panorama en este navegador. {ultraError}</Alert> : <video ref={ultra} muted playsInline src={api.url(clip.ultra_base_url)} onLoadedMetadata={(event) => { if (Math.abs(event.currentTarget.currentTime - localTime) > 1 / 30) event.currentTarget.currentTime = localTime }} onSeeked={(event) => coarseTime(event.currentTarget.currentTime)} onTimeUpdate={(event) => coarseTime(event.currentTarget.currentTime)} onPlay={() => handlePlayback('ultra', true)} onPause={() => handlePlayback('ultra', false)} onEnded={() => setPlaying(false)} onError={() => setUltraError('El archivo no tiene un códec de vídeo compatible.') } style={{ width: '100%', maxHeight: '100%', objectFit: 'contain' }} /> : <Alert severity="info">Ejecuta el análisis para generar el ultra panorama.</Alert>)}{view === 'interactive' && (source && context ? <MatchViewer media={source} mode="interactive" crop="stacked" context={context} width={1920} videoRef={clock} initialTime={localTime} cameraRef={interactiveCamera} onTime={coarseTime} onPlayback={(value) => handlePlayback('interactive', value)} points={[]} selectionEnabled={false} onClick={() => undefined} /> : <Alert severity="info">{!source ? 'Falta el vídeo físico del clip.' : 'Falta la calibración del partido.'}</Alert>)}{context && <TrackingOverlay mode={view} context={context} cameraRef={interactiveCamera} tracks={sample?.tracks ?? []} association={debug?.association ?? []} cameraSpecs={debug?.camera_specs ?? {}} />}</Box>
      <Box sx={{ minWidth: 0, minHeight: { xs: 220, lg: 0 }, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Box sx={{ width: '100%', maxHeight: '100%' }}><TrackingPitch fieldLength={fieldLength} fieldWidth={fieldWidth} tracks={sample?.tracks ?? []} onTrackSelect={setSelectedTrackId} /></Box></Box>
    </Box>
    <Stack direction="row" gap={1} alignItems="center" mt={1}><Button onClick={() => { masterVideo()?.pause(); seek(localTime - 10) }}>−10</Button><Button disabled={!source} onClick={() => { const master = masterVideo(); if (master?.paused) void master.play(); else master?.pause() }}>{playing ? 'Pausar' : 'Reproducir'}</Button><Button onClick={() => { masterVideo()?.pause(); seek(localTime + 10) }}>+10</Button><input aria-label="Tiempo del clip" type="range" min="0" max={duration} step=".01" value={localTime} onChange={(event) => seek(Number(event.target.value))} style={{ flex: 1 }} /><Typography variant="body2" sx={{ whiteSpace: 'nowrap' }}>{formatMediaTime(localTime)} · {formatMediaTime(matchTime)}</Typography></Stack>
    {selected && <Alert severity="info" sx={{ mt: 1 }}>T{selected.track_id} · {selected.position_kind} · {selected.lifecycle} · X {selected.x.toFixed(2)} Z {selected.z.toFixed(2)} · Vx {selected.vx?.toFixed(2) ?? '—'} Vz {selected.vz?.toFixed(2) ?? '—'} · hits {selected.hits ?? '—'} · missed {selected.missed_seconds?.toFixed(2) ?? '—'}</Alert>}
    <Box sx={{ mt: 1.5, borderTop: 1, borderColor: 'divider', pt: 1 }}><Stack direction="row" gap={1} alignItems="center" flexWrap="wrap"><Button disabled={active} variant="outlined" onClick={() => void start('all')}>Ejecutar todo</Button><Button disabled={active || !clip.analysis?.detection_artifact_id} onClick={() => void start('tracking')}>Rehacer tracking</Button>{job && <Typography variant="caption">{job.status} · {job.stage} · {job.processed_frames} / {job.total_frames}</Typography>}</Stack>{job && <LinearProgress sx={{ mt: 0.5 }} variant="determinate" value={Math.min(100, job.progress * 100)} />}{job?.status === 'failed' && <Alert severity="error" sx={{ mt: 0.5 }}>{job.error || 'Análisis fallido'}</Alert>}</Box>
  </Box>
}
