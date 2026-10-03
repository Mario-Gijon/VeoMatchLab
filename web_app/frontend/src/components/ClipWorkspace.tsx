import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Alert, Box, Button, Checkbox, FormControlLabel, LinearProgress, Slider, Stack, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { api } from '../api/client'
import MatchViewer from './MatchViewer'
import TrackingPitch from './TrackingPitch'
import TrackingOverlay from './trackingOverlay'
import type { InteractiveCamera } from './interactiveCamera'
import { formatMediaTime } from './mediaTime'
import { frameSampleIndex, latestSampleAtMediaTime, sampleAtMediaTime } from './clipDebug'
import { startVideoPresentationClock } from './videoPresentationClock'
import type { Clip, ClipJob, MediaItem, ViewContext } from '../types'
import TrackingIdentityOverlay from './TrackingIdentityOverlay'
import type { TrackingIdentityOverlayArtifact, TrackingIdentityOverlaySource } from '../types/trackingIdentityOverlay'
import FixedRosterPitch from './FixedRosterPitch'
import FixedRosterTrackingOverlay from './FixedRosterTrackingOverlay'
import { activeFixedRosterObservations } from './trackingIdentityOverlayData'
import type { ActorTrajectoryArtifact } from '../types/actorTrajectory'
import { activeActorTrajectoryPoints } from './actorTrajectoryData'
import ActorTrajectoryPitch from './ActorTrajectoryPitch'
import ActorTrajectoryOverlay from './ActorTrajectoryOverlay'
import ActorTrajectorySourceOverlay from './ActorTrajectorySourceOverlay'

type View = 'ultra' | 'interactive' | 'source'
type TrackingMode = 'actual' | 'v28_preview' | 'v8_canonical'
type TrackingDisplayMode = TrackingMode | 'fixed_roster' | 'actor_trajectories'
type Debug = Awaited<ReturnType<typeof api.clipDebugFrames>>
interface Props { matchId: string; clip: Clip; source?: MediaItem; context: ViewContext | null; onBack: () => void; onChanged: () => void }

export default function ClipWorkspace({ matchId, clip, source, context, onBack, onChanged }: Props) {
  const [view, setView] = useState<View>('ultra'), [localTime, setLocalTime] = useState(0), [playing, setPlaying] = useState(false), [actualDebug, setActualDebug] = useState<Debug | null>(null), [v28PreviewDebug, setV28PreviewDebug] = useState<Debug | null>(null), [v8CanonicalDebug, setV8CanonicalDebug] = useState<Debug | null>(null), [trackingDisplayMode, setTrackingDisplayMode] = useState<TrackingDisplayMode>('actual'), [detectionThreshold, setDetectionThreshold] = useState(0.40), [job, setJob] = useState<ClipJob | null>(null), [error, setError] = useState(''), [previewError, setPreviewError] = useState(''), [ultraError, setUltraError] = useState(''), [selectedTrackId, setSelectedTrackId] = useState<number | null>(null), [selectedFixedTracklet, setSelectedFixedTracklet] = useState<string | null>(null), [showIdentityOverlay, setShowIdentityOverlay] = useState(false), [identityOverlaySource, setIdentityOverlaySource] = useState<TrackingIdentityOverlaySource>('fixed_roster'), [identityOverlay, setIdentityOverlay] = useState<TrackingIdentityOverlayArtifact | null>(null), [identityOverlayError, setIdentityOverlayError] = useState(''), [actorTrajectory, setActorTrajectory] = useState<ActorTrajectoryArtifact | null>(null), [actorTrajectoryError, setActorTrajectoryError] = useState(''), [showTrackletId, setShowTrackletId] = useState(false), [showLens, setShowLens] = useState(false), [markInterpolation, setMarkInterpolation] = useState(false)
  const clock = useRef<HTMLVideoElement>(null), ultra = useRef<HTMLVideoElement>(null)
  const interactiveCamera = useRef<InteractiveCamera | null>(null)
  const switchingView = useRef(false)
  const playingRef = useRef(false)
  const duration = clip.duration_s ?? 0, matchTime = (clip.source_start_s ?? 0) + localTime
  const fieldLength = context?.field_length ?? 105, fieldWidth = context?.field_width ?? 68
  const actualTracksByFrame = useMemo(() => frameSampleIndex(actualDebug?.tracks ?? []), [actualDebug])
  const v28PreviewTracksByFrame = useMemo(() => frameSampleIndex(v28PreviewDebug?.tracks ?? []), [v28PreviewDebug])
  const v8CanonicalTracksByFrame = useMemo(() => frameSampleIndex(v8CanonicalDebug?.tracks ?? []), [v8CanonicalDebug])
  const fixedRosterSelected = trackingDisplayMode === 'fixed_roster'
  const actorTrajectorySelected = trackingDisplayMode === 'actor_trajectories'
  const requestedOverlaySource: TrackingIdentityOverlaySource = fixedRosterSelected ? 'fixed_roster' : identityOverlaySource
  const activeDebug = fixedRosterSelected || actorTrajectorySelected ? null : trackingDisplayMode === 'actual' ? actualDebug : trackingDisplayMode === 'v28_preview' ? v28PreviewDebug : v8CanonicalDebug
  const activeTracksByFrame = trackingDisplayMode === 'actual' ? actualTracksByFrame : trackingDisplayMode === 'v28_preview' ? v28PreviewTracksByFrame : v8CanonicalTracksByFrame
  const activeSample = activeDebug ? trackingDisplayMode === 'v8_canonical' ? latestSampleAtMediaTime(activeDebug.tracks, localTime) : sampleAtMediaTime(activeDebug.tracks, activeTracksByFrame, localTime, source?.fps) : null
  const fixedRosterArtifact = identityOverlay?.source_kind === 'fixed_roster' ? identityOverlay : null
  const activeFixedRoster = useMemo(() => fixedRosterArtifact ? activeFixedRosterObservations(fixedRosterArtifact.tracks, localTime, source?.fps) : [], [fixedRosterArtifact, localTime, source?.fps])
  const activeActorTrajectory = useMemo(() => actorTrajectory ? activeActorTrajectoryPoints(actorTrajectory.actors, localTime, source?.fps ?? actorTrajectory.estimated_fps) : [], [actorTrajectory, localTime, source?.fps])
  const selectedFixed = activeFixedRoster.find((item) => item.trackletId === selectedFixedTracklet)
  const selected = activeSample?.tracks.find((item) => item.track_id === selectedTrackId)
  const presentTime = useCallback((value: number) => {
    const next = Math.max(0, Math.min(duration, value))
    setLocalTime((current) => Math.abs(current - next) < 1e-9 ? current : next)
  }, [duration])
  const masterVideo = useCallback(() => view === 'ultra' ? ultra.current : clock.current, [view])
  const seek = useCallback((value: number) => { const next = Math.max(0, Math.min(duration, value)); presentTime(next); if (clock.current) clock.current.currentTime = next; if (ultra.current) ultra.current.currentTime = next }, [duration, presentTime])
  const start = async (action: 'all' | 'tracking') => { try { setJob(await api.startClipAnalysis(matchId, clip.id, action, trackingDisplayMode === 'v8_canonical' ? 'v8_canonical' : 'baseline', detectionThreshold)) } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo iniciar el análisis.') } }
  useEffect(() => { if (!clip.analysis?.tracking_artifact_id) return; api.clipDebugFrames(matchId, clip.id).then(setActualDebug).catch((reason: Error) => setError(reason.message)) }, [clip.analysis?.tracking_artifact_id, clip.id, matchId])
  useEffect(() => {
    if (trackingDisplayMode === 'actual' || trackingDisplayMode === 'fixed_roster' || trackingDisplayMode === 'actor_trajectories') return
    setPreviewError('')
    const mode = trackingDisplayMode
    if (mode === 'v28_preview') setV28PreviewDebug(null)
    else setV8CanonicalDebug(null)
    api.clipDebugFrames(matchId, clip.id, mode).then((debug) => {
      if (mode === 'v28_preview') setV28PreviewDebug(debug)
      else setV8CanonicalDebug(debug)
    }).catch((reason: Error) => {
      if (mode === 'v28_preview') setV28PreviewDebug(null)
      else setV8CanonicalDebug(null)
      setPreviewError(reason.message)
    })
  }, [clip.id, matchId, trackingDisplayMode])
  useEffect(() => {
    if (!actorTrajectorySelected || actorTrajectory || actorTrajectoryError) return
    let cancelled = false
    api.actorTrajectories(matchId, clip.id).then((artifact) => { if (!cancelled) setActorTrajectory(artifact) }).catch((reason: Error) => { if (!cancelled) setActorTrajectoryError(reason.message) })
    return () => { cancelled = true }
  }, [actorTrajectory, actorTrajectoryError, actorTrajectorySelected, clip.id, matchId])
  useEffect(() => { if (!job || !['queued', 'running'].includes(job.status)) return; const timer = window.setInterval(() => api.clipJob(job.job_id).then((next) => { setJob(next); if (next.status === 'completed') onChanged() }).catch((reason: Error) => setError(reason.message)), 1000); return () => window.clearInterval(timer) }, [job, onChanged])
  useEffect(() => {
    if (!(showIdentityOverlay || fixedRosterSelected) || identityOverlay?.source_kind === requestedOverlaySource || identityOverlayError) return
    let cancelled = false
    api.trackingIdentityOverlay(matchId, clip.id, requestedOverlaySource).then((artifact) => { if (!cancelled) setIdentityOverlay(artifact) }).catch((reason: Error) => { if (!cancelled) setIdentityOverlayError(reason.message) })
    return () => { cancelled = true }
  }, [clip.id, fixedRosterSelected, identityOverlay, identityOverlayError, matchId, requestedOverlaySource, showIdentityOverlay])
  useEffect(() => { setIdentityOverlay(null); setIdentityOverlayError('') }, [clip.id, identityOverlaySource, matchId, requestedOverlaySource])
  useEffect(() => { setActorTrajectory(null); setActorTrajectoryError('') }, [clip.id, matchId])
  useEffect(() => { setIdentityOverlay(null); setIdentityOverlayError(''); setShowIdentityOverlay(false) }, [clip.id, matchId])
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
  const v8WaitingForRealProgress = !!job && job.tracking_mode === 'v8_canonical' && active && job.processed_frames === 0
  const v8Stage: Record<string, string> = { 'v8-starting': 'Preparando V8', 'v8-crops': 'Detectando y generando crops', 'v8-reid': 'Calculando ReID', 'v8-identities': 'Resolviendo identidades', 'v8-publication': 'Generando resultado V8', 'v8-completed': 'Completado' }
  const jobLabel = job?.tracking_mode === 'v8_canonical' ? `Procesando V8 · ${v8Stage[job.stage] ?? job.stage}` : job ? `${job.status} · ${job.stage} · ${job.processed_frames} / ${job.total_frames}` : ''
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
    <ToggleButtonGroup sx={{ mt: 1 }} size="small" exclusive value={view} onChange={(_, value: View | null) => value && changeView(value)}><ToggleButton value="ultra">Ultra panorama</ToggleButton><ToggleButton value="interactive">Interactivo</ToggleButton><ToggleButton value="source">Doble cámara</ToggleButton></ToggleButtonGroup>
    {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
    {view === 'ultra' && <video ref={clock} preload="metadata" src={source ? api.url(source.url) : undefined} style={{ display: 'none' }} onLoadedMetadata={(event) => { event.currentTarget.currentTime = localTime }} onSeeked={(event) => coarseTime(event.currentTarget.currentTime)} onTimeUpdate={(event) => coarseTime(event.currentTarget.currentTime)} onPlay={() => handlePlayback('interactive', true)} onPause={() => handlePlayback('interactive', false)} onEnded={() => setPlaying(false)} />}
    <Box sx={{ mt: 1, display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 3fr) minmax(300px, 1fr)' }, gap: { xs: 1.5, lg: 2 }, alignItems: 'stretch' }}>
      <Box sx={{ minWidth: 0, height: '100%', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', '& > div': { width: '100%' }, '& canvas': { width: '100% !important', maxWidth: '100%' } }}>{view === 'ultra' && (clip.ultra_base_url ? ultraError ? <Alert severity="error">No se puede reproducir el Ultra panorama en este navegador. {ultraError}</Alert> : <video ref={ultra} muted playsInline src={api.url(clip.ultra_base_url)} onLoadedMetadata={(event) => { if (Math.abs(event.currentTarget.currentTime - localTime) > 1 / 30) event.currentTarget.currentTime = localTime }} onSeeked={(event) => coarseTime(event.currentTarget.currentTime)} onTimeUpdate={(event) => coarseTime(event.currentTarget.currentTime)} onPlay={() => handlePlayback('ultra', true)} onPause={() => handlePlayback('ultra', false)} onEnded={() => setPlaying(false)} onError={() => setUltraError('El archivo no tiene un códec de vídeo compatible.') } style={{ width: '100%', maxHeight: '100%', objectFit: 'contain' }} /> : <Alert severity="info">Ejecuta el análisis para generar el ultra panorama.</Alert>)}{view === 'interactive' && (source && context ? <MatchViewer media={source} mode="interactive" crop="stacked" context={context} width={1920} videoRef={clock} initialTime={localTime} cameraRef={interactiveCamera} onTime={coarseTime} onPlayback={(value) => handlePlayback('interactive', value)} points={[]} selectionEnabled={false} onClick={() => undefined} /> : <Alert severity="info">{!source ? 'Falta el vídeo físico del clip.' : 'Falta la calibración del partido.'}</Alert>)}{view === 'source' && (source ? <MatchViewer media={source} mode="dual_lens" crop="stacked" context={context} width={1920} videoRef={clock} initialTime={localTime} cameraRef={interactiveCamera} onTime={coarseTime} onPlayback={(value) => handlePlayback('source', value)} points={[]} selectionEnabled={false} onClick={() => undefined} overlay={actorTrajectorySelected && actorTrajectory ? <ActorTrajectorySourceOverlay artifact={actorTrajectory} time={localTime} fps={source.fps} /> : (showIdentityOverlay || fixedRosterSelected) && identityOverlay?.source_kind === requestedOverlaySource ? <TrackingIdentityOverlay artifact={identityOverlay} time={localTime} fps={source.fps} showTrackletId={showTrackletId} showLens={showLens} markInterpolation={markInterpolation} /> : undefined} /> : <Alert severity="info">Falta el vídeo físico del clip.</Alert>)}{context && fixedRosterSelected && fixedRosterArtifact && (view === 'ultra' || view === 'interactive') && <FixedRosterTrackingOverlay observations={activeFixedRoster} mode={view} context={context} cameraRef={interactiveCamera} cameraSpecs={actualDebug?.camera_specs ?? {}} />}{context && actorTrajectorySelected && actorTrajectory && (view === 'ultra' || view === 'interactive') && <ActorTrajectoryOverlay points={activeActorTrajectory} mode={view} context={context} cameraRef={interactiveCamera} cameraSpecs={actualDebug?.camera_specs ?? {}} />}{context && !fixedRosterSelected && !actorTrajectorySelected && (view === 'ultra' || view === 'interactive') && <TrackingOverlay mode={view} context={context} cameraRef={interactiveCamera} tracks={activeSample?.tracks ?? []} frameIndex={activeSample?.frame_index} timestamp={activeSample?.timestamp_s} trackFrames={activeDebug?.tracks ?? []} association={activeDebug?.association ?? []} cameraSpecs={activeDebug?.camera_specs ?? {}} />}</Box>
      <Box sx={{ minWidth: 0, minHeight: { xs: 220, lg: 0 }, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}><ToggleButtonGroup size="small" exclusive value={trackingDisplayMode} onChange={(_, value: TrackingDisplayMode | null) => { if (value) { setTrackingDisplayMode(value); setSelectedTrackId(null); setSelectedFixedTracklet(null); setActorTrajectoryError('') } }}><ToggleButton value="actual">Baseline</ToggleButton><ToggleButton value="v28_preview">V2.8 Preview</ToggleButton><ToggleButton value="v8_canonical">V8 Canonical</ToggleButton><ToggleButton value="fixed_roster">Fixed roster 7I.4</ToggleButton><ToggleButton value="actor_trajectories">Actor trajectories 7I.7</ToggleButton></ToggleButtonGroup>{!fixedRosterSelected && !actorTrajectorySelected && trackingDisplayMode !== 'actual' && previewError && <Alert severity="error" sx={{ mt: 0.5, width: '100%' }}>{previewError}</Alert>}{fixedRosterSelected && fixedRosterArtifact && <Typography variant="caption" sx={{ mt: 0.5 }}>Fixed roster 7I.4 · {fixedRosterArtifact.identity_count} actores · {fixedRosterArtifact.assigned_tracklet_count} asignados · {fixedRosterArtifact.unassigned_tracklet_count} pendientes</Typography>}{fixedRosterSelected && identityOverlayError && <Alert severity="info" sx={{ mt: 0.5 }}>Fixed roster Tracking V1 no disponible</Alert>}{actorTrajectorySelected && actorTrajectory && <Typography variant="caption" sx={{ mt: 0.5 }}>Actor trajectories 7I.7 · {actorTrajectory.actor_count} actores · {actorTrajectory.display_point_count.toLocaleString('es-ES')} puntos · {actorTrajectory.interpolated_point_count.toLocaleString('es-ES')} interpolados · {actorTrajectory.estimated_fps?.toFixed(2) ?? '—'} fps · {actorTrajectory.segment_count} segmentos</Typography>}{actorTrajectorySelected && actorTrajectoryError && <Alert severity="info" sx={{ mt: 0.5 }}>Actor trajectories 7I.7 no disponible</Alert>}<Box sx={{ width: '100%', maxHeight: '100%' }}>{actorTrajectorySelected ? <ActorTrajectoryPitch fieldLength={fieldLength} fieldWidth={fieldWidth} points={activeActorTrajectory} /> : fixedRosterSelected ? <FixedRosterPitch fieldLength={fieldLength} fieldWidth={fieldWidth} observations={activeFixedRoster} onTrackletSelect={setSelectedFixedTracklet} /> : <TrackingPitch fieldLength={fieldLength} fieldWidth={fieldWidth} tracks={activeSample?.tracks ?? []} onTrackSelect={setSelectedTrackId} />}</Box></Box>
    </Box>
    <Stack direction="row" gap={1} alignItems="center" mt={1}><Button onClick={() => { masterVideo()?.pause(); seek(localTime - 10) }}>−10</Button><Button disabled={!source} onClick={() => { const master = masterVideo(); if (master?.paused) void master.play(); else master?.pause() }}>{playing ? 'Pausar' : 'Reproducir'}</Button><Button onClick={() => { masterVideo()?.pause(); seek(localTime + 10) }}>+10</Button><input aria-label="Tiempo del clip" type="range" min="0" max={duration} step=".01" value={localTime} onChange={(event) => seek(Number(event.target.value))} style={{ flex: 1 }} /><Typography variant="body2" sx={{ whiteSpace: 'nowrap' }}>{formatMediaTime(localTime)} · {formatMediaTime(matchTime)}</Typography></Stack>
    {view === 'source' && <Box sx={{ mt: 1 }}><FormControlLabel control={<Checkbox checked={showIdentityOverlay} onChange={(event) => { setShowIdentityOverlay(event.target.checked); if (event.target.checked) setIdentityOverlayError('') }} />} label="Mostrar IDs Tracking V1" />{showIdentityOverlay && <ToggleButtonGroup size="small" exclusive value={identityOverlaySource} onChange={(_, source: TrackingIdentityOverlaySource | null) => { if (source) { setIdentityOverlaySource(source); setIdentityOverlay(null); setIdentityOverlayError('') } }}><ToggleButton value="fixed_roster">Fixed roster 7I.4</ToggleButton><ToggleButton value="round_state">Legacy round state</ToggleButton></ToggleButtonGroup>}{showIdentityOverlay && identityOverlay && <><Typography variant="caption" sx={{ display: 'block' }}>{identityOverlay.source_kind === 'fixed_roster' ? <>Tracking V1 · Fixed roster 7I.4 · {identityOverlay.identity_count} actores · {identityOverlay.assigned_tracklet_count} asignados · {identityOverlay.unassigned_tracklet_count} pendientes</> : <>Tracking V1 · {identityOverlay.state_id} · {identityOverlay.identity_count} fragmentos provisionales</>}</Typography><Stack direction="row" spacing={1} flexWrap="wrap"><FormControlLabel control={<Checkbox size="small" checked={showTrackletId} onChange={(event) => setShowTrackletId(event.target.checked)} />} label="Tracklet ID" /><FormControlLabel control={<Checkbox size="small" checked={showLens} onChange={(event) => setShowLens(event.target.checked)} />} label="Lente" /><FormControlLabel control={<Checkbox size="small" checked={markInterpolation} onChange={(event) => setMarkInterpolation(event.target.checked)} />} label="Marcar interpolación" /></Stack></>}{showIdentityOverlay && identityOverlayError && <Typography variant="caption" color="text.secondary">Tracking V1 no disponible</Typography>}</Box>}
    {selected && <Alert severity="info" sx={{ mt: 1 }}>T{selected.track_id} · {selected.position_kind} · {selected.lifecycle} · X {selected.x.toFixed(2)} Z {selected.z.toFixed(2)} · Vx {selected.vx?.toFixed(2) ?? '—'} Vz {selected.vz?.toFixed(2) ?? '—'} · hits {selected.hits ?? '—'} · missed {selected.missed_seconds?.toFixed(2) ?? '—'}</Alert>}{selectedFixed && <Alert severity="info" sx={{ mt: 1 }}>{selectedFixed.identityId} · {selectedFixed.trackletId} · {selectedFixed.assignmentStatus} · X {selectedFixed.x.toFixed(2)} Z {selectedFixed.z.toFixed(2)}</Alert>}
    <Box sx={{ mt: 1.5, borderTop: 1, borderColor: 'divider', pt: 1 }}><Stack direction="row" gap={1} alignItems="center" flexWrap="wrap"><Typography variant="body2">Detection threshold: {detectionThreshold.toFixed(2)}</Typography><Slider aria-label="Detection threshold" min={0.20} max={0.70} step={0.05} value={detectionThreshold} onChange={(_, value) => setDetectionThreshold(value as number)} sx={{ width: 150 }} /><Button disabled={active || fixedRosterSelected || actorTrajectorySelected} variant="outlined" onClick={() => void start('all')}>Analizar</Button><Button disabled={active || fixedRosterSelected || actorTrajectorySelected || (trackingDisplayMode !== 'v8_canonical' && !clip.analysis?.detection_artifact_id)} onClick={() => void start('tracking')}>Rehacer tracking</Button>{job && <Typography variant="caption">{jobLabel}</Typography>}</Stack>{job && <LinearProgress sx={{ mt: 0.5 }} variant={v8WaitingForRealProgress ? "indeterminate" : "determinate"} value={v8WaitingForRealProgress ? undefined : Math.min(100, job.progress * 100)} />}{job?.status === 'failed' && <Alert severity="error" sx={{ mt: 0.5 }}>{job.error || 'Análisis fallido'}</Alert>}</Box>
  </Box>
}
