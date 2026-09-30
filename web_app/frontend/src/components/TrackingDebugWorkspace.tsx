import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Alert, Box, Button, Divider, Stack, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { api } from '../api/client'
import type { Clip, MediaItem, ViewContext } from '../types'
import type { FinalPositionMode, TrackingFieldPoint, TrackingFinalArtifact, TrackingPersistentArtifact, TrackingReview, TrackingReviews } from '../types/trackingFinal'
import FinalTrackingPitch from './FinalTrackingPitch'
import MatchViewer from './MatchViewer'
import type { InteractiveCamera } from './interactiveCamera'
import { formatMediaTime } from './mediaTime'
import { nearestFinalFrame, reviewSeekTimestamp } from './trackingFinalDebug'
import { activePresentation, controlPresentationPlayback, hasActivePresentation, synchronizePresentations, timeForViewChange } from './trackingDebugPresentation'
import type { TrackingDebugView } from './trackingDebugPresentation'
import { startVideoPresentationClock } from './videoPresentationClock'

interface Props { matchId: string; clip: Clip; source?: MediaItem; context: ViewContext | null; onBack: () => void }
type TrackingSource = 'physical' | 'persistent'

const categories: Array<[keyof Pick<TrackingReviews, 'speed_reviews' | 'acceleration_reviews' | 'adjustment_reviews' | 'uncertainty_reviews' | 'short_segments'>, string]> = [
  ['speed_reviews', 'Speed reviews'], ['acceleration_reviews', 'Acceleration reviews'], ['adjustment_reviews', 'Adjustment reviews'], ['uncertainty_reviews', 'Uncertainty reviews'], ['short_segments', 'Short segments'],
]
const number = (value: number | null | undefined, digits = 3) => typeof value === 'number' ? value.toFixed(digits) : '—'

export default function TrackingDebugWorkspace({ matchId, clip, source, context, onBack }: Props) {
  const original = useRef<HTMLVideoElement>(null), ultra = useRef<HTMLVideoElement>(null), interactive = useRef<HTMLVideoElement>(null)
  const interactiveCamera = useRef<InteractiveCamera | null>(null)
  const switchingView = useRef(false)
  const [artifact, setArtifact] = useState<TrackingFinalArtifact | null>(null)
  const [persistent, setPersistent] = useState<TrackingPersistentArtifact | null>(null)
  const [reviews, setReviews] = useState<TrackingReviews | null>(null)
  const [view, setView] = useState<TrackingDebugView>('original'), [time, setTime] = useState(0), [playing, setPlaying] = useState(false), [error, setError] = useState(''), [ultraError, setUltraError] = useState('')
  const [trackingSource, setTrackingSource] = useState<TrackingSource>('physical'), [mode, setMode] = useState<FinalPositionMode>('smoothed'), [selectedIdentityId, setSelectedIdentityId] = useState<string | null>(null), [showPointOrigin, setShowPointOrigin] = useState(false), [persistentError, setPersistentError] = useState('')
  const duration = source?.duration_s ?? clip.duration_s ?? 0
  const fieldLength = context?.field_length ?? 105, fieldWidth = context?.field_width ?? 68
  const presentations = useCallback(() => ({ original: original.current, ultra: ultra.current, interactive: interactive.current }), [])
  const masterVideo = useCallback(() => activePresentation(view, presentations()) as HTMLVideoElement | null, [presentations, view])
  const viewAvailable = hasActivePresentation(view, { hasSource: !!source, hasUltra: !!clip.ultra_base_url, hasContext: !!context, ultraFailed: !!ultraError })
  const allPoints = useMemo<TrackingFieldPoint[]>(() => trackingSource === 'physical' ? artifact?.trajectories.flatMap((trajectory) => trajectory.points) ?? [] : persistent?.trajectories.flatMap((trajectory) => trajectory.points) ?? [], [artifact, persistent, trackingSource])
  const currentFrame = useMemo(() => nearestFinalFrame(allPoints, time), [allPoints, time])
  const selectedPoint = currentFrame?.points.find((point) => ('actor_id' in point ? point.actor_id : point.identity_id) === selectedIdentityId) ?? null
  const selectedTrajectory = useMemo<TrackingFieldPoint[]>(() => {
    if (!selectedIdentityId || !selectedPoint) return []
    if (trackingSource === 'physical') return artifact?.trajectories.find((trajectory) => trajectory.identity_id === selectedIdentityId)?.points.filter((point) => point.timestamp_s >= selectedPoint.timestamp_s - 5 && point.timestamp_s <= selectedPoint.timestamp_s) ?? []
    return persistent?.trajectories.find((trajectory) => trajectory.actor_id === selectedIdentityId)?.points.filter((point) => point.timestamp_s >= selectedPoint.timestamp_s - 5 && point.timestamp_s <= selectedPoint.timestamp_s) ?? []
  }, [artifact, persistent, selectedIdentityId, selectedPoint, trackingSource])

  useEffect(() => {
    let active = true
    setArtifact(null); setPersistent(null); setReviews(null); setError(''); setPersistentError(''); setUltraError(''); setTrackingSource('physical'); setSelectedIdentityId(null); setView('original'); setPlaying(false)
    Promise.all([api.trackingFinal(matchId, clip.id), api.trackingFinalSummary(matchId, clip.id), api.trackingFinalReviews(matchId, clip.id)]).then(([final, summary, reviewQueue]) => {
      if (active) { setArtifact({ ...final, summary }); setReviews(reviewQueue) }
    }).catch((reason: Error) => { if (active) setError(reason.message) })
    return () => { active = false }
  }, [clip.id, matchId])
  useEffect(() => {
    if (trackingSource !== 'persistent' || persistent) return
    let active = true
    setPersistentError('')
    api.trackingPersistent(matchId, clip.id).then((result) => { if (active) setPersistent(result) }).catch((reason: Error) => { if (active) setPersistentError(reason.message) })
    return () => { active = false }
  }, [clip.id, matchId, persistent, trackingSource])
  useEffect(() => {
    if (!artifact) return
    const element = masterVideo()
    if (!element) return
    return startVideoPresentationClock(element, (next) => setTime(next))
  }, [artifact, masterVideo, source?.id, clip.ultra_base_url, context])
  const seek = useCallback((next: number) => {
    const target = Math.max(0, Math.min(duration, next))
    setTime(target)
    synchronizePresentations(presentations(), target)
  }, [duration, presentations])
  useEffect(() => { synchronizePresentations(presentations(), time) }, [presentations, time, view])
  useEffect(() => { controlPresentationPlayback(presentations(), masterVideo(), playing) }, [masterVideo, playing, presentations, view])
  useEffect(() => { switchingView.current = false }, [view])
  const handlePlayback = useCallback((kind: TrackingDebugView, value: boolean) => {
    if (kind === view && (value || !switchingView.current)) setPlaying(value)
  }, [view])
  const changeView = (next: TrackingDebugView) => {
    const current = timeForViewChange(view, presentations(), time, duration)
    switchingView.current = true
    masterVideo()?.pause()
    setTime(current)
    setView(next)
  }
  const selectReview = (review: TrackingReview) => {
    if (review.identity_id) setSelectedIdentityId(review.identity_id)
    const timestamp = reviewSeekTimestamp(review)
    if (timestamp !== null) { masterVideo()?.pause(); setPlaying(false); seek(timestamp) }
  }
  const changeTrackingSource = (next: TrackingSource) => {
    if (next === trackingSource) return
    setSelectedIdentityId(null)
    setTrackingSource(next)
  }
  const persistentCoverage = typeof persistent?.audit?.represented_coverage_fraction === 'number' ? persistent.audit.represented_coverage_fraction : persistent?.summary.represented_coverage_fraction

  return <Box sx={{ width: '100%', px: { xs: 1, md: 3 }, py: 2 }}>
    <Button onClick={onBack}>← Clips</Button><Typography variant="h5">Tracking Debug</Typography><Typography color="text.secondary">Tracking V1 · final-publish-v1 canónico · {clip.name || clip.id}</Typography>
    {error && <Alert severity="info" sx={{ mt: 1 }}>{error}</Alert>}
    {!error && !artifact && <Typography sx={{ mt: 2 }}>Cargando FINAL…</Typography>}
    {artifact && <>
      <ToggleButtonGroup sx={{ mt: 1 }} size="small" exclusive value={view} onChange={(_, value: TrackingDebugView | null) => value && changeView(value)}><ToggleButton value="original">Original</ToggleButton><ToggleButton value="ultra">Ultra panorama</ToggleButton><ToggleButton value="interactive">Interactivo</ToggleButton></ToggleButtonGroup>
      <Stack direction="row" alignItems="center" gap={1} mt={1} flexWrap="wrap"><Typography variant="body2">Tracking source</Typography><ToggleButtonGroup size="small" exclusive value={trackingSource} onChange={(_, value: TrackingSource | null) => value && changeTrackingSource(value)}><ToggleButton value="physical">Physical V1</ToggleButton><ToggleButton value="persistent">Persistent V1</ToggleButton></ToggleButtonGroup>{trackingSource === 'persistent' && persistent && <Typography variant="caption">Persistent V1 · {String(persistent.summary.actor_count)} actors · {typeof persistentCoverage === 'number' ? `${(persistentCoverage * 100).toFixed(1)}% coverage` : ''}</Typography>}</Stack>
      {persistentError && trackingSource === 'persistent' && <Alert severity="info" sx={{ mt: 1 }}>{persistentError}</Alert>}
      <Box sx={{ mt: 1.5, display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) minmax(350px, .9fr)' }, gap: 2, alignItems: 'start' }}>
        <Box minWidth={0} sx={{ '& > div': { width: '100%' }, '& canvas': { width: '100% !important', maxWidth: '100%' } }}>
          {view === 'original' && (source ? <video ref={original} preload="metadata" playsInline src={api.url(source.url)} onLoadedMetadata={(event) => { event.currentTarget.currentTime = time }} onPlay={() => handlePlayback('original', true)} onPause={() => handlePlayback('original', false)} onEnded={() => handlePlayback('original', false)} style={{ width: '100%', maxHeight: '58vh', background: '#080a0c' }} /> : <Alert severity="warning">Falta el vídeo source.mp4 físico del clip.</Alert>)}
          {view === 'ultra' && (clip.ultra_base_url ? ultraError ? <Alert severity="error">No se puede reproducir el Ultra panorama en este navegador. {ultraError}</Alert> : <video ref={ultra} muted playsInline src={api.url(clip.ultra_base_url)} onLoadedMetadata={(event) => { event.currentTarget.currentTime = time }} onPlay={() => handlePlayback('ultra', true)} onPause={() => handlePlayback('ultra', false)} onEnded={() => handlePlayback('ultra', false)} onError={() => setUltraError('El archivo no tiene un códec de vídeo compatible.')} style={{ width: '100%', maxHeight: '58vh', objectFit: 'contain', background: '#080a0c' }} /> : <Alert severity="info">Ejecuta el análisis para generar el ultra panorama.</Alert>)}
          {view === 'interactive' && (source && context ? <MatchViewer media={source} mode="interactive" crop="stacked" context={context} width={1920} videoRef={interactive} initialTime={time} cameraRef={interactiveCamera} onTime={setTime} onPlayback={(value) => handlePlayback('interactive', value)} points={[]} selectionEnabled={false} onClick={() => undefined} /> : <Alert severity="info">{!source ? 'Falta el vídeo físico del clip.' : 'Falta la calibración del partido.'}</Alert>)}
          <Stack direction="row" alignItems="center" gap={1} mt={1} flexWrap="wrap"><Button disabled={!viewAvailable} onClick={() => { if (masterVideo()?.paused) void masterVideo()?.play().catch(() => undefined); else masterVideo()?.pause() }}>{playing ? 'Pausar' : 'Reproducir'}</Button><Button disabled={!viewAvailable} onClick={() => { masterVideo()?.pause(); seek(time - 10) }}>−10</Button><Button disabled={!viewAvailable} onClick={() => { masterVideo()?.pause(); seek(time + 10) }}>+10</Button><input aria-label="Tiempo Tracking Debug" type="range" min="0" max={duration} step=".01" value={time} onChange={(event) => seek(Number(event.target.value))} style={{ flex: 1, minWidth: 160 }} /></Stack>
          <Typography mt={.5} variant="body2" sx={{ fontVariantNumeric: 'tabular-nums' }}>Vídeo {formatMediaTime(time)} / {formatMediaTime(duration)} · V1 frame {currentFrame?.frameIndex ?? '—'} · t publicado {currentFrame ? formatMediaTime(currentFrame.timestampS) : '—'}</Typography>
        </Box>
        <Box minWidth={0}><Stack direction="row" alignItems="center" justifyContent="space-between" mb={.5}><Typography variant="subtitle1">Campo 2D</Typography><ToggleButtonGroup size="small" exclusive value={mode} onChange={(_, value: FinalPositionMode | null) => value && setMode(value)}><ToggleButton value="smoothed">Final/smoothed</ToggleButton><ToggleButton value="measurement">Measurement/raw</ToggleButton></ToggleButtonGroup></Stack>
          <FinalTrackingPitch fieldLength={fieldLength} fieldWidth={fieldWidth} points={currentFrame?.points ?? []} selectedIdentityId={selectedIdentityId} selectedTrajectory={selectedTrajectory} segmentBreaks={trackingSource === 'physical' ? artifact.segment_breaks : []} mode={mode} showPointOrigin={trackingSource === 'persistent' && showPointOrigin} onSelect={setSelectedIdentityId} />
          {trackingSource === 'physical' ? <Typography variant="caption">● observed · ◆ fused · ■ interpolated. La línea blanca sólo une puntos del mismo segmento.</Typography> : <Stack direction="row" alignItems="center" gap={1} flexWrap="wrap"><ToggleButton size="small" value="origin" selected={showPointOrigin} onChange={() => setShowPointOrigin((value) => !value)}>Show point origin</ToggleButton>{showPointOrigin && <Typography variant="caption">F fused · I interpolated · P predicted</Typography>}{mode === 'measurement' && <Typography variant="caption">Raw muestra sólo mediciones publicadas; puntos synthetic no tienen medición raw.</Typography>}</Stack>}
        </Box>
      </Box>
      <Box sx={{ mt: 2, display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(300px, .8fr) minmax(0, 1.2fr)' }, gap: 2 }}>
        <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5 }}><Typography variant="subtitle1">{trackingSource === 'physical' ? 'Identidad seleccionada' : 'Actor seleccionado'}</Typography>{selectedPoint ? <Stack spacing={.25} mt={.5} data-testid="selected-point-details">
          {'actor_id' in selectedPoint ? <><Typography><b>{selectedPoint.actor_id}</b> · {selectedPoint.segment_id}</Typography><Typography variant="body2">origin {selectedPoint.origin} · frame {selectedPoint.frame_index} · t {number(selectedPoint.timestamp_s)} s</Typography><Typography variant="body2">published x/z: {number(selectedPoint.x)} / {number(selectedPoint.z)} m</Typography><Typography variant="body2">measurement x/z: {selectedPoint.measurement_x === null || selectedPoint.measurement_z === null ? '— (synthetic: no raw measurement)' : `${number(selectedPoint.measurement_x)} / ${number(selectedPoint.measurement_z)} m`}</Typography><Typography variant="body2">speed {number(selectedPoint.speed_mps)} m/s · std {number(selectedPoint.position_std_m)} m · {selectedPoint.quality ?? '—'}</Typography></> : <><Typography><b>{selectedPoint.identity_id}</b> · {selectedPoint.segment_id}</Typography><Typography variant="body2">origin {selectedPoint.origin} · frame {selectedPoint.frame_index} · t {number(selectedPoint.timestamp_s)} s</Typography><Typography variant="body2">measurement x/z: {number(selectedPoint.measurement_x)} / {number(selectedPoint.measurement_z)} m</Typography><Typography variant="body2">smoothed x/z: {number(selectedPoint.smoothed_x)} / {number(selectedPoint.smoothed_z)} m</Typography><Typography variant="body2">speed {number(selectedPoint.speed_mps)} m/s · acceleration {number(selectedPoint.acceleration_mps2)} m/s²</Typography><Typography variant="body2">adjustment {number(selectedPoint.position_adjustment_m)} m · std {number(selectedPoint.position_std_m)} m · {selectedPoint.kinematics_quality ?? '—'}</Typography></>}
        </Stack> : <Typography mt={.5} color="text.secondary">Haz clic en una identidad del campo.</Typography>}</Box>
        <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5, maxHeight: 310, overflow: 'auto' }}><Typography variant="subtitle1">{trackingSource === 'physical' ? 'Review queue' : 'Persistent validation'}</Typography>{trackingSource === 'physical' ? categories.map(([key, label]) => <Box key={key} mt={1}><Typography variant="body2" fontWeight={700}>{label} ({reviews?.[key]?.length ?? 0})</Typography>{(reviews?.[key] ?? []).map((review, index) => <Button key={`${key}-${index}`} size="small" sx={{ mr: .5, mt: .25, textTransform: 'none' }} onClick={() => selectReview(review)}>{review.identity_id ?? 'sin identidad'} · {review.frame_index ?? review.start_frame_index ?? '—'}</Button>)}</Box>) : <Typography mt={.5} variant="body2">{persistent ? `Audit: ${String(persistent.audit?.status ?? 'available')} · ${String(persistent.audit?.full_population_frame_count ?? '—')} full-population frames.` : 'Cargando Persistent V1…'}</Typography>}</Box>
      </Box>
      <Divider sx={{ my: 1.5 }} /><Typography variant="caption">Publicado: {trackingSource === 'physical' ? `${String(artifact.index.publisher_version)} · ${artifact.trajectories.length} trayectorias` : persistent ? `${String(persistent.index.trajectory_version)} · ${persistent.trajectories.length} actores` : 'Persistent V1 pendiente'} · no se recalculan posiciones, velocidades ni identidades en esta vista.</Typography>
    </>}
  </Box>
}
