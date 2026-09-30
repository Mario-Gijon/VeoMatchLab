import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Alert, Box, Button, Divider, Stack, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { api } from '../api/client'
import type { Clip, MediaItem } from '../types'
import type { FinalPositionMode, TrackingFinalArtifact, TrackingReview, TrackingReviews } from '../types/trackingFinal'
import FinalTrackingPitch from './FinalTrackingPitch'
import { formatMediaTime } from './mediaTime'
import { nearestFinalFrame, reviewSeekTimestamp } from './trackingFinalDebug'
import { startVideoPresentationClock } from './videoPresentationClock'

interface Props { matchId: string; clip: Clip; source?: MediaItem; fieldLength?: number; fieldWidth?: number; onBack: () => void }

const categories: Array<[keyof Pick<TrackingReviews, 'speed_reviews' | 'acceleration_reviews' | 'adjustment_reviews' | 'uncertainty_reviews' | 'short_segments'>, string]> = [
  ['speed_reviews', 'Speed reviews'], ['acceleration_reviews', 'Acceleration reviews'], ['adjustment_reviews', 'Adjustment reviews'], ['uncertainty_reviews', 'Uncertainty reviews'], ['short_segments', 'Short segments'],
]
const number = (value: number | null | undefined, digits = 3) => typeof value === 'number' ? value.toFixed(digits) : '—'

export default function TrackingDebugWorkspace({ matchId, clip, source, fieldLength = 105, fieldWidth = 68, onBack }: Props) {
  const video = useRef<HTMLVideoElement>(null)
  const [artifact, setArtifact] = useState<TrackingFinalArtifact | null>(null)
  const [reviews, setReviews] = useState<TrackingReviews | null>(null)
  const [time, setTime] = useState(0), [playing, setPlaying] = useState(false), [error, setError] = useState('')
  const [mode, setMode] = useState<FinalPositionMode>('smoothed'), [selectedIdentityId, setSelectedIdentityId] = useState<string | null>(null)
  const duration = source?.duration_s ?? clip.duration_s ?? 0
  const allPoints = useMemo(() => artifact?.trajectories.flatMap((trajectory) => trajectory.points) ?? [], [artifact])
  const currentFrame = useMemo(() => nearestFinalFrame(allPoints, time), [allPoints, time])
  const selectedPoint = currentFrame?.points.find((point) => point.identity_id === selectedIdentityId) ?? null
  const selectedTrajectory = useMemo(() => {
    if (!selectedIdentityId || !selectedPoint) return []
    return artifact?.trajectories.find((trajectory) => trajectory.identity_id === selectedIdentityId)?.points.filter((point) => point.timestamp_s >= selectedPoint.timestamp_s - 5 && point.timestamp_s <= selectedPoint.timestamp_s) ?? []
  }, [artifact, selectedIdentityId, selectedPoint])

  useEffect(() => {
    let active = true
    setArtifact(null); setReviews(null); setError(''); setSelectedIdentityId(null)
    Promise.all([api.trackingFinal(matchId, clip.id), api.trackingFinalSummary(matchId, clip.id), api.trackingFinalReviews(matchId, clip.id)]).then(([final, summary, reviewQueue]) => {
      if (active) { setArtifact({ ...final, summary }); setReviews(reviewQueue) }
    }).catch((reason: Error) => { if (active) setError(reason.message) })
    return () => { active = false }
  }, [clip.id, matchId])
  useEffect(() => {
    if (!artifact) return
    const element = video.current
    if (!element) return
    return startVideoPresentationClock(element, (next) => setTime(next))
  }, [artifact, source?.id])
  const seek = useCallback((next: number) => {
    const target = Math.max(0, Math.min(duration, next))
    setTime(target)
    if (video.current) video.current.currentTime = target
  }, [duration])
  const selectReview = (review: TrackingReview) => {
    if (review.identity_id) setSelectedIdentityId(review.identity_id)
    const timestamp = reviewSeekTimestamp(review)
    if (timestamp !== null) { video.current?.pause(); setPlaying(false); seek(timestamp) }
  }

  return <Box sx={{ width: '100%', px: { xs: 1, md: 3 }, py: 2 }}>
    <Button onClick={onBack}>← Clips</Button><Typography variant="h5">Tracking Debug</Typography><Typography color="text.secondary">Tracking V1 · final-publish-v1 canónico · {clip.name || clip.id}</Typography>
    {error && <Alert severity="info" sx={{ mt: 1 }}>{error}</Alert>}
    {!error && !artifact && <Typography sx={{ mt: 2 }}>Cargando FINAL…</Typography>}
    {artifact && <>
      <Box sx={{ mt: 1.5, display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) minmax(350px, .9fr)' }, gap: 2, alignItems: 'start' }}>
        <Box minWidth={0}><video ref={video} preload="metadata" playsInline src={source ? api.url(source.url) : undefined} onLoadedMetadata={(event) => { event.currentTarget.currentTime = time }} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} style={{ width: '100%', maxHeight: '58vh', display: source ? 'block' : 'none', background: '#080a0c' }} />
          {!source && <Alert severity="warning">Falta el vídeo source.mp4 físico del clip.</Alert>}
          <Stack direction="row" alignItems="center" gap={1} mt={1} flexWrap="wrap"><Button disabled={!source} onClick={() => { if (video.current?.paused) void video.current.play().catch(() => undefined); else video.current?.pause() }}>{playing ? 'Pausar' : 'Reproducir'}</Button><Button disabled={!source} onClick={() => { video.current?.pause(); seek(time - 10) }}>−10</Button><Button disabled={!source} onClick={() => { video.current?.pause(); seek(time + 10) }}>+10</Button><input aria-label="Tiempo Tracking Debug" type="range" min="0" max={duration} step=".01" value={time} onChange={(event) => seek(Number(event.target.value))} style={{ flex: 1, minWidth: 160 }} /></Stack>
          <Typography mt={.5} variant="body2" sx={{ fontVariantNumeric: 'tabular-nums' }}>Vídeo {formatMediaTime(time)} / {formatMediaTime(duration)} · V1 frame {currentFrame?.frameIndex ?? '—'} · t publicado {currentFrame ? formatMediaTime(currentFrame.timestampS) : '—'}</Typography>
        </Box>
        <Box minWidth={0}><Stack direction="row" alignItems="center" justifyContent="space-between" mb={.5}><Typography variant="subtitle1">Campo 2D</Typography><ToggleButtonGroup size="small" exclusive value={mode} onChange={(_, value: FinalPositionMode | null) => value && setMode(value)}><ToggleButton value="smoothed">Final/smoothed</ToggleButton><ToggleButton value="measurement">Measurement/raw</ToggleButton></ToggleButtonGroup></Stack>
          <FinalTrackingPitch fieldLength={fieldLength} fieldWidth={fieldWidth} points={currentFrame?.points ?? []} selectedIdentityId={selectedIdentityId} selectedTrajectory={selectedTrajectory} segmentBreaks={artifact.segment_breaks} mode={mode} onSelect={setSelectedIdentityId} />
          <Typography variant="caption">● observed · ◆ fused · ■ interpolated. La línea blanca sólo une puntos del mismo segmento.</Typography>
        </Box>
      </Box>
      <Box sx={{ mt: 2, display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(300px, .8fr) minmax(0, 1.2fr)' }, gap: 2 }}>
        <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5 }}><Typography variant="subtitle1">Identidad seleccionada</Typography>{selectedPoint ? <Stack spacing={.25} mt={.5} data-testid="selected-point-details">
          <Typography><b>{selectedPoint.identity_id}</b> · {selectedPoint.segment_id}</Typography>
          <Typography variant="body2">origin {selectedPoint.origin} · frame {selectedPoint.frame_index} · t {number(selectedPoint.timestamp_s)} s</Typography>
          <Typography variant="body2">measurement x/z: {number(selectedPoint.measurement_x)} / {number(selectedPoint.measurement_z)} m</Typography>
          <Typography variant="body2">smoothed x/z: {number(selectedPoint.smoothed_x)} / {number(selectedPoint.smoothed_z)} m</Typography>
          <Typography variant="body2">speed {number(selectedPoint.speed_mps)} m/s · acceleration {number(selectedPoint.acceleration_mps2)} m/s²</Typography>
          <Typography variant="body2">adjustment {number(selectedPoint.position_adjustment_m)} m · std {number(selectedPoint.position_std_m)} m · {selectedPoint.kinematics_quality ?? '—'}</Typography>
        </Stack> : <Typography mt={.5} color="text.secondary">Haz clic en una identidad del campo.</Typography>}</Box>
        <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5, maxHeight: 310, overflow: 'auto' }}><Typography variant="subtitle1">Review queue</Typography>{categories.map(([key, label]) => <Box key={key} mt={1}><Typography variant="body2" fontWeight={700}>{label} ({reviews?.[key]?.length ?? 0})</Typography>{(reviews?.[key] ?? []).map((review, index) => <Button key={`${key}-${index}`} size="small" sx={{ mr: .5, mt: .25, textTransform: 'none' }} onClick={() => selectReview(review)}>{review.identity_id ?? 'sin identidad'} · {review.frame_index ?? review.start_frame_index ?? '—'}</Button>)}</Box>)}</Box>
      </Box>
      <Divider sx={{ my: 1.5 }} /><Typography variant="caption">Publicado: {String(artifact.index.publisher_version)} · {artifact.trajectories.length} trayectorias · no se recalculan posiciones, velocidades ni identidades en esta vista.</Typography>
    </>}
  </Box>
}
