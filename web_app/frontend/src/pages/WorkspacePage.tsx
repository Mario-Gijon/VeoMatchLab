import { useEffect, useRef, useState } from 'react'
import Replay10Icon from '@mui/icons-material/Replay10'
import Forward10Icon from '@mui/icons-material/Forward10'
import PauseIcon from '@mui/icons-material/Pause'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Divider, FormControl, IconButton, InputLabel, LinearProgress, MenuItem, Select, Stack, ToggleButton, ToggleButtonGroup, Tooltip, Typography } from '@mui/material'
import { api } from '../api/client'
import ManageFilesDialog from '../components/ManageFilesDialog'
import MatchViewer from '../components/MatchViewer'
import type { ViewerPick, ViewMode } from '../components/MatchViewer'
import { interactiveGroundPoint } from '../components/interactiveCamera'
import type { InteractiveCamera } from '../components/interactiveCamera'
import { appendMeasurement, matchTimestamp, pitchPoints, ultraPanoramaSize, viewLabel, VIEW_MODES } from '../components/workspaceGeometry'
import { clampMediaTime, formatMediaTime } from '../components/mediaTime'
import PanoramaViewer from '../components/PanoramaViewer'
import PitchSvg from '../components/PitchSvg'
import ValidationTable from '../components/ValidationTable'
import EventsPanel from '../components/EventsPanel'
import MatchTimeline from '../components/MatchTimeline'
import ClipsPanel from '../components/ClipsPanel'
import ClipWorkspace from '../components/ClipWorkspace'
import { useMatchEvents } from '../hooks/useMatchEvents'
import type { MatchEvent } from '../types/matchData'
import type { Calibration, Clip, GroundPoint, MatchWorkspace, MediaItem, RenderedPanorama, ViewContext } from '../types'

interface Props { matchId: string; calibrations: Calibration[]; onBack: () => void }

export default function WorkspacePage({ matchId, calibrations, onBack }: Props) {
  const [match, setMatch] = useState<MatchWorkspace | null>(null)
  const events = useMatchEvents(matchId, match?.updated_at)
  const [context, setContext] = useState<ViewContext | null>(null)
  const [mode, setMode] = useState<ViewMode>('panorama')
  const [crop, setCrop] = useState('stacked')
  const [sourceId, setSourceId] = useState('')
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [width, setWidth] = useState(1920)
  const [stillImage, setStillImage] = useState<MediaItem | null>(null)
  const [frameRender, setFrameRender] = useState<RenderedPanorama | null>(null)
  const [points, setPoints] = useState<GroundPoint[]>([])
  const [landmark, setLandmark] = useState('')
  const [filesOpen, setFilesOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [selectionEnabled, setSelectionEnabled] = useState(true)
  const [highlightsOpen, setHighlightsOpen] = useState(false)
  const [section, setSection] = useState<'visualization' | 'clips'>('visualization')
  const [openClip, setOpenClip] = useState<Clip | null>(null)
  const highlightsVideo = useRef<HTMLVideoElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const interactiveCamera = useRef<InteractiveCamera | null>(null)
  const measurementSequence = useRef(0)
  const generation = useRef(0)
  const selectedCalibrationPath = match?.calibration?.path
  const loadedMatchId = match?.id
  const directionsId = match?.media.find((item) => item.role === 'camera_directions')?.id
  useEffect(() => {
    let active = true
    api.match(matchId).then((result) => { if (active) { setMatch(result); if (!result.media.some((item) => item.role === 'dual_lens') && result.media.some((item) => item.role === 'tracking')) setMode('tracking') } }).catch((e: Error) => setError(e.message))
    return () => { active = false }
  }, [matchId])
  useEffect(() => {
    let active = true
    setContext(null); setFrameRender(null); setPoints([]); interactiveCamera.current = null; generation.current++
    if (selectedCalibrationPath && loadedMatchId) api.viewContext(loadedMatchId).then((result) => { if (active) setContext(result) }).catch((e: Error) => { if (active) setError(e.message) })
    return () => { active = false }
  }, [loadedMatchId, selectedCalibrationPath, directionsId])
  const run = async (work: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try { await work() } catch (reason) { setError(reason instanceof Error ? reason.message : 'La operación ha fallado.') }
    finally { setBusy(false) }
  }
  const roleMedia = match?.media.filter((item) => mode === 'tracking' ? item.role === 'tracking' : item.role === 'dual_lens') ?? []
  const source = roleMedia.find((item) => item.id === sourceId) ?? roleMedia[0]
  const highlights = match?.media.find((item) => item.role === 'veo_highlights')
  const duration = source?.duration_s ?? 0
  const playbackDisabled = !source || !!frameRender || !!stillImage
  const canSeekEvent = !!source && (!['panorama', 'ultra_panorama', 'interactive'].includes(mode) || !!context)
  const ultraSize = mode === 'ultra_panorama' ? ultraPanoramaSize(context) : null
  const renderedViewWidth = mode === 'ultra_panorama' ? ultraSize!.width : width
  const renderedViewHeight = mode === 'ultra_panorama' ? ultraSize!.height : width * 9 / 16
  const seek = (value: number) => {
    const next = clampMediaTime(value, duration)
    if (video.current && video.current.readyState >= 1) video.current.currentTime = next
    setTime(next)
  }
  const seekEvent = (event: MatchEvent) => {
    if (!canSeekEvent) return
    video.current?.pause(); setPlaying(false)
    generation.current++; setFrameRender(null); setStillImage(null)
    seek(event.start_s)
  }
  const switchView = (value: ViewMode) => {
    setTime(video.current && video.current.readyState >= 1 ? matchTimestamp(video.current, time) : time)
    generation.current++; setMode(value); setFrameRender(null); setStillImage(null)
    if ((value === 'tracking') !== (mode === 'tracking')) { video.current?.pause(); setPlaying(false) }
  }
  const renderImage = async (item: MediaItem) => {
    video.current?.pause(); setPlaying(false); generation.current++
    const rendered = await api.render(matchId, item.id, width, width * 9 / 16)
    setFrameRender({ ...rendered, timestamp_s: item.timestamp_s }); setStillImage(null); setMode('panorama'); setPoints([])
  }
  const measure = async (x: number, y: number, pick?: ViewerPick) => {
    if (!context) return
    const currentGeneration = generation.current
    const sequence = ++measurementSequence.current
    const timestamp = pick?.timestamp ?? (frameRender ? frameRender.timestamp_s ?? time : matchTimestamp(video.current, time))
    try {
      const result = mode === 'interactive'
        ? (pick?.camera ? interactiveGroundPoint(context, pick.camera, x, y, pick.width, pick.height) : null)
        : mode === 'tracking'
        ? (pick ? await api.trackingPoint(matchId, timestamp, x, y, pick.width, pick.height) : null)
        : frameRender
        ? await api.groundPoint(frameRender.id, x, y, landmark || null)
        : await api.viewPoint(context.id, mode === 'ultra_panorama' ? 'panorama' : mode, x, y, mode === 'dual_lens' ? 2048 : renderedViewWidth, mode === 'dual_lens' ? 2048 : renderedViewHeight, landmark || null)
      if (!result || generation.current !== currentGeneration || sequence !== measurementSequence.current) return
      setPoints((items) => appendMeasurement(items, { ...result, timestamp_s: timestamp, view: mode, view_width: renderedViewWidth, media_id: frameRender?.id ?? source?.id }))
    } catch (e) { if (mode !== 'tracking' && mode !== 'interactive') setError(e instanceof Error ? e.message : 'No se pudo medir el punto.') }
  }
  if (!match) return <Typography sx={{ p: 3 }}>{error || 'Cargando partido…'}</Typography>
  const displayPoints = points.filter((point) => point.view === mode && point.media_id === (frameRender?.id ?? source?.id) && (mode === 'tracking' || point.view_width === renderedViewWidth))
  return <Box sx={{ p: { xs: 1, md: 2 } }}>
    <Stack direction="row" alignItems="center" gap={2} mb={1}><Button onClick={onBack}>← Partidos</Button><Typography variant="h6">{match.name}</Typography><Box flex={1} /><Button variant="outlined" size="small" onClick={() => setFilesOpen(true)}>Gestionar archivos</Button></Stack>
    {error && <Alert severity="error" onClose={() => setError('')}>{error}</Alert>}{busy && <LinearProgress />}
    {filesOpen && <ManageFilesDialog match={match} calibrations={calibrations} onClose={() => setFilesOpen(false)} onUpdated={(result) => { video.current?.pause(); setPlaying(false); setFrameRender(null); setStillImage(null); setMatch(result); generation.current++ }} onImage={(item) => { video.current?.pause(); setPlaying(false); setStillImage(item); setFrameRender(null); setFilesOpen(false) }} onRender={(item) => { setFilesOpen(false); void run(() => renderImage(item)) }} />}
    <ToggleButtonGroup size="small" exclusive value={section} onChange={(_, value: 'visualization' | 'clips' | null) => value && setSection(value)} sx={{ mb: 1 }}><ToggleButton value="visualization">Visualización</ToggleButton><ToggleButton value="clips">Clips</ToggleButton></ToggleButtonGroup>
    {section === 'clips' ? (openClip ? <ClipWorkspace matchId={matchId} clip={openClip} source={match.media.find((item) => item.id === openClip.clip_media_id)} onBack={() => setOpenClip(null)} onChanged={() => { void api.match(matchId).then((next) => { setMatch(next); setOpenClip(next.clips?.find((item) => item.id === openClip.id) ?? null) }).catch((reason: Error) => setError(reason.message)) }} /> : <ClipsPanel matchId={matchId} source={match.media.find((item) => item.role === 'dual_lens')} time={time} clips={match.clips ?? []} onOpen={setOpenClip} onChanged={() => { void api.match(matchId).then(setMatch).catch((reason: Error) => setError(reason.message)) }} />) : <>
    <Stack direction="row" alignItems="center" flexWrap="wrap" gap={1} mb={1}>
      <ToggleButtonGroup size="small" exclusive value={mode} onChange={(_, value: ViewMode | null) => value && switchView(value)}>{VIEW_MODES.map((view) => <ToggleButton key={view} value={view}>{viewLabel(view)}</ToggleButton>)}</ToggleButtonGroup>
      <ToggleButton size="small" value="select" selected={selectionEnabled} onChange={() => setSelectionEnabled((value) => !value)}>Seleccionar</ToggleButton>
      {highlights && <Button size="small" variant="outlined" onClick={() => setHighlightsOpen(true)}>Ver jugadas destacadas</Button>}
      {roleMedia.length > 1 && <Select size="small" value={source?.id ?? ''} displayEmpty onChange={(e) => { video.current?.pause(); setPlaying(false); setSourceId(e.target.value); setFrameRender(null); setStillImage(null); generation.current++ }}><MenuItem value="">Sin fuente asignada</MenuItem>{roleMedia.map((item) => <MenuItem key={item.id} value={item.id}>{item.name}</MenuItem>)}</Select>}
      {mode === 'dual_lens' && <ToggleButtonGroup size="small" exclusive value={crop} onChange={(_, value: string | null) => value && setCrop(value)}><ToggleButton value="stacked">Apilado</ToggleButton><ToggleButton value="left">IZQUIERDA</ToggleButton><ToggleButton value="right">DERECHA</ToggleButton></ToggleButtonGroup>}
      {mode === 'panorama' && <Select size="small" value={width} onChange={(e) => { setWidth(Number(e.target.value)); setFrameRender(null); generation.current++ }}><MenuItem value={1920}>1920×1080</MenuItem><MenuItem value={3840}>3840×2160</MenuItem></Select>}
      <Button size="small" variant="outlined" onClick={() => setSection('clips')}>Crear clip</Button>
    </Stack>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 7fr) minmax(280px, 3fr)' }, gap: 2 }}>
      <Box minWidth={0}>
        {stillImage ? <Box><img src={api.url(stillImage.url)} alt={stillImage.name} style={{ width: '100%', maxHeight: '65vh', objectFit: 'contain' }} /><Button onClick={() => setStillImage(null)}>Volver al vídeo</Button></Box> : frameRender ? <PanoramaViewer panorama={frameRender} points={selectionEnabled ? displayPoints : []} onPixelClick={(x, y) => selectionEnabled && void measure(x, y)} /> : source && (!['panorama', 'ultra_panorama', 'interactive'].includes(mode) || context) ? <MatchViewer key={source.id} media={source} mode={mode} crop={crop} context={context} width={width} initialTime={time} videoRef={video} cameraRef={interactiveCamera} onTime={setTime} onPlayback={setPlaying} points={selectionEnabled ? displayPoints : []} selectionEnabled={selectionEnabled} trackingAvailable={!!context?.tracking_geometry.available} onClick={(x, y, pick) => void measure(x, y, pick)} /> : <Alert severity="info">{['panorama', 'ultra_panorama', 'interactive'].includes(mode) && !context ? 'Falta la calibración.' : 'Falta el vídeo. Añádelo en Gestionar archivos.'}</Alert>}
        <Stack direction="row" alignItems="center" gap={1} mt={1} flexWrap="wrap">
          <Tooltip title="Retroceder 10 segundos"><span><IconButton aria-label="Retroceder 10 segundos" disabled={playbackDisabled} onClick={() => { video.current?.pause(); seek(time - 10) }}><Replay10Icon /></IconButton></span></Tooltip>
          <Tooltip title={playing ? 'Pausar' : 'Reproducir'}><span><IconButton aria-label={playing ? 'Pausar' : 'Reproducir'} disabled={playbackDisabled} onClick={() => { if (video.current?.paused) void video.current.play().catch((e: Error) => setError(e.message)); else video.current?.pause() }}>{playing ? <PauseIcon /> : <PlayArrowIcon />}</IconButton></span></Tooltip>
          <Tooltip title="Avanzar 10 segundos"><span><IconButton aria-label="Avanzar 10 segundos" disabled={playbackDisabled} onClick={() => { video.current?.pause(); seek(time + 10) }}><Forward10Icon /></IconButton></span></Tooltip>
          <Typography aria-label="Tiempo del vídeo" variant="body2" sx={{ minWidth: 112, fontVariantNumeric: 'tabular-nums' }}>{formatMediaTime(time)} / {formatMediaTime(duration)}</Typography>
          {frameRender && <Button onClick={() => { setFrameRender(null); generation.current++ }}>Volver al vídeo</Button>}
        </Stack><MatchTimeline time={time} duration={duration} fps={source?.fps} disabled={playbackDisabled} eventsDisabled={!canSeekEvent} events={events.events} onSeek={seek} onEvent={seekEvent} />
      </Box>
      <Box minWidth={0}><Typography variant="subtitle1">Campo</Typography>{context ? <PitchSvg length={context.field_length} width={context.field_width} points={pitchPoints(points)} /> : <Alert severity="info">Falta la calibración.</Alert>}
        {mode === 'tracking' || mode === 'interactive' ? mode === 'tracking' && selectionEnabled && context && !context.tracking_geometry.available ? <Typography variant="caption">Medición no disponible.</Typography> : null : <FormControl size="small" sx={{ mt: 1 }}><InputLabel shrink>Modo de clic / hito</InputLabel><Select displayEmpty renderValue={(value) => value ? context?.landmarks.find((item) => item.id === value)?.name : 'Punto libre'} label="Modo de clic / hito" value={landmark} onChange={(e) => setLandmark(e.target.value)}><MenuItem value="">Punto libre</MenuItem>{context?.landmarks.map((item) => <MenuItem key={item.id} value={item.id}>{item.name} · {item.x.toFixed(2)}, {item.z.toFixed(2)} m</MenuItem>)}</Select></FormControl>}
        <Button onClick={() => { setPoints([]); generation.current++ }}>Borrar puntos</Button>
      </Box>
    </Box>
    {events.error && <Alert severity="warning">{events.error}</Alert>}
    <EventsPanel events={events.events} total={events.total} canSeek={canSeekEvent} loading={events.loading} hasMore={events.next !== null} onMore={events.loadMore} onEvent={seekEvent} />
    {points.length > 0 && <><Divider sx={{ my: 1 }} /><Accordion disableGutters><AccordionSummary>Mediciones · {points.length} puntos</AccordionSummary><AccordionDetails><ValidationTable points={points} /></AccordionDetails></Accordion></>}
    {highlights && <Dialog open={highlightsOpen} onClose={() => { highlightsVideo.current?.pause(); setHighlightsOpen(false) }} fullWidth maxWidth="lg"><DialogTitle>Jugadas destacadas</DialogTitle><DialogContent><video ref={highlightsVideo} controls preload="metadata" src={api.url(highlights.url)} style={{ width: '100%', maxHeight: '70vh', background: '#080a0c' }} /></DialogContent><DialogActions><Button onClick={() => { highlightsVideo.current?.pause(); setHighlightsOpen(false) }}>Cerrar</Button></DialogActions></Dialog>}
    </>}
  </Box>
}
