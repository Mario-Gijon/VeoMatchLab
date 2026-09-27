import { useEffect, useMemo, useState } from 'react'
import { Alert, Box, Button, LinearProgress, Stack, TextField, Typography } from '@mui/material'
import { api } from '../api/client'
import { formatMediaTime } from './mediaTime'
import type { Clip, ClipJob, MediaItem } from '../types'

interface Props { matchId: string; source?: MediaItem; time: number; clips: Clip[]; onOpen: (clip: Clip) => void; onChanged: () => void }

export default function ClipsPanel({ matchId, source, time, clips, onOpen, onChanged }: Props) {
  const [creating, setCreating] = useState(false)
  const [start, setStart] = useState(time)
  const [end, setEnd] = useState(Math.min(source?.duration_s ?? time + 10, time + 10))
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [jobs, setJobs] = useState<Record<string, ClipJob>>({})
  const activeJobs = useMemo(() => Object.values(jobs).filter((job) => ['queued', 'running'].includes(job.status)), [jobs])

  useEffect(() => {
    if (!activeJobs.length) return
    const timer = window.setInterval(() => activeJobs.forEach((job) => api.clipJob(job.job_id).then((next) => {
      setJobs((items) => ({ ...items, [next.job_id]: next }))
      if (next.status === 'completed') onChanged()
    }).catch((reason: Error) => setError(reason.message))), 1000)
    return () => window.clearInterval(timer)
  }, [activeJobs, onChanged])

  const showCreate = () => { setStart(time); setEnd(Math.min(source?.duration_s ?? time + 10, time + 10)); setName(''); setCreating(true) }
  const create = async () => {
    if (!source) return
    try { await api.createClip(matchId, source.id, start, end, name); setCreating(false); onChanged() } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo crear el clip.') }
  }
  const startJob = async (clip: Clip, action: 'all' | 'tracking') => {
    try { const job = await api.startClipAnalysis(matchId, clip.id, action); setJobs((items) => ({ ...items, [job.job_id]: job })) } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo iniciar el análisis.') }
  }
  const remove = async (clip: Clip) => {
    try { await api.deleteClip(matchId, clip.id); setConfirmDelete(null); onChanged() } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo eliminar el clip.') }
  }

  return <Box sx={{ maxWidth: 1120, mx: 'auto', py: 2 }}>
    <Stack direction="row" alignItems="center" justifyContent="space-between" gap={1} mb={2}><Typography variant="h5">Clips</Typography><Button variant="contained" disabled={!source} onClick={showCreate}>Crear clip</Button></Stack>
    {!source && <Alert severity="info">Añade un vídeo dual_lens para crear clips virtuales.</Alert>}
    {error && <Alert severity="error" sx={{ mb: 1 }} onClose={() => setError('')}>{error}</Alert>}
    {creating && <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 2, mb: 2 }}><Typography variant="subtitle1">Crear clip virtual</Typography><Stack direction={{ xs: 'column', sm: 'row' }} gap={1} mt={1}><TextField label="Nombre opcional" value={name} onChange={(event) => setName(event.target.value)} /><TextField label="Inicio (s)" type="number" value={start} onChange={(event) => setStart(Number(event.target.value))} /><TextField label="Fin (s)" type="number" value={end} onChange={(event) => setEnd(Number(event.target.value))} /></Stack><Stack direction="row" gap={1} mt={1} flexWrap="wrap"><Button size="small" onClick={() => setStart(time)}>Usar como inicio</Button><Button size="small" onClick={() => setEnd(time)}>Usar como fin</Button><Typography variant="body2" sx={{ alignSelf: 'center' }}>Duración: {formatMediaTime(Math.max(0, end - start))}</Typography><Box flex={1} /><Button onClick={() => setCreating(false)}>Cancelar</Button><Button variant="contained" disabled={end <= start} onClick={() => void create()}>Crear</Button></Stack></Box>}
    <Stack spacing={1}>
      {clips.length === 0 && <Typography color="text.secondary">Aún no hay clips para este partido.</Typography>}
      {clips.map((clip) => {
        const job = Object.values(jobs).find((item) => item.clip_id === clip.id && ['queued', 'running'].includes(item.status))
        const analysis = clip.analysis
        return <Box key={clip.id} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5 }}><Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" gap={1}><Box><Typography variant="subtitle1">{clip.name || `${formatMediaTime(clip.start_s)} → ${formatMediaTime(clip.end_s)}`}</Typography><Typography variant="body2" color="text.secondary">{formatMediaTime(clip.start_s)} → {formatMediaTime(clip.end_s)} · {formatMediaTime(clip.end_s - clip.start_s)}</Typography><Typography variant="caption">Detección: {analysis?.detection_artifact_id ? 'completada' : 'pendiente'} · Tracking: {analysis?.tracking_artifact_id ? 'completado' : 'pendiente'}</Typography></Box><Stack direction="row" gap={1} flexWrap="wrap" alignItems="center"><Button size="small" onClick={() => onOpen(clip)}>Abrir</Button><Button size="small" variant="outlined" onClick={() => void startJob(clip, 'all')}>Ejecutar todo</Button><Button size="small" disabled={!analysis?.detection_artifact_id} onClick={() => void startJob(clip, 'tracking')}>Rehacer tracking</Button><Button size="small" color="error" onClick={() => setConfirmDelete(clip.id)}>Eliminar</Button></Stack></Stack>{job && <Box mt={1}><Typography variant="caption">{job.status} · {job.stage} · {job.processed_frames} / {job.total_frames} frames</Typography><LinearProgress variant="determinate" value={Math.min(100, job.progress * 100)} />{job.error && <Alert severity="error">{job.error}</Alert>}</Box>}{confirmDelete === clip.id && <Alert severity="warning" sx={{ mt: 1 }} action={<Stack direction="row"><Button color="inherit" size="small" onClick={() => setConfirmDelete(null)}>Cancelar</Button><Button color="inherit" size="small" onClick={() => void remove(clip)}>Eliminar</Button></Stack>}>Eliminará la definición y outputs exclusivos de este clip.</Alert>}</Box>
      })}
    </Stack>
  </Box>
}
