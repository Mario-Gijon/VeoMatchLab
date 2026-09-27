import { useEffect, useMemo, useState } from 'react'
import { Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Divider, LinearProgress, Stack, TextField, Typography } from '@mui/material'
import { api } from '../api/client'
import { formatMediaTime } from './mediaTime'
import type { Clip, ClipJob, MediaItem } from '../types'

interface Props {
  matchId: string
  source: MediaItem | undefined
  time: number
  clips: Clip[]
  onChanged: () => void
}

export default function ClipsPanel({ matchId, source, time, clips, onChanged }: Props) {
  const [open, setOpen] = useState(false)
  const [workspace, setWorkspace] = useState<Clip | null>(null)
  const [start, setStart] = useState(0)
  const [end, setEnd] = useState(10)
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [jobs, setJobs] = useState<Record<string, ClipJob>>({})
  const activeJobs = useMemo(() => Object.values(jobs).filter((job) => ['queued', 'running'].includes(job.status)), [jobs])

  useEffect(() => {
    if (!activeJobs.length) return
    const timer = window.setInterval(() => {
      activeJobs.forEach((job) => api.clipJob(job.job_id).then((next) => {
        setJobs((items) => ({ ...items, [next.job_id]: next }))
        if (next.status === 'completed') onChanged()
      }).catch((reason: Error) => setError(reason.message)))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [activeJobs, onChanged])

  const openCreate = () => { setStart(time); setEnd(Math.min((source?.duration_s ?? time + 10), time + 10)); setName(''); setError(''); setOpen(true) }
  const create = async () => {
    if (!source) return
    try { await api.createClip(matchId, source.id, start, end, name); setOpen(false); onChanged() } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo crear el clip.') }
  }
  const startJob = async (clip: Clip, action: 'all' | 'tracking') => {
    try { const job = await api.startClipAnalysis(matchId, clip.id, action); setJobs((items) => ({ ...items, [job.job_id]: job })) } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo iniciar el análisis.') }
  }

  return <Box sx={{ mt: 2, mb: 2 }}>
    <Stack direction="row" alignItems="center" gap={1}><Typography variant="h6">Clips</Typography><Button size="small" variant="contained" disabled={!source} onClick={openCreate}>Crear clip</Button></Stack>
    {error && <Alert sx={{ mt: 1 }} severity="error" onClose={() => setError('')}>{error}</Alert>}
    <Stack spacing={1} mt={1}>
      {clips.length === 0 && <Typography variant="body2" color="text.secondary">Aún no hay clips. Un clip referencia una ventana del vídeo dual_lens original.</Typography>}
      {clips.map((clip) => {
        const job = Object.values(jobs).find((item) => item.clip_id === clip.id && ['queued', 'running'].includes(item.status))
        return <Box key={clip.id} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.25 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={1}>
            <Box><Typography variant="subtitle2">{clip.name || `${formatMediaTime(clip.start_s)} → ${formatMediaTime(clip.end_s)}`}</Typography><Typography variant="caption">{formatMediaTime(clip.start_s)} → {formatMediaTime(clip.end_s)} · {formatMediaTime(clip.end_s - clip.start_s)} · {clip.analysis?.status === 'completed' ? 'Analizado' : 'Sin analizar'}</Typography></Box>
            <Stack direction="row" gap={1}><Button size="small" onClick={() => setWorkspace(clip)}>Abrir</Button><Button size="small" variant="outlined" onClick={() => void startJob(clip, 'all')}>Ejecutar todo</Button><Button size="small" disabled={!clip.analysis?.detection_artifact_id} onClick={() => void startJob(clip, 'tracking')}>Rehacer tracking</Button></Stack>
          </Stack>
          {job && <Box mt={1}><Typography variant="caption">{job.stage} · {job.processed_frames} / {job.total_frames} frames</Typography><LinearProgress variant="determinate" value={Math.min(100, job.progress * 100)} />{job.error && <Alert severity="error">{job.error}</Alert>}</Box>}
        </Box>
      })}
    </Stack>
    <Dialog open={open} onClose={() => setOpen(false)} fullWidth maxWidth="sm"><DialogTitle>Crear clip virtual</DialogTitle><DialogContent><Stack spacing={2} mt={1}><TextField label="Nombre opcional" value={name} onChange={(event) => setName(event.target.value)} /><TextField label="Inicio (s)" type="number" value={start} onChange={(event) => setStart(Number(event.target.value))} /><TextField label="Fin (s)" type="number" value={end} onChange={(event) => setEnd(Number(event.target.value))} /><Stack direction="row" gap={1}><Button onClick={() => setStart(time)}>Usar tiempo actual como inicio</Button><Button onClick={() => setEnd(time)}>Usar tiempo actual como fin</Button></Stack><Typography variant="body2">Duración: {formatMediaTime(Math.max(0, end - start))}</Typography></Stack></DialogContent><DialogActions><Button onClick={() => setOpen(false)}>Cancelar</Button><Button variant="contained" onClick={() => void create()} disabled={!source || end <= start}>Crear</Button></DialogActions></Dialog>
    <Dialog open={workspace !== null} onClose={() => setWorkspace(null)} fullWidth maxWidth="lg"><DialogTitle>{workspace?.name || 'Clip workspace'} · {workspace && `${formatMediaTime(workspace.start_s)} → ${formatMediaTime(workspace.end_s)}`}</DialogTitle><DialogContent><Typography variant="body2" mb={1}>Fuente original y resultados de desarrollo del clip.</Typography>{source && workspace && <video controls src={`${api.url(source.url)}#t=${workspace.start_s},${workspace.end_s}`} style={{ width: '100%', maxHeight: '42vh', background: '#080a0c' }} />}{workspace?.debug_video_url && <><Divider sx={{ my: 2 }} /><Typography variant="subtitle2">Tracking debug</Typography><video controls src={api.url(workspace.debug_video_url)} style={{ width: '100%', maxHeight: '55vh', background: '#080a0c' }} /></>}</DialogContent><DialogActions><Button onClick={() => setWorkspace(null)}>Cerrar</Button></DialogActions></Dialog>
  </Box>
}
