import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Divider, LinearProgress, MenuItem, Stack, TextField, Typography } from '@mui/material'
import { useState } from 'react'
import { api } from '../api/client'
import type { Calibration, MatchWorkspace, MediaItem } from '../types'

const resourceRoles = [['tracking', 'Seguimiento de balón'], ['dual_lens', 'Doble lente'], ['camera_directions', 'Direcciones de cámara'], ['calibration', 'Calibración'], ['clip', 'Clip'], ['image', 'Imagen']] as const
export default function ManageFilesDialog({ match, calibrations, onClose, onUpdated, onImage, onRender }: {
  match: MatchWorkspace; calibrations: Calibration[]; onClose: () => void; onUpdated: (match: MatchWorkspace) => void
  onImage?: (item: MediaItem) => void; onRender?: (item: MediaItem) => void
}) {
  const [role, setRole] = useState('dual_lens')
  const [path, setPath] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [confirmation, setConfirmation] = useState<{ title: string; action: () => Promise<unknown> } | null>(null)
  const run = async (action: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try { await action(); onUpdated(await api.match(match.id)); setPath('') }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudieron actualizar los archivos.') }
    finally { setBusy(false); setConfirmation(null) }
  }
  const replace = (action: () => Promise<unknown>) => {
    const exists = role === 'calibration' ? !!match.calibration : ['tracking', 'dual_lens'].includes(role) && match.media.some((item) => item.role === role)
    if (exists) setConfirmation({ title: `¿Sustituir ${resourceRoles.find(([id]) => id === role)?.[1]}?`, action })
    else void run(action)
  }
  const upload = (file: File) => replace(() => role === 'calibration' ? api.uploadCalibration(match.id, file) : api.uploadMedia(match.id, file, role))
  const reference = (value: string) => replace(() => role === 'calibration' ? api.importCalibrationPath(match.id, value) : api.importMediaPath(match.id, value, role))
  return <Dialog open onClose={busy ? undefined : onClose} fullWidth maxWidth="sm">
    <DialogTitle>Gestionar archivos · {match.name}</DialogTitle><DialogContent><Stack gap={2} mt={1}>
      {error && <Alert severity="error">{error}</Alert>}{busy && <LinearProgress />}
      {resourceRoles.slice(0, 4).map(([id, label]) => {
        const item = id === 'calibration' ? match.calibration : match.media.find((value) => value.role === id)
        return <Stack key={id} direction="row" alignItems="center" gap={1}><Box flex={1} minWidth={0}><Typography variant="subtitle2">{label}</Typography><Typography variant="body2" noWrap color="text.secondary">{item?.name || 'Falta'}</Typography></Box>{item && <Button size="small" disabled={busy} onClick={() => setConfirmation({ title: `¿Eliminar ${label} del partido?`, action: () => id === 'calibration' ? api.removeCalibration(match.id) : api.removeMedia(item.id) })}>Eliminar</Button>}</Stack>
      })}
      {match.media.filter((item) => !['tracking', 'dual_lens'].includes(item.role || '')).map((item) => <Stack direction="row" key={item.id} gap={1} alignItems="center"><Typography variant="body2" noWrap flex={1}>{item.name}</Typography>{['image', 'frame'].includes(item.kind) && <><Button size="small" onClick={() => onImage ? onImage(item) : window.open(api.url(item.url), '_blank', 'noopener')}>Ver</Button>{onRender && item.width === 2048 && item.height === 2048 && <Button size="small" disabled={!match.calibration} onClick={() => onRender(item)}>Renderizar fotograma</Button>}</>}<Button size="small" disabled={busy} onClick={() => setConfirmation({ title: `¿Eliminar ${item.name}?`, action: () => api.removeMedia(item.id) })}>Eliminar</Button></Stack>)}
      <Divider />
      {match.media.some((item) => ['video', 'clip'].includes(item.kind)) && <Accordion disableGutters elevation={0}><AccordionSummary>Roles de vídeo</AccordionSummary><AccordionDetails><Stack gap={1}>{match.media.filter((item) => ['video', 'clip'].includes(item.kind)).map((item) => <TextField key={item.id} select size="small" label={item.name} value={item.role || 'unknown'} disabled={busy} onChange={(e) => { const value = e.target.value; setConfirmation({ title: '¿Cambiar el rol del vídeo? Se sustituirá el archivo actual.', action: () => api.setRole(item.id, value) }) }}><MenuItem value="unknown">Sin asignar</MenuItem><MenuItem value="tracking">Seguimiento de balón</MenuItem><MenuItem value="dual_lens">Doble lente</MenuItem><MenuItem value="clip">Clip</MenuItem></TextField>)}</Stack></AccordionDetails></Accordion>}
      <Stack direction="row" gap={1}><TextField select size="small" label="Añadir o sustituir" value={role} disabled={busy} onChange={(e) => setRole(e.target.value)} sx={{ flex: 1 }}>{resourceRoles.map(([id, label]) => <MenuItem key={id} value={id}>{label}</MenuItem>)}</TextField><Button variant="contained" component="label" disabled={busy}>Elegir archivo<input hidden type="file" accept={role === 'calibration' ? '.veo' : role === 'image' ? 'image/*' : role === 'camera_directions' ? '.det' : 'video/*'} onChange={(e) => { const file = e.target.files?.[0]; if (file) upload(file); e.target.value = '' }} /></Button></Stack>
      <Accordion disableGutters elevation={0}><AccordionSummary>Vincular archivo local</AccordionSummary><AccordionDetails><Stack gap={1}><Typography variant="caption" color="text.secondary">El archivo original no se mueve ni se elimina.</Typography><TextField size="small" label="Ubicación del archivo" value={path} onChange={(e) => setPath(e.target.value)} /><Button disabled={!path || busy} onClick={() => reference(path)}>Vincular</Button>{role === 'calibration' && <TextField select size="small" label="Calibración guardada" value="" onChange={(e) => reference(e.target.value)}>{calibrations.map((item) => <MenuItem key={item.id} value={item.path}>{item.name}</MenuItem>)}</TextField>}</Stack></AccordionDetails></Accordion>
    </Stack></DialogContent><DialogActions><Button onClick={onClose} disabled={busy}>Listo</Button></DialogActions>
    <Dialog open={!!confirmation} onClose={busy ? undefined : () => setConfirmation(null)}><DialogTitle>{confirmation?.title}</DialogTitle><DialogContent>Las copias se eliminarán. Los archivos originales permanecerán.</DialogContent><DialogActions><Button disabled={busy} onClick={() => setConfirmation(null)}>Cancelar</Button><Button disabled={busy} variant="contained" onClick={() => confirmation && void run(confirmation.action)}>Confirmar</Button></DialogActions></Dialog>
  </Dialog>
}
