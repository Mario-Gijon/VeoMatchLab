import MoreHorizIcon from '@mui/icons-material/MoreHoriz'
import { Alert, Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Menu, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material'
import { useState } from 'react'
import { api } from '../api/client'
import ManageFilesDialog from '../components/ManageFilesDialog'
import VeoImportDialog from '../components/VeoImportDialog'
import type { Calibration, MatchWorkspace } from '../types'

interface Props { matches: MatchWorkspace[]; calibrations: Calibration[]; onRefresh: () => Promise<void>; onOpen: (id: string) => void }

export default function LibraryPage({ matches, calibrations, onRefresh, onOpen }: Props) {
  const [dialog, setDialog] = useState<'rename' | 'delete' | null>(null)
  const [veoOpen, setVeoOpen] = useState(false)
  const [retryMatch, setRetryMatch] = useState<MatchWorkspace | null>(null)
  const [selected, setSelected] = useState<MatchWorkspace | null>(null)
  const [files, setFiles] = useState<MatchWorkspace | null>(null)
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const submit = async () => {
    setSaving(true); setError('')
    try {
      if (dialog === 'delete' && selected) await api.deleteMatch(selected.id)
      else if (dialog === 'rename' && selected) await api.renameMatch(selected.id, name)
      await onRefresh(); setDialog(null)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo actualizar el partido.') }
    finally { setSaving(false) }
  }
  return <Box sx={{ maxWidth: 1440, mx: 'auto', px: { xs: 2, md: 4 }, py: 3 }}>
    <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" gap={2} mb={3}>
      <Typography variant="h5" component="h1">Partidos</Typography>
      <Button variant="contained" onClick={() => { setRetryMatch(null); setVeoOpen(true) }}>Importar desde Veo</Button>
    </Stack>
    {matches.length === 0 ? <Typography color="text.secondary" sx={{ py: 6, textAlign: 'center' }}>Aún no hay partidos.</Typography> :
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 1.5 }}>{matches.map((match) => {
        const metadata = match.metadata || {}
        const date = metadata.date && Number.isFinite(Date.parse(metadata.date)) ? new Intl.DateTimeFormat('es-ES', { dateStyle: 'medium' }).format(new Date(metadata.date)) : ''
        const details = [metadata.home_team && metadata.away_team ? `${metadata.home_team} · ${metadata.away_team}` : '', date].filter(Boolean).join(' · ')
        const coreReady = !!match.calibration && match.media.some((item) => item.role === 'tracking') && match.media.some((item) => item.role === 'dual_lens') && match.media.some((item) => item.role === 'camera_directions')
        const status = ['queued', 'downloading'].includes(match.import_status || '') ? 'Importando' : match.import_status && match.import_status !== 'completed' ? 'Revisar importación' : metadata.veo_url ? coreReady ? 'Listo' : 'Revisar archivos' : ''
        return <Paper variant="outlined" key={match.id} sx={{ p: 1.5 }}>
        <Stack direction="row" alignItems="center" gap={1}><Box flex={1} minWidth={0}><Typography variant="subtitle1" fontWeight={650} noWrap>{match.name}</Typography>{details && <Typography variant="body2" color="text.secondary" noWrap>{details}</Typography>}</Box>{status && <Chip size="small" label={status} color={status === 'Listo' ? 'success' : status === 'Importando' ? 'primary' : 'default'} variant="outlined" />}<IconButton size="small" aria-label={'Acciones de ' + match.name} onClick={(e) => { setSelected(match); setAnchor(e.currentTarget) }}><MoreHorizIcon /></IconButton></Stack>
        <Stack direction="row" justifyContent="flex-end" mt={1}><Button size="small" variant="contained" onClick={() => onOpen(match.id)}>Abrir</Button></Stack>
      </Paper>})}</Box>}
    <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}><MenuItem onClick={() => { setName(selected?.name || ''); setError(''); setDialog('rename'); setAnchor(null) }}>Renombrar</MenuItem><MenuItem onClick={() => { if (selected) void api.match(selected.id).then(setFiles).catch((e: Error) => setError(e.message)); setAnchor(null) }}>Archivos</MenuItem>{selected?.metadata?.veo_url && <MenuItem onClick={() => { setRetryMatch(selected); setVeoOpen(true); setAnchor(null) }}>Revisar importación</MenuItem>}<MenuItem sx={{ color: 'error.main' }} onClick={() => { setError(''); setDialog('delete'); setAnchor(null) }}>Eliminar partido</MenuItem></Menu>
    {error && !dialog && <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError('')}>{error}</Alert>}
    <Dialog open={!!dialog} onClose={saving ? undefined : () => setDialog(null)} fullWidth maxWidth="xs"><DialogTitle>{dialog === 'delete' ? '¿Eliminar “' + selected?.name + '”?' : 'Renombrar partido'}</DialogTitle><DialogContent><Stack gap={2} mt={1}>{error && <Alert severity="error">{error}</Alert>}{dialog === 'delete' ? <Typography variant="body2">Se eliminarán las copias. Los archivos originales permanecerán.</Typography> : <TextField autoFocus required label="Nombre del partido" value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && name.trim() && !saving) void submit() }} />}</Stack></DialogContent><DialogActions><Button disabled={saving} onClick={() => setDialog(null)}>Cancelar</Button><Button variant="contained" color={dialog === 'delete' ? 'error' : 'primary'} disabled={saving || (dialog === 'rename' && !name.trim())} onClick={() => void submit()}>{dialog === 'delete' ? 'Eliminar partido' : 'Guardar'}</Button></DialogActions></Dialog>
    {files && <ManageFilesDialog match={files} calibrations={calibrations} onUpdated={(match) => { setFiles(match); void onRefresh() }} onClose={() => { setFiles(null); void onRefresh() }} />}
    {veoOpen && <VeoImportDialog key={retryMatch?.id || 'new'} open retryMatch={retryMatch} onClose={() => { setVeoOpen(false); void onRefresh() }} onComplete={(id) => { setVeoOpen(false); onOpen(id) }} />}
  </Box>
}
