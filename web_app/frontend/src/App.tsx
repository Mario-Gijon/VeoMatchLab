import HomeIcon from '@mui/icons-material/Home'
import { AppBar, Box, Button, CircularProgress, Toolbar, Typography } from '@mui/material'
import { useEffect, useState } from 'react'
import { api } from './api/client'
import LibraryPage from './pages/LibraryPage'
import WorkspacePage from './pages/WorkspacePage'
import type { Calibration, MatchWorkspace } from './types'

export default function App() {
  const [matches, setMatches] = useState<MatchWorkspace[]>([])
  const [calibrations, setCalibrations] = useState<Calibration[]>([])
  const [workspaceId, setWorkspaceId] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const refresh = async () => { try { const [nextMatches, nextCalibrations] = await Promise.all([api.matches(), api.calibrations()]); setMatches(nextMatches); setCalibrations(nextCalibrations) } catch (reason) { setError(reason instanceof Error ? reason.message : 'No se pudo cargar la biblioteca.') } finally { setReady(true) } }
  useEffect(() => {
    if (workspaceId) return
    void refresh()
    const timer = window.setInterval(() => void refresh(), 4000)
    return () => window.clearInterval(timer)
  }, [workspaceId])
  return <><AppBar position="sticky" elevation={0} sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper' }}><Toolbar variant="dense"><Typography variant="subtitle1" fontWeight={800} sx={{ flexGrow: 1 }}>Veo Match Lab</Typography>{workspaceId && <Button color="inherit" size="small" startIcon={<HomeIcon />} onClick={() => setWorkspaceId(null)}>Partidos</Button>}</Toolbar></AppBar>{!ready ? <Box sx={{ minHeight: '50vh', display: 'grid', placeItems: 'center' }}><CircularProgress /></Box> : error ? <Box sx={{ p: 4 }}><Typography color="error">{error}</Typography><Button onClick={() => { setError(''); setReady(false); void refresh() }}>Reintentar</Button></Box> : workspaceId ? <WorkspacePage key={workspaceId} matchId={workspaceId} calibrations={calibrations} onBack={() => setWorkspaceId(null)} /> : <LibraryPage matches={matches} calibrations={calibrations} onOpen={setWorkspaceId} onRefresh={refresh} />}</>
}
