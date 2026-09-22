import { Box, Chip, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material'
import type { GroundPoint } from '../types'
import { validationStatistics } from './workspaceGeometry'

const coordinate = (x: number | null, z: number | null) => x === null || z === null ? '—' : `${x.toFixed(2)}, ${z.toFixed(2)}`
const statusColor = (status: GroundPoint['status']) => status === 'inside_field' ? 'success' : status === 'outside_field' ? 'warning' : 'default'
const viewLabel = (view?: string) => ({ tracking: 'seguimiento', interactive: 'interactivo', panorama: 'panorámica', ultra_panorama: 'ultra panorámica', dual_lens: 'doble lente' }[view || 'panorama'] || view || 'panorámica')
const statusLabel = (status: GroundPoint['status']) => ({ inside_field: 'dentro del campo', outside_field: 'fuera del campo', no_ground_intersection: 'sin intersección', camera_state_unavailable: 'estado de cámara no disponible', ray_parallel_to_ground: 'rayo paralelo al suelo', invalid_ground_intersection: 'intersección no válida' }[status])

export default function ValidationTable({ points }: { points: GroundPoint[] }) {
  const stats = validationStatistics(points)
  return <Paper variant="outlined" sx={{ overflow: 'hidden' }}>
    {stats.n > 0 && <Typography variant="body2" sx={{ px: 1.5, pt: 1 }}>Hitos (n={stats.n}) · mediana {stats.median!.toFixed(3)} m · RMS {stats.rms!.toFixed(3)} m · máx. {stats.max!.toFixed(3)} m · media ΔX {stats.meanDeltaX!.toFixed(3)} m · media ΔZ {stats.meanDeltaZ!.toFixed(3)} m</Typography>}
    <Stack direction="row" justifyContent="space-between" alignItems="center" p={1.5} gap={1} flexWrap="wrap">
      <Typography variant="subtitle1" fontWeight={700}>Puntos de validación</Typography>
      <Stack direction="row" gap={.75}><Chip size="small" label={`${points.length} seleccionados`} /><Chip size="small" color="secondary" label={stats.mean === null ? 'Sin error de hito' : `Error medio ${stats.mean.toFixed(2)} m`} /></Stack>
    </Stack>
    {points.length === 0 ? <Typography color="text.secondary" sx={{ px: 1.5, pb: 2 }}>Aún no hay mediciones.</Typography> :
      <TableContainer sx={{ maxHeight: 280 }}><Table stickyHeader size="small"><TableHead><TableRow><TableCell>#</TableCell><TableCell>Vista / tiempo</TableCell><TableCell>Hito</TableCell><TableCell>Píxel / UV</TableCell><TableCell>Esperado X/Z</TableCell><TableCell>Medido X/Z</TableCell><TableCell>Δ X/Z</TableCell><TableCell>Error</TableCell><TableCell>Estado</TableCell></TableRow></TableHead>
        <TableBody>{points.map((point, index) => <TableRow key={`${index}-${point.pixel_x}-${point.pixel_y}`} hover><TableCell>{point.point_index ?? index + 1}</TableCell><TableCell>{viewLabel(point.view)}<br />{point.timestamp_s?.toFixed(3) ?? '—'} s</TableCell><TableCell>{point.landmark_name ?? 'Punto libre'}</TableCell><TableCell>({point.pixel_x}, {point.pixel_y})<br /><Box component="span" color="text.secondary">{point.u.toFixed(5)}, {point.v.toFixed(5)}</Box></TableCell><TableCell>{coordinate(point.expected_x, point.expected_z)}</TableCell><TableCell>{coordinate(point.measured_x, point.measured_z)}</TableCell><TableCell>{coordinate(point.delta_x, point.delta_z)}</TableCell><TableCell>{point.field_error_m === null ? '—' : `${point.field_error_m.toFixed(2)} m`}</TableCell><TableCell><Chip size="small" color={statusColor(point.status)} label={statusLabel(point.status)} /></TableCell></TableRow>)}</TableBody>
      </Table></TableContainer>}
    </Paper>
}
