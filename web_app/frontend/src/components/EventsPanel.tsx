import { Box, Button, Typography } from '@mui/material'
import type { MatchEvent } from '../types/matchData'
import { eventKey } from './matchEvents'
import { formatMediaTime } from './mediaTime'

interface Props { events: MatchEvent[]; total: number; canSeek: boolean; loading: boolean; hasMore: boolean; onMore: () => void; onEvent: (event: MatchEvent) => void }

export default function EventsPanel({ events, total, canSeek, loading, hasMore, onMore, onEvent }: Props) {
  if (!events.length) return null
  return <Box component="section" aria-label="Eventos del partido" sx={{ mt: 1, p: 1, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
    <Typography variant="subtitle2">{events.every((event) => event.source_dataset === 'highlights') ? 'Jugadas destacadas' : 'Eventos'} · {total}</Typography>
    {!canSeek && <Typography variant="caption">Activa una vista disponible para saltar a la jugada.</Typography>}
    <Box sx={{ maxHeight: 220, overflowY: 'auto', display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' }, gap: .5, mt: .5 }}>
      {events.map((event) => <button key={eventKey(event)} type="button" disabled={!canSeek} onClick={() => onEvent(event)}
        style={{ display: 'flex', alignItems: 'center', gap: 10, textAlign: 'left', color: 'inherit', background: 'transparent', border: '1px solid #53645d', borderRadius: 4, padding: 6, cursor: canSeek ? 'pointer' : 'default' }}>
        {event.thumbnail_url && <img src={event.thumbnail_url} alt="" loading="lazy" referrerPolicy="no-referrer" onError={(error) => { error.currentTarget.hidden = true }} style={{ width: 70, height: 40, objectFit: 'cover', borderRadius: 2 }} />}
        <span><span style={{ display: 'block' }}>{event.label}</span><span style={{ fontSize: 12, fontVariantNumeric: 'tabular-nums' }}>{formatMediaTime(event.start_s)}{event.duration_s !== undefined && ` · ${event.duration_s} s`} · {event.source === 'veo' ? 'Veo' : event.source === 'manual' ? 'Manual' : 'Análisis propio'}</span></span>
      </button>)}
    </Box>
    {hasMore && <Button size="small" disabled={loading} onClick={onMore}>{loading ? 'Cargando…' : `Mostrar más (${events.length} de ${total})`}</Button>}
  </Box>
}
