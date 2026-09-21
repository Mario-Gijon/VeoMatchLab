import { useMemo, useRef, useState } from 'react'
import { Box, Slider } from '@mui/material'
import type { MatchEvent } from '../types/matchData'
import { eventKey, orderEvents } from './matchEvents'
import { formatMediaTime } from './mediaTime'

interface Props {
  time: number; duration: number; fps?: number | null; disabled: boolean; eventsDisabled: boolean
  events: MatchEvent[]; onSeek: (time: number) => void; onEvent: (event: MatchEvent) => void
}

export default function MatchTimeline({ time, duration, fps, disabled, eventsDisabled, events, onSeek, onEvent }: Props) {
  const group = useRef<SVGGElement>(null)
  const [focused, setFocused] = useState('')
  const visible = useMemo(() => orderEvents(events.filter((event) => event.start_s >= 0 && event.start_s <= duration)), [events, duration])
  const tabKey = visible.some((event) => eventKey(event) === focused) ? focused : visible[0] && eventKey(visible[0])
  return <Box>
    <Slider aria-label="Línea de tiempo del partido" value={Math.min(time, duration)} max={duration || 1} step={fps ? 1 / fps : .04} disabled={disabled} onChange={(_, value) => onSeek(value as number)} />
    {visible.length > 0 && duration > 0 && <svg width="100%" height="42" viewBox="-8 0 1016 42" preserveAspectRatio="none" role="group" aria-label="Jugadas en la línea de tiempo. Usa las flechas para recorrerlas y Entrar para saltar.">
      <g ref={group}>
        {visible.map((event, index) => {
          const key = eventKey(event)
          return <g key={key} role="button" tabIndex={!eventsDisabled && key === tabKey ? 0 : -1} aria-disabled={eventsDisabled} aria-label={`${event.label} · ${formatMediaTime(event.start_s)}`}
            style={{ cursor: eventsDisabled ? 'default' : 'pointer', opacity: eventsDisabled ? .4 : 1 }}
            onFocus={() => setFocused(key)} onClick={() => !eventsDisabled && onEvent(event)}
            onKeyDown={(input) => {
              if (eventsDisabled) return
              if (input.key === 'Enter' || input.key === ' ') { input.preventDefault(); onEvent(event); return }
              const next = input.key === 'ArrowRight' ? Math.min(index + 1, visible.length - 1) : input.key === 'ArrowLeft' ? Math.max(index - 1, 0) : input.key === 'Home' ? 0 : input.key === 'End' ? visible.length - 1 : null
              if (next !== null) { input.preventDefault(); group.current?.querySelectorAll<SVGElement>('[role="button"]')[next]?.focus() }
            }}>
            <title>{event.label} · {formatMediaTime(event.start_s)}</title>
            <rect x={event.start_s / duration * 1000 - 7} y={event.type === 'goal' ? 0 : 20} width={14} height={20} fill="transparent" />
            <circle cx={event.start_s / duration * 1000} cy={event.type === 'goal' ? 10 : 30} r={5} fill={event.type === 'goal' ? '#ffd166' : '#75c9b3'} stroke={focused === key ? '#fff' : '#101815'} strokeWidth={focused === key ? 2 : 1} />
          </g>
        })}
      </g>
    </svg>}
  </Box>
}
