import { useEffect, useState } from 'react'
import { matchDataApi } from '../api/matchData'
import { eventKey, orderEvents } from '../components/matchEvents'
import type { MatchEvent } from '../types/matchData'

export function useMatchEvents(matchId: string, revision?: string) {
  const [offset, setOffset] = useState(0)
  const [state, setState] = useState<{ matchId: string; events: MatchEvent[]; total: number; next: number | null; loading: boolean; error: string }>({ matchId, events: [], total: 0, next: null, loading: true, error: '' })
  useEffect(() => { setOffset(0); setState({ matchId, events: [], total: 0, next: null, loading: true, error: '' }) }, [matchId, revision])
  useEffect(() => {
    const controller = new AbortController()
    setState((previous) => ({ ...previous, loading: true, error: '' }))
    matchDataApi.events(matchId, offset, controller.signal).then((page) => {
      if (controller.signal.aborted) return
      setState((previous) => {
        const rows = offset === 0 || previous.matchId !== matchId ? page.items : [...previous.events, ...page.items]
        return { matchId, events: orderEvents([...new Map(rows.map((event) => [eventKey(event), event])).values()]), total: page.total, next: page.next_offset, loading: false, error: '' }
      })
    }).catch(() => {
      if (!controller.signal.aborted) setState((previous) => ({ ...previous, loading: false, error: 'No se pudieron cargar las jugadas del partido.' }))
    })
    return () => controller.abort()
  }, [matchId, revision, offset])
  return { ...state, events: state.matchId === matchId ? state.events : [], loadMore: () => { if (state.next !== null) setOffset(state.next) } }
}
