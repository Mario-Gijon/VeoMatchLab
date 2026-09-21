import type { MatchEvent } from '../types/matchData'

// Source/run remain part of identity: two providers can describe the same play.
export const eventKey = (event: MatchEvent) => `${event.source}:${event.source_dataset ?? ''}:${event.run_id ?? ''}:${event.id}`
export const orderEvents = (events: MatchEvent[]) => [...events].sort((a, b) => a.start_s - b.start_s || eventKey(a).localeCompare(eventKey(b)))

export function filterEvents(events: MatchEvent[], filter: Partial<Pick<MatchEvent, 'type' | 'source' | 'team' | 'player'>>) {
  return events.filter((event) => Object.entries(filter).every(([key, value]) => value === undefined || event[key as keyof MatchEvent] === value))
}

export function adjacentEvent(events: MatchEvent[], time: number, direction: -1 | 1) {
  const ordered = orderEvents(events)
  return direction === 1 ? ordered.find((event) => event.start_s > time) : ordered.reverse().find((event) => event.start_s < time)
}
