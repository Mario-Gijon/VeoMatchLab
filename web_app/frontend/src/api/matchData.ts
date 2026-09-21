import { request } from './client'
import type { DatasetInventory, EventPage } from '../types/matchData'

export const matchDataApi = {
  datasets: (id: string) => request<DatasetInventory>(`/api/matches/${id}/datasets`),
  events: (id: string, offset = 0, signal?: AbortSignal) => request<EventPage>(`/api/matches/${id}/events?offset=${offset}&limit=500`, { signal }),
}
