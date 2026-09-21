export type DataSource = 'veo' | 'our_analysis' | 'manual'
export interface MatchEvent {
  id: string
  match_id: string
  source: DataSource
  source_dataset?: string
  run_id?: string
  type: string
  label: string
  start_s: number
  duration_s?: number
  end_s?: number
  team?: string
  player?: string
  x?: number
  z?: number
  confidence?: number
  thumbnail_url?: string
}
export interface EventPage { items: MatchEvent[]; total: number; next_offset: number | null }
export interface DatasetInventory {
  datasets: { id: string; source: DataSource; name: string; status: 'observed' | 'empty' | 'forbidden' | 'unavailable' | 'not_observed'; event_count: number }[]
  capabilities: Record<string, boolean>
}
