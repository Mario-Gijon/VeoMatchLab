export type TrackingDebugView = 'original' | 'ultra' | 'interactive'

export interface DebugPresentation {
  currentTime: number
  play?: () => Promise<void>
  pause?: () => void
}

export interface DebugPresentations { original: DebugPresentation | null; ultra: DebugPresentation | null; interactive: DebugPresentation | null }

export function activePresentation(view: TrackingDebugView, presentations: DebugPresentations) {
  return presentations[view]
}

export function timeForViewChange(view: TrackingDebugView, presentations: DebugPresentations, fallback: number, duration: number) {
  const current = activePresentation(view, presentations)?.currentTime
  const value = typeof current === 'number' && Number.isFinite(current) ? current : fallback
  return Math.max(0, Math.min(duration, value))
}

/** Keep every mounted decoder at the shared local time without deriving any tracking data. */
export function synchronizePresentations(presentations: DebugPresentations, time: number) {
  for (const presentation of Object.values(presentations)) {
    if (presentation && Math.abs(presentation.currentTime - time) > 1 / 30) presentation.currentTime = time
  }
}

/** Only the active visualisation is allowed to play. */
export function controlPresentationPlayback(presentations: DebugPresentations, active: DebugPresentation | null, playing: boolean) {
  for (const presentation of Object.values(presentations)) if (presentation && presentation !== active) presentation.pause?.()
  if (!active) return
  if (playing) void active.play?.().catch(() => undefined)
  else active.pause?.()
}
