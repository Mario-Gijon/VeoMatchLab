export function clampMediaTime(value: number | null | undefined, duration: number | null | undefined) {
  const limit = Number.isFinite(duration) ? Math.max(0, duration as number) : 0
  const current = Number.isFinite(value) ? Math.max(0, value as number) : 0
  return Math.min(current, limit)
}

export function formatMediaTime(value: number | null | undefined) {
  const seconds = Number.isFinite(value) ? Math.max(0, Math.floor(value as number)) : 0
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor(seconds / 60) % 60
  const remainder = seconds % 60
  return [hours, minutes, remainder].map((part) => String(part).padStart(2, '0')).join(':')
}
