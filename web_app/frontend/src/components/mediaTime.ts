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

export function parseMediaTime(value: string): number | null {
  const parts = value.trim().split(':')
  if (parts.length < 1 || parts.length > 3 || parts.some((part) => !/^\d+(?:\.\d+)?$/.test(part))) return null
  const numbers = parts.map(Number)
  if (!numbers.every(Number.isFinite) || numbers.some((part) => part < 0)) return null
  const seconds = numbers.pop()!
  const minutes = numbers.pop() ?? 0
  const hours = numbers.pop() ?? 0
  if (seconds >= 60 || minutes >= 60) return null
  return hours * 3600 + minutes * 60 + seconds
}
