export function stackedLensBox(lens: { lens: string; x_min: number; y_min: number; x_max: number; y_max: number }) {
  const offset = lens.lens.toLowerCase() === 'right' ? 1024 : 0
  return { left: lens.x_min / 2048 * 100, top: (lens.y_min + offset) / 2048 * 100, width: (lens.x_max - lens.x_min) / 2048 * 100, height: (lens.y_max - lens.y_min) / 2048 * 100 }
}

export function nearestSample<T extends { timestamp_s: number }>(rows: T[], time: number) {
  return rows.reduce<T | null>((best, row) => !best || Math.abs(row.timestamp_s - time) < Math.abs(best.timestamp_s - time) ? row : best, null)
}

export function measurementTrack<T extends { measurement_id?: string | null; track_id: number }>(tracks: T[] | undefined, frameIndex: number, measurementId: number | undefined) {
  return tracks?.find((track) => track.measurement_id === `${frameIndex}:${measurementId}`) ?? null
}
