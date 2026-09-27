export function stackedLensBox(lens: { lens: string; x_min: number; y_min: number; x_max: number; y_max: number }) {
  const offset = lens.lens.toLowerCase() === 'right' ? 1024 : 0
  return { left: lens.x_min / 2048 * 100, top: (lens.y_min + offset) / 2048 * 100, width: (lens.x_max - lens.x_min) / 2048 * 100, height: (lens.y_max - lens.y_min) / 2048 * 100 }
}

export function nearestSample<T extends { timestamp_s: number }>(rows: T[], time: number) {
  return rows.reduce<T | null>((best, row) => !best || Math.abs(row.timestamp_s - time) < Math.abs(best.timestamp_s - time) ? row : best, null)
}

export function frameSampleIndex<T extends { frame_index?: number }>(rows: T[]) {
  const result = new Map<number, T>()
  for (const row of rows) if (Number.isInteger(row.frame_index)) result.set(row.frame_index!, row)
  return result
}

/** The backend uses round(time * source_fps) when it reads physical frames. */
export function sampleAtMediaTime<T extends { timestamp_s: number; frame_index?: number }>(rows: T[], byFrame: Map<number, T>, mediaTime: number, sourceFps: number | null | undefined) {
  if (Number.isFinite(mediaTime) && sourceFps && sourceFps > 0) {
    const exact = byFrame.get(Math.round(mediaTime * sourceFps))
    if (exact) return exact
  }
  return nearestSample(rows, mediaTime)
}

export function measurementTrack<T extends { measurement_id?: string | null; track_id: number }>(tracks: T[] | undefined, frameIndex: number, measurementId: number | undefined) {
  return tracks?.find((track) => track.measurement_id === `${frameIndex}:${measurementId}`) ?? null
}
