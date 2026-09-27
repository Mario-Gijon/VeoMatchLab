export interface PitchPoint { x: number; y: number }

/** Maps canonical world coordinates to the SVG plane (SVG y points down). */
export function trackingPitchPoint(x: number, z: number): PitchPoint {
  return { x, y: z === 0 ? 0 : -z }
}

export function trackingPitchViewBox(length: number, width: number, margin = 1.5) {
  return `${-length / 2 - margin} ${-width / 2 - margin} ${length + margin * 2} ${width + margin * 2}`
}
