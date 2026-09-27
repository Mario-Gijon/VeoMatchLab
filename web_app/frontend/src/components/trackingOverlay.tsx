import { useEffect, useRef, useState } from 'react'
import type { ViewContext } from '../types'
import { initialCamera, interactiveTransform, projectGroundPoint } from './interactiveCamera'
import type { InteractiveCamera } from './interactiveCamera'
import { projectGroundPointToUltra } from './trackingProjection'

export interface OverlayTrack { track_id: number; x: number; z: number }
interface Props { mode: 'ultra' | 'interactive'; context: ViewContext | null; cameraRef: { current: InteractiveCamera | null }; tracks: OverlayTrack[] }

export default function TrackingOverlay({ mode, context, cameraRef, tracks }: Props) {
  const surface = useRef<SVGSVGElement>(null)
  const [, redraw] = useState(0)
  useEffect(() => {
    if (mode !== 'interactive') return
    let frame = 0
    const tick = () => { redraw((value) => value + 1); frame = requestAnimationFrame(tick) }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [mode])
  const bounds = surface.current?.getBoundingClientRect()
  const camera = mode === 'interactive' ? cameraRef.current ?? (context ? initialCamera(context) : null) : null
  const aspect = bounds && bounds.height > 0 ? bounds.width / bounds.height : 0
  const points = tracks.map((track) => {
    const projected = mode === 'ultra'
      ? context && bounds ? projectGroundPointToUltra(context, track.x, track.z, bounds.width, bounds.height) : null
      : context && camera && aspect > 0 ? projectGroundPoint(interactiveTransform(context, camera), camera.fov, aspect, track.x, track.z) : null
    return projected ? { track, ...projected } : null
  }).filter((value): value is { track: OverlayTrack; u: number; v: number } => value !== null)
  return <svg ref={surface} viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
    {points.map(({ track, u, v }) => <g key={track.track_id}>
      <circle cx={u} cy={v} r="0.006" fill="#f6cf4a" stroke="#101820" strokeWidth="0.002" />
      <line x1={u} y1={v} x2={u} y2={Math.min(1, v + 0.018)} stroke="#f6cf4a" strokeWidth="0.002" />
      <text x={u} y={Math.min(1, v + 0.038)} textAnchor="middle" fontSize="0.035" fill="#fff" stroke="#101820" strokeWidth="0.004" paintOrder="stroke">{track.track_id}</text>
    </g>)}
  </svg>
}
