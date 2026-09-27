import { useEffect, useRef, useState } from 'react'
import type { ViewContext } from '../types'
import { initialCamera, interactiveTransform } from './interactiveCamera'
import type { InteractiveCamera } from './interactiveCamera'
import { presentationDirectionFromVirtualPixel, projectVirtualPixelToUltra, projectVirtualRayToInteractive, type VirtualCameraSpec } from './trackingProjection'
import { representativeObservation, type OverlayAssociationSample } from './trackingOverlayData'

export interface OverlayTrack { track_id: number; x: number; z: number; position_kind: string; measurement_id?: string | null }
interface Props { mode: 'ultra' | 'interactive'; context: ViewContext | null; cameraRef: { current: InteractiveCamera | null }; tracks: OverlayTrack[]; association: OverlayAssociationSample[]; cameraSpecs: Record<string, VirtualCameraSpec> }

export default function TrackingOverlay({ mode, context, cameraRef, tracks, association, cameraSpecs }: Props) {
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
    const representative = representativeObservation(track, association)
    const spec = representative ? cameraSpecs[representative.view_name] : null
    const foot = representative ? { x: (representative.xyxy[0] + representative.xyxy[2]) / 2, y: representative.xyxy[3] } : null
    const direction = spec && foot ? presentationDirectionFromVirtualPixel(spec, foot.x, foot.y, 1280, 720) : null
    const projected = mode === 'ultra'
      ? context && bounds && spec && foot ? projectVirtualPixelToUltra(context, spec, foot.x, foot.y, 1280, 720) : null
      : context && camera && direction && aspect > 0 ? projectVirtualRayToInteractive(interactiveTransform(context, camera), camera.fov, direction, aspect) : null
    return projected ? { track, ...projected } : null
  }).filter((value): value is { track: OverlayTrack; u: number; v: number } => value !== null)
  return <svg ref={surface} viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
    {points.map(({ track, u, v }) => <g key={track.track_id}>
      <circle cx={u} cy={v} r="0.003" fill="#f6cf4a" stroke="#101820" strokeWidth="0.001" />
      <line x1={u} y1={v} x2={u} y2={Math.min(1, v + 0.009)} stroke="#f6cf4a" strokeWidth="0.001" />
      <text x={u} y={Math.min(1, v + 0.020)} textAnchor="middle" fontSize="0.015" fill="#fff" stroke="#101820" strokeWidth="0.0015" paintOrder="stroke">{track.track_id}</text>
    </g>)}
  </svg>
}
