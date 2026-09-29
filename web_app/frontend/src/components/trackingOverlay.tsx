import { useEffect, useMemo, useRef, useState } from 'react'
import type { ViewContext } from '../types'
import { initialCamera, interactiveTransform } from './interactiveCamera'
import type { InteractiveCamera } from './interactiveCamera'
import { projectPresentationDirectionToUltra, projectVirtualRayToInteractive, type VirtualCameraSpec } from './trackingProjection'
import { visualAnchorsByTrack, visualDirectionForTrack, type OverlayAssociationSample, type OverlayTrackFrame } from './trackingOverlayData'

export interface OverlayTrack { track_id: number; x: number; z: number; position_kind: string; measurement_id?: string | null }
interface Props { mode: 'ultra' | 'interactive'; context: ViewContext | null; cameraRef: { current: InteractiveCamera | null }; tracks: OverlayTrack[]; frameIndex?: number; timestamp?: number; trackFrames: OverlayTrackFrame[]; association: OverlayAssociationSample[]; cameraSpecs: Record<string, VirtualCameraSpec> }

export default function TrackingOverlay({ mode, context, cameraRef, tracks, frameIndex, timestamp, trackFrames, association, cameraSpecs }: Props) {
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
  const anchorsByTrack = useMemo(() => visualAnchorsByTrack(trackFrames, association, cameraSpecs), [trackFrames, association, cameraSpecs])
  const points = tracks.map((track) => {
    const direction = visualDirectionForTrack(track, frameIndex, timestamp, anchorsByTrack)
    const projected = mode === 'ultra'
      ? context && bounds && direction ? projectPresentationDirectionToUltra(context, direction) : null
      : context && camera && direction && aspect > 0 ? projectVirtualRayToInteractive(context, interactiveTransform(context, camera), camera.fov, direction, aspect) : null
    return projected ? { track, ...projected } : null
  }).filter((value): value is { track: OverlayTrack; u: number; v: number } => value !== null)
  return <svg ref={surface} viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
    {points.map(({ track, u, v }) => <g key={track.track_id}>
      <circle cx={u} cy={v} r="0.003" fill="#f6cf4a" stroke="#101820" strokeWidth="0.001" />
      <text x={u} y={v} textAnchor="middle" dominantBaseline="central" fontSize="0.010" fontWeight="700" fill="#ffffff" paintOrder="stroke" stroke="#101820" strokeWidth="0.0015" strokeLinejoin="round">{track.track_id}</text>
    </g>)}
  </svg>
}
