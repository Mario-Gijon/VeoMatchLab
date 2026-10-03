import { useEffect, useRef, useState } from 'react'
import type { ViewContext } from '../types'
import type { InteractiveCamera } from './interactiveCamera'
import { initialCamera, interactiveTransform } from './interactiveCamera'
import type { ActiveActorTrajectoryPoint } from './actorTrajectoryData'
import { presentationDirectionForActorPoint } from './actorTrajectoryData'
import { identityColor } from './trackingIdentityOverlayData'
import { projectPresentationDirectionToUltra, projectVirtualRayToInteractive, type VirtualCameraSpec } from './trackingProjection'

interface Props { points: ActiveActorTrajectoryPoint[]; mode: 'ultra' | 'interactive'; context: ViewContext; cameraRef: { current: InteractiveCamera | null }; cameraSpecs: Record<string, VirtualCameraSpec> }

export default function ActorTrajectoryOverlay({ points, mode, context, cameraRef, cameraSpecs }: Props) {
  const surface = useRef<SVGSVGElement>(null), [, redraw] = useState(0)
  useEffect(() => { let frame = 0; const tick = () => { redraw((value) => value + 1); if (mode === 'interactive') frame = requestAnimationFrame(tick) }; frame = requestAnimationFrame(tick); return () => cancelAnimationFrame(frame) }, [mode])
  const bounds = surface.current?.getBoundingClientRect(), camera = cameraRef.current ?? initialCamera(context), aspect = bounds && bounds.height > 0 ? bounds.width / bounds.height : 0
  return <svg ref={surface} viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Actor trajectories 7I.7" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
    {points.map((point) => {
      const direction = presentationDirectionForActorPoint(point, cameraSpecs)
      const projection = direction && (mode === 'ultra' ? projectPresentationDirectionToUltra(context, direction) : aspect > 0 ? projectVirtualRayToInteractive(context, interactiveTransform(context, camera), camera.fov, direction, aspect) : null)
      if (!projection) return null
      const color = identityColor(point.actorId)
      return <g key={point.actorId}><title>{point.actorId} · {point.origin} · {point.trackletIds.join(', ')}</title><circle cx={projection.u} cy={projection.v} r=".009" fill={color} stroke="#101820" strokeWidth=".002" /><text x={projection.u} y={projection.v - .014} textAnchor="middle" fontSize=".024" fontWeight="700" fill={color} stroke="#101820" strokeWidth=".003" paintOrder="stroke">{point.actorId.slice(-2)}</text></g>
    })}
  </svg>
}
