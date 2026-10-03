import { useEffect, useRef, useState } from 'react'
import type { ViewContext } from '../types'
import { initialCamera, interactiveTransform } from './interactiveCamera'
import type { InteractiveCamera } from './interactiveCamera'
import { projectPresentationDirectionToUltra, projectVirtualRayToInteractive, type VirtualCameraSpec } from './trackingProjection'
import { identityOverlayStyle, type ActiveIdentityObservation } from './trackingIdentityOverlayData'
import { presentationDirectionForObservation } from './trackingOverlayData'

interface Props {
  observations: ActiveIdentityObservation[]
  mode: 'ultra' | 'interactive'
  context: ViewContext
  cameraRef: { current: InteractiveCamera | null }
  cameraSpecs: Record<string, VirtualCameraSpec>
}

export default function FixedRosterTrackingOverlay({ observations, mode, context, cameraRef, cameraSpecs }: Props) {
  const surface = useRef<SVGSVGElement>(null)
  const [, redraw] = useState(0)
  useEffect(() => {
    let frame = 0
    if (mode !== 'interactive') {
      frame = requestAnimationFrame(() => redraw((value) => value + 1))
      return () => cancelAnimationFrame(frame)
    }
    const tick = () => { redraw((value) => value + 1); frame = requestAnimationFrame(tick) }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [mode])
  const bounds = surface.current?.getBoundingClientRect()
  const camera = cameraRef.current ?? initialCamera(context)
  const aspect = bounds && bounds.height > 0 ? bounds.width / bounds.height : 0
  return <svg ref={surface} viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Fixed roster Tracking V1" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
    {observations.map((observation) => {
      const direction = presentationDirectionForObservation(observation.representative_view, observation.virtual_bbox, cameraSpecs)
      const projection = direction && (mode === 'ultra'
        ? projectPresentationDirectionToUltra(context, direction)
        : aspect > 0 ? projectVirtualRayToInteractive(context, interactiveTransform(context, camera), camera.fov, direction, aspect) : null)
      if (!projection) return null
      const style = identityOverlayStyle(observation.identityId, observation.assignmentStatus)
      const label = observation.assignmentStatus === 'unassigned' ? 'U' : observation.identityId.slice(-2)
      return <g key={observation.trackletId}><circle cx={projection.u} cy={projection.v} r=".009" fill={style.color} stroke="#101820" strokeWidth=".002" strokeDasharray={style.dash} /><text x={projection.u} y={projection.v - .014} textAnchor="middle" fontSize=".024" fontWeight="700" fill={style.color} stroke="#101820" strokeWidth=".003" paintOrder="stroke">{label}</text></g>
    })}
  </svg>
}
