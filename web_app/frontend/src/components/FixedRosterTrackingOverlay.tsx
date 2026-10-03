import { useEffect, useMemo, useRef, useState } from 'react'
import type { ViewContext } from '../types'
import type { TrackingIdentityOverlayArtifact } from '../types/trackingIdentityOverlay'
import { initialCamera, interactiveTransform, projectGroundPoint } from './interactiveCamera'
import type { InteractiveCamera } from './interactiveCamera'
import { projectGroundPointToUltra } from './trackingProjection'
import { activeIdentityObservations, identityOverlayStyle } from './trackingIdentityOverlayData'

interface Props {
  artifact: TrackingIdentityOverlayArtifact
  mode: 'ultra' | 'interactive'
  context: ViewContext
  cameraRef: { current: InteractiveCamera | null }
  time: number
  fps?: number | null
}

export default function FixedRosterTrackingOverlay({ artifact, mode, context, cameraRef, time, fps }: Props) {
  const surface = useRef<SVGSVGElement>(null)
  const [, redraw] = useState(0)
  const observations = useMemo(() => activeIdentityObservations(artifact.tracks, time, fps), [artifact.tracks, fps, time])
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
      const projection = mode === 'ultra'
        ? projectGroundPointToUltra(context, observation.x, observation.z)
        : aspect > 0 ? projectGroundPoint(interactiveTransform(context, camera), camera.fov, aspect, observation.x, observation.z) : null
      if (!projection) return null
      const style = identityOverlayStyle(observation.identityId, observation.assignmentStatus)
      const label = observation.assignmentStatus === 'unassigned' ? 'U' : observation.identityId.slice(-2)
      return <g key={observation.trackletId}><circle cx={projection.u} cy={projection.v} r=".009" fill={style.color} stroke="#101820" strokeWidth=".002" strokeDasharray={style.dash} /><text x={projection.u} y={projection.v - .014} textAnchor="middle" fontSize=".024" fontWeight="700" fill={style.color} stroke="#101820" strokeWidth=".003" paintOrder="stroke">{label}</text></g>
    })}
  </svg>
}
