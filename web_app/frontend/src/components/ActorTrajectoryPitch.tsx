import type { ActiveActorTrajectoryPoint } from './actorTrajectoryData'
import { identityColor } from './trackingIdentityOverlayData'
import { trackingPitchPoint, trackingPitchViewBox } from './trackingPitchGeometry'

interface Props { fieldLength: number; fieldWidth: number; points: ActiveActorTrajectoryPoint[] }

export default function ActorTrajectoryPitch({ fieldLength, fieldWidth, points }: Props) {
  const halfLength = fieldLength / 2, halfWidth = fieldWidth / 2, markerRadius = Math.max(fieldLength, fieldWidth) / 140
  return <svg viewBox={trackingPitchViewBox(fieldLength, fieldWidth)} width="100%" role="img" aria-label="Campo 2D Actor trajectories 7I.7" style={{ display: 'block', aspectRatio: `${fieldLength}/${fieldWidth}`, background: '#17643d', borderRadius: 6 }}>
    <g fill="none" stroke="#f2fff6" strokeWidth=".55" vectorEffect="non-scaling-stroke"><rect x={-halfLength} y={-halfWidth} width={fieldLength} height={fieldWidth} /><line x1="0" y1={-halfWidth} x2="0" y2={halfWidth} /><circle cx="0" cy="0" r="9.15" /><rect x={-halfLength} y="-20.16" width="16.5" height="40.32" /><rect x={halfLength - 16.5} y="-20.16" width="16.5" height="40.32" /></g>
    {points.map((point) => {
      const position = trackingPitchPoint(point.x, point.z), color = identityColor(point.actorId)
      return <g key={point.actorId}><title>{point.actorId} · {point.origin}</title><circle cx={position.x} cy={position.y} r={markerRadius} fill={color} stroke="#0e1721" strokeWidth=".5" vectorEffect="non-scaling-stroke" /><text x={position.x} y={position.y} textAnchor="middle" dominantBaseline="central" fontSize={markerRadius * 2.2} fill="#101820" paintOrder="stroke" stroke={color} strokeWidth=".2">{point.actorId.slice(-2)}</text></g>
    })}
  </svg>
}
