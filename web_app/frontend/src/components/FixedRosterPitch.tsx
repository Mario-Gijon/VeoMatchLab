import { trackingPitchPoint, trackingPitchViewBox } from './trackingPitchGeometry'
import type { ActiveIdentityObservation } from './trackingIdentityOverlayData'
import { identityOverlayStyle } from './trackingIdentityOverlayData'

interface Props {
  fieldLength: number
  fieldWidth: number
  observations: ActiveIdentityObservation[]
  onTrackletSelect: (trackletId: string) => void
}

export default function FixedRosterPitch({ fieldLength, fieldWidth, observations, onTrackletSelect }: Props) {
  const halfLength = fieldLength / 2, halfWidth = fieldWidth / 2
  const markerRadius = Math.max(fieldLength, fieldWidth) / 140
  const label = (observation: ActiveIdentityObservation) => observation.assignmentStatus === 'unassigned' ? 'U' : observation.identityId.slice(-2)
  return <svg viewBox={trackingPitchViewBox(fieldLength, fieldWidth)} width="100%" role="img" aria-label="Campo 2D Fixed roster" style={{ display: 'block', aspectRatio: `${fieldLength}/${fieldWidth}`, background: '#17643d', borderRadius: 6 }}>
    <g fill="none" stroke="#f2fff6" strokeWidth=".55" vectorEffect="non-scaling-stroke">
      <rect x={-halfLength} y={-halfWidth} width={fieldLength} height={fieldWidth} /><line x1="0" y1={-halfWidth} x2="0" y2={halfWidth} /><circle cx="0" cy="0" r="9.15" />
      <rect x={-halfLength} y="-20.16" width="16.5" height="40.32" /><rect x={halfLength - 16.5} y="-20.16" width="16.5" height="40.32" />
      <rect x={-halfLength} y="-9.16" width="5.5" height="18.32" /><rect x={halfLength - 5.5} y="-9.16" width="5.5" height="18.32" />
    </g>
    {observations.map((observation) => {
      const position = trackingPitchPoint(observation.x, observation.z)
      const style = identityOverlayStyle(observation.identityId, observation.assignmentStatus)
      return <g key={observation.trackletId} role="button" tabIndex={0} aria-label={`${observation.identityId} ${observation.trackletId}`} onClick={() => onTrackletSelect(observation.trackletId)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onTrackletSelect(observation.trackletId) }} style={{ cursor: 'pointer' }}>
        <title>{observation.identityId} · {observation.trackletId} · {observation.assignmentStatus}</title>
        <circle cx={position.x} cy={position.y} r={markerRadius} fill={style.color} stroke="#0e1721" strokeWidth=".5" strokeDasharray={style.dash} vectorEffect="non-scaling-stroke" />
        <text x={position.x} y={position.y} textAnchor="middle" dominantBaseline="central" fontSize={markerRadius * 2.2} fill="#101820" paintOrder="stroke" stroke={style.color} strokeWidth=".2">{label(observation)}</text>
      </g>
    })}
  </svg>
}
