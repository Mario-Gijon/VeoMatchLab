import { useMemo } from 'react'
import type { TrackingIdentityOverlayArtifact } from '../types/trackingIdentityOverlay'
import { activeIdentityObservations, identityOverlayStyle } from './trackingIdentityOverlayData'

interface Props {
  artifact: TrackingIdentityOverlayArtifact
  time: number
  showTrackletId?: boolean
  showLens?: boolean
  markInterpolation?: boolean
}

export default function TrackingIdentityOverlay({ artifact, time, showTrackletId = false, showLens = false, markInterpolation = false }: Props) {
  const observations = useMemo(() => activeIdentityObservations(artifact.tracks, time), [artifact.tracks, time])
  return <svg aria-label="IDs Tracking V1" viewBox={`0 0 ${artifact.source_width} ${artifact.source_height}`} preserveAspectRatio="xMidYMid meet" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
    {observations.map((observation) => {
      const [x1, y1, x2, y2] = observation.bbox
      const style = identityOverlayStyle(observation.identityId, observation.assignmentStatus)
      const color = style.color
      const label = [observation.identityId, showTrackletId ? observation.trackletId : null, showLens ? observation.source_lens : null].filter(Boolean).join(' · ')
      return <g key={observation.trackletId}>
        <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={color} strokeWidth={4} vectorEffect="non-scaling-stroke" strokeDasharray={style.dash ?? (markInterpolation && observation.interpolated ? '6 4' : undefined)} />
        <text x={x1} y={Math.max(14, y1 - 5)} fill={color} stroke="#111" strokeWidth={3} paintOrder="stroke" fontSize={24} fontFamily="ui-monospace, monospace" fontWeight={700}>{label}</text>
      </g>
    })}
  </svg>
}
