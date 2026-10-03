import { useMemo } from 'react'
import type { ActorTrajectoryArtifact } from '../types/actorTrajectory'
import { activeActorTrajectoryPoints, sourceBboxForActorPoint } from './actorTrajectoryData'
import { identityColor } from './trackingIdentityOverlayData'

interface Props { artifact: ActorTrajectoryArtifact; time: number; fps?: number | null }

export default function ActorTrajectorySourceOverlay({ artifact, time, fps }: Props) {
  const points = useMemo(() => activeActorTrajectoryPoints(artifact.actors, time, fps), [artifact.actors, fps, time])
  return <svg aria-label="Actor trajectories 7I.7" viewBox="0 0 2048 2048" preserveAspectRatio="xMidYMid meet" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
    {points.map((point) => {
      const bbox = sourceBboxForActorPoint(point)
      if (!bbox) return null
      const [x1, y1, x2, y2] = bbox, color = identityColor(point.actorId)
      return <g key={point.actorId}><title>{point.actorId} · {point.origin}</title><rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={color} strokeWidth={4} vectorEffect="non-scaling-stroke" /><text x={x1} y={Math.max(14, y1 - 5)} fill={color} stroke="#111" strokeWidth={3} paintOrder="stroke" fontSize={24} fontFamily="ui-monospace, monospace" fontWeight={700}>{point.actorId.slice(-2)}</text></g>
    })}
  </svg>
}
