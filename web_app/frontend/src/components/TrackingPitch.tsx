import { trackingPitchPoint, trackingPitchViewBox } from './trackingPitchGeometry'

export interface TrackingPitchTrack { track_id: number; x: number; z: number }
interface Props { fieldLength: number; fieldWidth: number; tracks: TrackingPitchTrack[]; onTrackSelect: (trackId: number) => void }

/** Compact SVG pitch for the raw tracks.person sample. */
export default function TrackingPitch({ fieldLength, fieldWidth, tracks, onTrackSelect }: Props) {
  const halfLength = fieldLength / 2
  const halfWidth = fieldWidth / 2
  const penaltyDepth = 16.5
  const penaltyWidth = 40.32
  const goalAreaDepth = 5.5
  const goalAreaWidth = 18.32
  const point = (x: number, z: number) => trackingPitchPoint(x, z)
  const markerRadius = Math.max(fieldLength, fieldWidth) / 140
  return <svg viewBox={trackingPitchViewBox(fieldLength, fieldWidth)} width="100%" role="img" aria-label="Campo 2D del tracking" style={{ display: 'block', aspectRatio: `${fieldLength}/${fieldWidth}`, background: '#17643d', borderRadius: 6 }}>
    <g fill="none" stroke="#f2fff6" strokeWidth="0.55" vectorEffect="non-scaling-stroke">
      <rect x={-halfLength} y={-halfWidth} width={fieldLength} height={fieldWidth} />
      <line x1="0" y1={-halfWidth} x2="0" y2={halfWidth} />
      <circle cx="0" cy="0" r="9.15" />
      <rect x={-halfLength} y={-penaltyWidth / 2} width={penaltyDepth} height={penaltyWidth} />
      <rect x={halfLength - penaltyDepth} y={-penaltyWidth / 2} width={penaltyDepth} height={penaltyWidth} />
      <rect x={-halfLength} y={-goalAreaWidth / 2} width={goalAreaDepth} height={goalAreaWidth} />
      <rect x={halfLength - goalAreaDepth} y={-goalAreaWidth / 2} width={goalAreaDepth} height={goalAreaWidth} />
    </g>
    <g fill="#f2fff6"><circle cx="0" cy="0" r="1" /><circle cx={-halfLength + 11} cy="0" r="1" /><circle cx={halfLength - 11} cy="0" r="1" /></g>
    <g>
      {tracks.map((track) => {
        const position = point(track.x, track.z)
        return <g key={track.track_id} role="button" tabIndex={0} aria-label={`Track ${track.track_id}`} onClick={() => onTrackSelect(track.track_id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onTrackSelect(track.track_id) }} style={{ cursor: 'pointer' }}>
          <circle cx={position.x} cy={position.y} r={markerRadius} fill="#f6cf4a" stroke="#0e1721" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
          <text x={position.x} y={position.y} textAnchor="middle" dominantBaseline="central" fontSize={markerRadius * 2.2} fill="#101820" paintOrder="stroke" stroke="#f6cf4a" strokeWidth="0.2">{track.track_id}</text>
        </g>
      })}
    </g>
  </svg>
}
