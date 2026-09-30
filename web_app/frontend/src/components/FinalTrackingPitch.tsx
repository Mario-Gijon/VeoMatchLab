import { Box } from '@mui/material'
import { trackingPitchPoint, trackingPitchViewBox } from './trackingPitchGeometry'
import { pointCoordinates, trajectorySegments } from './trackingFinalDebug'
import type { FinalPoint, FinalPositionMode, SegmentBreak } from '../types/trackingFinal'

interface Props {
  fieldLength: number
  fieldWidth: number
  points: FinalPoint[]
  selectedIdentityId: string | null
  selectedTrajectory: FinalPoint[]
  segmentBreaks: SegmentBreak[]
  mode: FinalPositionMode
  onSelect: (identityId: string) => void
}

const originStyle = (origin: string) => origin === 'fused'
  ? { fill: '#69d7ff', shape: 'diamond' }
  : origin === 'interpolated' ? { fill: '#ff9e46', shape: 'square' } : { fill: '#f6cf4a', shape: 'circle' }

/** SVG field keeps canonical metres in the viewBox; only SVG flips the Z axis. */
export default function FinalTrackingPitch({ fieldLength, fieldWidth, points, selectedIdentityId, selectedTrajectory, segmentBreaks, mode, onSelect }: Props) {
  const halfLength = fieldLength / 2, halfWidth = fieldWidth / 2
  const marker = Math.max(fieldLength, fieldWidth) / 130
  const selectedSegments = trajectorySegments(selectedTrajectory, segmentBreaks)
  const xy = (point: FinalPoint) => trackingPitchPoint(pointCoordinates(point, mode).x, pointCoordinates(point, mode).z)
  return <Box sx={{ width: '100%' }}><svg viewBox={trackingPitchViewBox(fieldLength, fieldWidth)} width="100%" role="img" aria-label="Campo Tracking Debug" style={{ display: 'block', aspectRatio: `${fieldLength}/${fieldWidth}`, background: '#17643d', borderRadius: 6 }}>
    <g fill="none" stroke="#f2fff6" strokeWidth="0.55" vectorEffect="non-scaling-stroke">
      <rect x={-halfLength} y={-halfWidth} width={fieldLength} height={fieldWidth} /><line x1="0" y1={-halfWidth} x2="0" y2={halfWidth} /><circle cx="0" cy="0" r="9.15" />
      <rect x={-halfLength} y={-20.16} width="16.5" height="40.32" /><rect x={halfLength - 16.5} y={-20.16} width="16.5" height="40.32" />
      <rect x={-halfLength} y={-9.16} width="5.5" height="18.32" /><rect x={halfLength - 5.5} y={-9.16} width="5.5" height="18.32" />
    </g>
    {selectedSegments.map((segment, index) => segment.length > 1 && <polyline key={`${segment[0].segment_id}-${index}`} points={segment.map((point) => { const position = xy(point); return `${position.x},${position.y}` }).join(' ')} fill="none" stroke="#ffffff" strokeWidth="1.1" strokeOpacity=".9" vectorEffect="non-scaling-stroke" data-testid="trajectory-segment" />)}
    {points.map((point) => {
      const position = xy(point), style = originStyle(point.origin), selected = point.identity_id === selectedIdentityId
      return <g key={`${point.identity_id}-${point.frame_index}`} role="button" tabIndex={0} aria-label={`Identity ${point.identity_id}`} onClick={() => onSelect(point.identity_id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onSelect(point.identity_id) }} style={{ cursor: 'pointer' }}>
        {style.shape === 'diamond' ? <rect x={position.x - marker / 1.4} y={position.y - marker / 1.4} width={marker * 1.4} height={marker * 1.4} transform={`rotate(45 ${position.x} ${position.y})`} fill={style.fill} stroke={selected ? '#fff' : '#0e1721'} strokeWidth={selected ? 1 : .45} vectorEffect="non-scaling-stroke" /> : style.shape === 'square' ? <rect x={position.x - marker / 2} y={position.y - marker / 2} width={marker} height={marker} rx=".15" fill={style.fill} stroke={selected ? '#fff' : '#0e1721'} strokeWidth={selected ? 1 : .45} vectorEffect="non-scaling-stroke" /> : <circle cx={position.x} cy={position.y} r={marker / 2} fill={style.fill} stroke={selected ? '#fff' : '#0e1721'} strokeWidth={selected ? 1 : .45} vectorEffect="non-scaling-stroke" />}
        <text x={position.x + marker * .7} y={position.y - marker * .55} fontSize={marker * 1.15} fill="#fff" stroke="#102a1e" strokeWidth=".22" paintOrder="stroke">{point.identity_id}</text>
      </g>
    })}
  </svg></Box>
}
