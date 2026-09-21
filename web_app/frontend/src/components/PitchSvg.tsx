import { Box } from '@mui/material'
import type { GroundPoint } from '../types'

interface Props { length: number; width: number; points: GroundPoint[] }

const Point = ({ x, z, size, color, diamond = false }: { x: number; z: number; size: number; color: string; diamond?: boolean }) => diamond
  ? <rect x={x - size / 2} y={z - size / 2} width={size} height={size} fill={color} stroke="#0e1721" strokeWidth="0.5" transform={`rotate(45 ${x} ${z})`} />
  : <circle cx={x} cy={z} r={size / 2} fill={color} stroke="#0e1721" strokeWidth="0.5" />

/** Visual pitch only; consumes canonical field coordinates from all four views. */
export default function PitchSvg({ length, width, points }: Props) {
  const halfL = length / 2
  const halfW = width / 2
  const margin = 7
  const measured = points.filter((point) => point.measured_x !== null && point.measured_z !== null)
  const minX = Math.min(-halfL - margin, ...measured.map((point) => point.measured_x! - margin))
  const maxX = Math.max(halfL + margin, ...measured.map((point) => point.measured_x! + margin))
  const minZ = Math.min(-halfW - margin, ...measured.map((point) => point.measured_z! - margin))
  const maxZ = Math.max(halfW + margin, ...measured.map((point) => point.measured_z! + margin))
  const viewBox = `${minX} ${-maxZ} ${maxX - minX} ${maxZ - minZ}`
  const box = (x: number, z: number, w: number, h: number) => <rect x={x} y={-z - h / 2} width={w} height={h} fill="none" />
  return <Box>
    <svg viewBox={viewBox} width="100%" style={{ display: 'block', background: '#17643d', borderRadius: 6, aspectRatio: `${length}/${width}` }} aria-label="Referencia del campo">
      <g transform="scale(1 -1)" fill="none" stroke="#f2fff6" strokeWidth="0.55" vectorEffect="non-scaling-stroke">
        <rect x={-halfL} y={-halfW} width={length} height={width} />
        <line x1="0" y1={-halfW} x2="0" y2={halfW} />
        <circle cx="0" cy="0" r="9.15" />
        {box(-halfL, 0, 16.5, 40.32)} {box(halfL - 16.5, 0, 16.5, 40.32)}
        {box(-halfL, 0, 5.5, 18.32)} {box(halfL - 5.5, 0, 5.5, 18.32)}
      </g>
      <g transform="scale(1 -1)" fill="#f2fff6"><circle cx="0" cy="0" r="1" /><circle cx={-halfL + 11} cy="0" r="1" /><circle cx={halfL - 11} cy="0" r="1" /></g>
      <g transform="scale(1 -1)">{points.map((point, index) => <g key={`${index}-${point.pixel_x}`}>
        {point.expected_x !== null && point.expected_z !== null && point.measured_x !== null && point.measured_z !== null && <line x1={point.expected_x} y1={point.expected_z} x2={point.measured_x} y2={point.measured_z} stroke="#7fc7ff" strokeWidth="0.6" vectorEffect="non-scaling-stroke" />}
        {point.expected_x !== null && point.expected_z !== null && <Point x={point.expected_x} z={point.expected_z} size={3.5} color="#65c7ff" diamond />}
        {point.measured_x !== null && point.measured_z !== null && <Point x={point.measured_x} z={point.measured_z} size={3.5} color={point.status === 'inside_field' ? '#ffcf4a' : '#ff953f'} />}
      </g>)}</g>
      {points.map((point, index) => point.measured_x !== null && point.measured_z !== null && <text key={index} x={point.measured_x + 2} y={-point.measured_z - 2} fontSize="3" fill="white" stroke="#123" strokeWidth="0.15" paintOrder="stroke">{point.point_index ?? index + 1}</text>)}
    </svg>
  </Box>
}
