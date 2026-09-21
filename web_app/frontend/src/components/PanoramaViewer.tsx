import { Box, Chip } from '@mui/material'
import { useEffect, useRef, useState } from 'react'
import type { GroundPoint, RenderedPanorama } from '../types'
import { api } from '../api/client'

interface Props {
  panorama: RenderedPanorama
  points: GroundPoint[]
  onPixelClick: (x: number, y: number) => void
}

interface PointerPixel { x: number; y: number; cssX: number; cssY: number }

/** Maps CSS display space to the renderer's exact natural panorama pixel. */
function cssToNaturalPixel(event: React.MouseEvent<HTMLDivElement>, element: HTMLDivElement, panorama: RenderedPanorama): PointerPixel {
  const bounds = element.getBoundingClientRect()
  const x = Math.max(0, Math.min(panorama.width - 1, Math.floor((event.clientX - bounds.left) * panorama.width / bounds.width)))
  const y = Math.max(0, Math.min(panorama.height - 1, Math.floor((event.clientY - bounds.top) * panorama.height / bounds.height)))
  return { x, y, cssX: event.clientX - bounds.left, cssY: event.clientY - bounds.top }
}

export default function PanoramaViewer({ panorama, points, onPixelClick }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [pointer, setPointer] = useState<PointerPixel | null>(null)
  const [magnification, setMagnification] = useState(4)
  useEffect(() => {
    const element = ref.current
    if (!element) return
    const wheel = (event: WheelEvent) => {
      event.preventDefault()
      setMagnification((value) => Math.max(2, Math.min(10, value + (event.deltaY < 0 ? 1 : -1))))
    }
    element.addEventListener('wheel', wheel, { passive: false })
    return () => element.removeEventListener('wheel', wheel)
  }, [])
  const imageUrl = api.url(panorama.url)

  const updatePointer = (event: React.PointerEvent<HTMLDivElement>) => {
    if (ref.current) setPointer(cssToNaturalPixel(event, ref.current, panorama))
  }
  const magnifierPosition = () => {
    if (!pointer || !ref.current) return {}
    const size = 180
    const width = ref.current.clientWidth
    const height = ref.current.clientHeight
    let left = pointer.cssX + 18
    let top = pointer.cssY + 18
    if (left + size > width) left = pointer.cssX - size - 18
    if (top + size > height) top = pointer.cssY - size - 18
    return { left: Math.max(0, Math.min(left, Math.max(0, width - size))), top: Math.max(0, Math.min(top, Math.max(0, height - size))) }
  }

  return <Box>
    <Box
      ref={ref}
      onPointerMove={updatePointer}
      onPointerLeave={() => setPointer(null)}
      onClick={(event) => { if (ref.current) { const pixel = cssToNaturalPixel(event, ref.current, panorama); onPixelClick(pixel.x, pixel.y) } }}
      sx={{ position: 'relative', overflow: 'hidden', lineHeight: 0, cursor: 'crosshair', borderRadius: 1, bgcolor: '#05080d', touchAction: 'none', '& img': { display: 'block', width: '100%', height: 'auto' } }}
    >
      <img src={imageUrl} alt="Panorámica Veo renderizada" draggable={false} />
      <svg viewBox={`0 0 ${panorama.width} ${panorama.height}`} preserveAspectRatio="none" aria-hidden="true" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        {points.map((point, index) => <g key={`${point.pixel_x}-${point.pixel_y}-${index}`}>
          <circle cx={point.pixel_x + 0.5} cy={point.pixel_y + 0.5} r="10" fill={point.status === 'inside_field' ? '#ffcf4a' : '#ff953f'} stroke="#101722" strokeWidth="2" />
          <text x={point.pixel_x + 0.5} y={point.pixel_y + 0.5} textAnchor="middle" dominantBaseline="central" fill="#101722" fontSize="13" fontWeight="700">{point.point_index ?? index + 1}</text>
        </g>)}
      </svg>
      {pointer && <Box aria-hidden="true" sx={{ position: 'absolute', width: 180, height: 180, pointerEvents: 'none', border: '2px solid #9fc1ff', boxShadow: '0 5px 20px #0009', backgroundImage: `url("${imageUrl}")`, backgroundSize: `${panorama.width * magnification}px ${panorama.height * magnification}px`, backgroundPosition: `${90 - (pointer.x + 0.5) * magnification}px ${90 - (pointer.y + 0.5) * magnification}px`, ...magnifierPosition(), '&:before, &:after': { content: '""', position: 'absolute', background: '#f5f8ff', boxShadow: '0 0 2px #000' }, '&:before': { width: 1, height: '100%', left: '50%' }, '&:after': { height: 1, width: '100%', top: '50%' } }}>
        <Chip size="small" label={`${magnification}×`} sx={{ position: 'absolute', right: 4, bottom: 4, fontSize: 11, height: 20, bgcolor: '#101722e6' }} />
      </Box>}
    </Box>
  </Box>
}
