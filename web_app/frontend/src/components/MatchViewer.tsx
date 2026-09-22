import { Alert, Box } from '@mui/material'
import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import type { GroundPoint, MediaItem, ViewContext } from '../types'
import { api } from '../api/client'
import { createPanoramaGL } from './panoramaGL'
import { dragCamera, initialCamera, interactiveTransform, isDrag, wheelCamera } from './interactiveCamera'
import type { InteractiveCamera } from './interactiveCamera'
import { ultraPanoramaSize } from './workspaceGeometry'
import type { ViewMode } from './workspaceGeometry'
import { naturalPixel, nativeVideoPixel, stackedPixel } from './viewCoordinates'

export type { ViewMode } from './workspaceGeometry'
export interface ViewerPick { width: number; height: number; timestamp: number; camera?: InteractiveCamera }
interface Props {
  media: MediaItem; mode: ViewMode; crop: string; context: ViewContext | null
  width: number; videoRef: RefObject<HTMLVideoElement | null>; initialTime: number
  cameraRef: RefObject<InteractiveCamera | null>
  onTime: (value: number) => void; onPlayback: (value: boolean) => void
  points: GroundPoint[]; selectionEnabled: boolean
  onClick: (x: number, y: number, pick: ViewerPick) => void; trackingAvailable?: boolean
}

export default function MatchViewer({ media, mode, crop, context, width, videoRef, initialTime, cameraRef, onTime, onPlayback, points, selectionEnabled, onClick, trackingAvailable }: Props) {
  // One decoder, one persistent WebGL canvas and one RAW 2D canvas per source.
  const canvas = useRef<HTMLCanvasElement>(null)
  const raw = useRef<HTMLCanvasElement>(null)
  const magnifier = useRef<HTMLCanvasElement>(null)
  const surface = useRef<HTMLDivElement>(null)
  const zoom = useRef(4)
  const pointer = useRef<{ x: number; y: number } | null>(null)
  const gesture = useRef<{ id: number; x: number; y: number; camera: InteractiveCamera; dragged: boolean } | null>(null)
  const rendered = useRef<{ camera: InteractiveCamera; width: number; height: number; timestamp: number } | null>(null)
  const [error, setError] = useState('')
  const [hover, setHover] = useState(false)
  const panoramaSize = mode === 'ultra_panorama' ? ultraPanoramaSize(context) : null
  const outputWidth = mode === 'panorama' ? width : mode === 'ultra_panorama' ? panoramaSize!.width : 2048
  const outputHeight = mode === 'panorama' ? width * 9 / 16 : mode === 'ultra_panorama' ? panoramaSize!.height : crop === 'stacked' ? 2048 : 1024
  const live = useRef({ mode, crop, width, outputWidth, outputHeight })
  useEffect(() => { live.current = { mode, crop, width, outputWidth, outputHeight }; gesture.current = null; pointer.current = null; setHover(false) }, [mode, crop, width, outputWidth, outputHeight])
  const sourceUrl = api.url(media.url)

  useEffect(() => {
    const element = surface.current
    if (!element) return
    const wheel = (event: WheelEvent) => {
      event.preventDefault()
      if (live.current.mode === 'interactive' && cameraRef.current) {
        const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? element.clientHeight : 1)
        cameraRef.current = wheelCamera(cameraRef.current, delta)
        gesture.current = null
      } else zoom.current = Math.max(2, Math.min(10, zoom.current + (event.deltaY < 0 ? 1 : -1)))
    }
    element.addEventListener('wheel', wheel, { passive: false })
    return () => element.removeEventListener('wheel', wheel)
  }, [cameraRef])

  useEffect(() => {
    setError('')
    const destination = canvas.current
    const rawCanvas = raw.current
    if (!destination || !rawCanvas) return
    let gpu: ReturnType<typeof createPanoramaGL> | undefined
    let animation = 0
    let uploadedTime = -1
    let lastDraw = ''
    let size = { width: 1, height: 1 }
    const observer = new ResizeObserver(() => {
      const bounds = destination.getBoundingClientRect()
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
      size = { width: Math.max(1, Math.round(bounds.width * ratio)), height: Math.max(1, Math.round(bounds.height * ratio)) }
    })
    observer.observe(destination)
    const draw = () => {
      try {
        const video = videoRef.current
        const view = live.current
        if (video && video.readyState >= 2 && !video.seeking && view.mode !== 'tracking') {
          const interactive = view.mode === 'interactive'
          const camera = cameraRef.current ?? (context ? initialCamera(context) : null)
          if (camera && !cameraRef.current) cameraRef.current = camera
          const w = interactive ? size.width : view.mode === 'panorama' || view.mode === 'ultra_panorama' ? view.outputWidth : 2048
          const h = interactive ? size.height : view.mode === 'panorama' || view.mode === 'ultra_panorama' ? view.outputHeight : view.crop === 'stacked' ? 2048 : 1024
          const target = view.mode === 'dual_lens' ? rawCanvas : destination
          if (target.width !== w || target.height !== h) { target.width = w; target.height = h }
          const signature = `${view.mode}/${view.crop}/${w}/${h}/${video.currentTime}/${interactive ? JSON.stringify(camera) : ''}`
          if (signature !== lastDraw) {
            if (view.mode === 'dual_lens') rawCanvas.getContext('2d')?.drawImage(video, 0, view.crop === 'right' ? 1024 : 0, 2048, h, 0, 0, w, h)
            else if (context && camera) {
              gpu ??= createPanoramaGL(destination, context, true)
              gpu.draw(video, interactive ? { matrix: interactiveTransform(context, camera), fov: camera.fov } : undefined, uploadedTime !== video.currentTime)
              uploadedTime = video.currentTime
              if (interactive) rendered.current = { camera: { ...camera }, width: w, height: h, timestamp: video.currentTime }
            }
            lastDraw = signature
          }
          const cursor = pointer.current, glass = magnifier.current
          const ctx = glass?.getContext('2d')
          if (!interactive && cursor && ctx && glass) {
            ctx.fillStyle = '#080a0c'; ctx.fillRect(0, 0, 180, 180)
            ctx.imageSmoothingEnabled = false
            ctx.drawImage(target, 90 - (cursor.x + .5) * zoom.current, 90 - (cursor.y + .5) * zoom.current, w * zoom.current, h * zoom.current)
            ctx.strokeStyle = 'white'; ctx.beginPath(); ctx.moveTo(90, 0); ctx.lineTo(90, 180); ctx.moveTo(0, 90); ctx.lineTo(180, 90); ctx.stroke()
            ctx.fillStyle = '#101216'; ctx.fillRect(0, 158, 180, 22)
            ctx.fillStyle = 'white'; ctx.font = '12px sans-serif'; ctx.fillText(`${zoom.current}× · ${cursor.x}, ${cursor.y}`, 6, 173)
          }
        }
        animation = requestAnimationFrame(draw)
      } catch (reason) {
        videoRef.current?.pause()
        setError(reason instanceof Error ? reason.message : 'No se puede renderizar este vídeo')
      }
    }
    animation = requestAnimationFrame(draw)
    return () => { cancelAnimationFrame(animation); observer.disconnect(); gpu?.dispose(); rendered.current = null }
  }, [context, videoRef, cameraRef, sourceUrl])

  const canPick = () => !!videoRef.current && !videoRef.current.seeking && videoRef.current.readyState >= 2 && !error
  return <Box sx={{ minWidth: 0, maxWidth: '100%' }}>
    {error && <Alert severity="error">{error}</Alert>}
    <Box sx={{ position: 'relative', display: mode === 'tracking' ? 'block' : 'none', lineHeight: 0 }}>
    <video ref={videoRef} crossOrigin="anonymous" src={sourceUrl} preload="metadata" playsInline
      style={{ display: mode === 'tracking' ? 'block' : 'none', width: '100%', maxHeight: '65vh', objectFit: 'contain', objectPosition: 'center', background: '#080a0c' }}
      onClick={(event) => {
        const video = videoRef.current
        if (!selectionEnabled || mode !== 'tracking' || !trackingAvailable || !video || !canPick()) return
        const pixel = nativeVideoPixel(event.clientX, event.clientY, video.getBoundingClientRect(), video.videoWidth, video.videoHeight)
        if (!pixel) return
        video.pause(); onTime(video.currentTime)
        onClick(pixel.x, pixel.y, { width: video.videoWidth, height: video.videoHeight, timestamp: video.currentTime })
      }}
      onLoadedMetadata={(event) => { event.currentTarget.currentTime = Math.min(initialTime, Math.max(0, event.currentTarget.duration - .001)) }}
      onTimeUpdate={(event) => onTime(event.currentTarget.currentTime)}
      onPlay={() => onPlayback(true)} onPause={() => onPlayback(false)} onEnded={() => onPlayback(false)}
      onError={() => setError('No se puede reproducir el formato de este vídeo.')} />
    </Box>
    <Box ref={surface} sx={{ display: mode === 'tracking' ? 'none' : 'block', position: 'relative', width: '100%', lineHeight: 0,
      cursor: mode === 'interactive' ? (selectionEnabled ? 'crosshair' : 'grab') : selectionEnabled ? 'crosshair' : 'default', touchAction: mode === 'interactive' ? 'none' : 'auto' }}
      onPointerDown={(event) => {
        if (mode !== 'interactive' || !event.isPrimary || event.button !== 0 || !cameraRef.current || gesture.current) return
        event.currentTarget.setPointerCapture(event.pointerId)
        gesture.current = { id: event.pointerId, x: event.clientX, y: event.clientY, camera: { ...cameraRef.current }, dragged: false }
      }}
      onPointerMove={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect()
        if (mode === 'interactive') {
          const start = gesture.current
          if (!start || event.pointerId !== start.id) return
          const dx = event.clientX - start.x, dy = event.clientY - start.y
          start.dragged ||= isDrag(dx, dy)
          if (start.dragged) cameraRef.current = dragCamera(start.camera, dx, dy, bounds.width, bounds.height)
          return
        }
        pointer.current = naturalPixel(event.clientX, event.clientY, bounds, outputWidth, outputHeight)
        if (magnifier.current) {
          magnifier.current.style.left = `${Math.max(0, Math.min(bounds.width - 180, event.clientX - bounds.left + 20))}px`
          magnifier.current.style.top = `${Math.max(0, Math.min(bounds.height - 180, event.clientY - bounds.top + 20))}px`
        }
        setHover(true)
      }}
      onPointerUp={(event) => {
        const start = gesture.current
        if (mode !== 'interactive' || !start || start.id !== event.pointerId) return
        gesture.current = null
        event.currentTarget.releasePointerCapture(event.pointerId)
        if (!selectionEnabled || start.dragged || isDrag(event.clientX - start.x, event.clientY - start.y) || !canPick()) return
        const frame = rendered.current
        if (!frame) return
        const pixel = nativeVideoPixel(event.clientX, event.clientY, canvas.current!.getBoundingClientRect(), frame.width, frame.height)
        if (!pixel) return
        videoRef.current!.pause(); onTime(videoRef.current!.currentTime)
        onClick(pixel.x, pixel.y, frame)
      }}
      onPointerCancel={() => { gesture.current = null }} onLostPointerCapture={() => { gesture.current = null }}
      onPointerLeave={() => { pointer.current = null; setHover(false) }}
      onClick={(event) => {
        if (!selectionEnabled || mode === 'interactive' || !canPick()) return
        videoRef.current!.pause(); onTime(videoRef.current!.currentTime)
        const pixel = naturalPixel(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect(), outputWidth, outputHeight)
        const source = mode === 'dual_lens' ? stackedPixel(pixel.x, pixel.y, crop) : pixel
        onClick(source.x, source.y, { width: outputWidth, height: mode === 'dual_lens' ? 2048 : outputHeight, timestamp: videoRef.current!.currentTime })
      }}>
      <canvas ref={canvas} aria-label={mode === 'interactive' ? 'Vista interactiva' : mode === 'ultra_panorama' ? 'Vista ultra panorámica' : 'Vista panorámica'}
        style={{ width: '100%', display: mode === 'dual_lens' ? 'none' : 'block', background: '#080a0c',
          aspectRatio: mode === 'ultra_panorama' ? `${outputWidth} / ${outputHeight}` : '16 / 9', height: mode === 'interactive' ? 'auto' : undefined, maxHeight: mode === 'interactive' ? '65vh' : undefined }} />
      <canvas ref={raw} aria-label="Vista de doble lente" width={2048} height={crop === 'stacked' ? 2048 : 1024}
        style={{ width: '100%', display: mode === 'dual_lens' ? 'block' : 'none', background: '#080a0c' }} />
      {selectionEnabled && mode !== 'interactive' && <svg aria-label={mode === 'dual_lens' ? 'Marcadores de doble lente' : mode === 'ultra_panorama' ? 'Marcadores ultra panorámicos' : 'Marcadores panorámicos'} viewBox={`0 0 ${outputWidth} ${outputHeight}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        {points.map((point) => {
          const y = point.pixel_y - (mode === 'dual_lens' && crop === 'right' ? 1024 : 0)
          if (y < 0 || y >= outputHeight) return null
          return <g key={point.point_index}><circle cx={point.pixel_x + .5} cy={y + .5} r={outputWidth / 140} fill="#ffd258" stroke="#000" /><text x={point.pixel_x + .5} y={y + .5} textAnchor="middle" dominantBaseline="central" fontSize={outputWidth / 90} fill="#000">{point.point_index}</text></g>
        })}
      </svg>}
      <canvas ref={magnifier} width={180} height={180} style={{ display: hover && mode !== 'interactive' ? 'block' : 'none', position: 'absolute', pointerEvents: 'none', border: '1px solid #eee', width: 180, height: 180 }} />
    </Box>
  </Box>
}
