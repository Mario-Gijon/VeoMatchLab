/** Display-to-source geometry shared by the native video and canvas viewers. */
export function naturalPixel(clientX: number, clientY: number, bounds: { left: number; top: number; width: number; height: number }, width: number, height: number) {
  return {
    x: Math.max(0, Math.min(width - 1, Math.floor((clientX - bounds.left) * width / bounds.width))),
    y: Math.max(0, Math.min(height - 1, Math.floor((clientY - bounds.top) * height / bounds.height))),
  }
}

export function stackedPixel(x: number, y: number, crop: string) {
  return { x, y: y + (crop === 'right' ? 1024 : 0) }
}

/** Maps a click in a possibly letterboxed video element to native video pixels. */
export function nativeVideoPixel(clientX: number, clientY: number, bounds: { left: number; top: number; width: number; height: number }, nativeWidth: number, nativeHeight: number) {
  if (![clientX, clientY, bounds.left, bounds.top, bounds.width, bounds.height, nativeWidth, nativeHeight].every(Number.isFinite)
    || bounds.width <= 0 || bounds.height <= 0 || nativeWidth <= 0 || nativeHeight <= 0) return null
  const scale = Math.min(bounds.width / nativeWidth, bounds.height / nativeHeight)
  const drawnWidth = nativeWidth * scale
  const drawnHeight = nativeHeight * scale
  const offsetX = (bounds.width - drawnWidth) / 2
  const offsetY = (bounds.height - drawnHeight) / 2
  const x = (clientX - bounds.left - offsetX) / scale
  const y = (clientY - bounds.top - offsetY) / scale
  if (x < 0 || y < 0 || x >= nativeWidth || y >= nativeHeight) return null
  return { x: Math.floor(x), y: Math.floor(y) }
}
